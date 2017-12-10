use faster::{IntoPackedRefIterator, f64s};
use rand::random;
use itertools::zip;
use rayon::prelude::*;

use parser::RawModel;
use data::{Class, Kernel, Problem, SVM};
use rbfkernel::RbfKernel;
use randomization::{random_vec, Randomize};
use util::{find_max_index, set_all, sum_f64s};


pub type RbfCSVM = SVM<RbfKernel>;


impl RbfCSVM {
    /// Creates a new random CSVM
    pub fn random(num_classes: usize, sv_per_class: usize, num_attributes: usize) -> RbfCSVM {
        let total_sv = num_classes * sv_per_class;
        let classes: Vec<Class> = (0..num_classes)
            .map(|class| {
                Class::with_parameters(num_classes, sv_per_class, num_attributes, class as u32)
                    .randomize()
            })
            .collect();

        RbfCSVM {
            num_attributes,
            total_support_vectors: total_sv,
            rho: random_vec(num_classes),
            kernel: RbfKernel { gamma: random() },
            classes,
        }
    }


    /// Creates a SVM from the given raw model.
    pub fn from_raw_model(raw_model: &RawModel) -> Result<RbfCSVM, &'static str> {
        let header = &raw_model.header;
        let vectors = &raw_model.vectors;

        // Get basic info
        let num_attributes = vectors[0].features.len();
        let num_classes = header.nr_class as usize;
        let total_support_vectors = header.total_sv as usize;

        // Construct vector of classes
        let classes: Vec<Class> = (0..num_classes)
            .map(|class| {
                let label = header.label[class];
                Class::with_parameters(
                    num_classes,
                    header.nr_sv[class] as usize,
                    num_attributes,
                    label,
                )
            })
            .collect();

        // Allocate model
        let mut svm = RbfCSVM {
            num_attributes,
            total_support_vectors,
            kernel: RbfKernel {
                gamma: header.gamma,
            },
            rho: header.rho.clone(),
            classes,
        };


        // TODO: Things down here are a bit ugly as the file format is a bit ugly ...

        // Now read all vectors and decode stored information
        let mut start_offset = 0;

        // In the raw file, support vectors are grouped by class
        for i in 0..num_classes {
            let num_sv_per_class = &header.nr_sv[i];
            let stop_offset = start_offset + *num_sv_per_class as usize;

            // Set support vector and coefficients
            for (i_vector, vector) in vectors[start_offset..stop_offset].iter().enumerate() {
                // Set support vectors
                for (i_attribute, attribute) in vector.features.iter().enumerate() {
                    // Make sure we have a "sane" file.
                    if attribute.index as usize != i_attribute {
                        return Result::Err("SVM support vector indices MUST range from [0 ... #(num_attributes - 1)].");
                    }

                    svm.classes[i].support_vectors.set(
                        i_vector,
                        attribute.index as usize,
                        attribute.value,
                    );
                }

                // Set coefficients
                for (i_coefficient, coefficient) in vector.coefs.iter().enumerate() {
                    svm.classes[i]
                        .coefficients
                        .set(i_coefficient, i_vector, *coefficient as f64);
                }
            }

            // Update last offset.
            start_offset = stop_offset;
        }

        // Return what we have
        return Result::Ok(svm);
    }


    /// Predicts all values for a set of problems.
    pub fn predict_values(&self, problems: &mut [Problem]) {
        // Compute all problems ...
        problems
            .par_iter_mut()
            .for_each(|problem| self.predict_value_one(problem));
    }


    // Predict the value for one problem.
    pub fn predict_value_one(&self, problem: &mut Problem) {
        // TODO: Dirty hack until faster allows us to operate on zipped, non-aligned arrays.
        assert_eq!(problem.features.len() % 8, 0);

        // Compute kernel, decision values and eventually the label
        self.compute_kernel_values(problem);
        self.compute_decision_values(problem);

        // Compute highest vote
        let highest_vote = find_max_index(&problem.vote);
        problem.label = self.classes[highest_vote].label;
    }


    /// Computes the kernel values for this problem
    fn compute_kernel_values(&self, problem: &mut Problem) {
        // Get current problem and decision values array
        let problem_features = &problem.features[..];


        // Compute kernel values per class
        for (i, class) in self.classes.iter().enumerate() {
            let kvalues = problem.kernel_values.get_vector_mut(i);

            self.kernel
                .compute(&class.support_vectors, problem_features, kvalues);
        }
    }


    /// Based on kernel values, computes the decision values for this problem.
    fn compute_decision_values(&self, problem: &mut Problem) {
        // Reset all votes
        set_all(&mut problem.vote, 0);

        // TODO: For some strange reason this code here seems to have little performance impact ...
        let mut p = 0;
        let dec_values = &mut problem.decision_values;


        // Since classification is symmetric, if we have N classes, we only need to go through
        // (N * N - 1) - 1 cases. For example for 4 classes we do:
        //
        //          j --->
        //          0    1   2   3
        //    i 0        x   x   x
        //    | 1            x   x
        //    v 2                x
        //      3
        //
        // For each valid combination (e.g., i:1, j:2), we then need to compute
        // the decision values, which consists of two parts:
        //
        // a) The coefficients of class(1) related to class(2) and
        // b) The coefficients of class(2) related to class(1).
        //
        // Both a) and b) are multiplied with the computed kernel values and summed,
        // and eventually used to compute on which side we are.

        for i in 0..self.classes.len() {
            for j in (i + 1)..self.classes.len() {
                let sv_coef0 = self.classes[i].coefficients.get_vector(j - 1);
                let sv_coef1 = self.classes[j].coefficients.get_vector(i);

                let kvalues0 = problem.kernel_values.get_vector(i);
                let kvalues1 = problem.kernel_values.get_vector(j);

                let mut simd_sum = f64s::splat(0.0);

                for (x, y) in zip(sv_coef0.simd_iter(), kvalues0.simd_iter()) {
                    simd_sum = simd_sum + x * y;
                }

                for (x, y) in zip(sv_coef1.simd_iter(), kvalues1.simd_iter()) {
                    simd_sum = simd_sum + x * y;
                }

                // TODO: Double check the index for RHO if it makes sense how we traverse the classes
                let sum = sum_f64s(simd_sum) - self.rho[p];
                let index_to_vote = if sum > 0.0 { i } else { j };

                dec_values[p] = sum;
                problem.vote[index_to_vote] += 1;

                p += 1;
            }
        }
    }
}


mod test {
    use rbfcsvm::RbfCSVM;
    use data::Problem;
    use randomization::Randomize;

    #[allow(unused_imports)] // TODO: Removing this causes 'unused import' warnings although it's being used.
    use test::Bencher;

    /// Produces a test case run for benchmarking
    #[allow(dead_code)]
    fn produce_testcase(
        classes: usize,
        sv_per_class: usize,
        attributes: usize,
        num_problems: usize,
    ) -> impl FnMut() {
        let mut svm = RbfCSVM::random(classes, sv_per_class, attributes);

        let mut problems = (0..num_problems)
            .map(|_| Problem::from_svm(&svm).randomize())
            .collect::<Vec<Problem>>();

        move || (&mut svm).predict_values(&mut problems)
    }

    #[bench]
    fn csvm_predict_sv128_attr16_problems1(b: &mut Bencher) {
        b.iter(produce_testcase(2, 64, 16, 1));
    }

    #[bench]
    fn csvm_predict_sv1024_attr16_problems1(b: &mut Bencher) {
        b.iter(produce_testcase(2, 512, 16, 1));
    }

    #[bench]
    fn csvm_predict_sv128_attr16_problems1024(b: &mut Bencher) {
        b.iter(produce_testcase(2, 64, 16, 1024));
    }

    #[bench]
    fn csvm_predict_aaa_t(b: &mut Bencher) {
        b.iter(produce_testcase(2, 3000, 8, 40));
    }

    #[bench]
    fn csvm_predict_sv1024_attr16_problems128(b: &mut Bencher) {
        b.iter(produce_testcase(2, 512, 16, 128));
    }

    #[bench]
    fn csvm_predict_sv1024_attr1024_problems1(b: &mut Bencher) {
        b.iter(produce_testcase(2, 512, 1024, 1));
    }

    #[test]
    fn test_something() {
        assert_eq!(4, 2 + 2);
    }
}
