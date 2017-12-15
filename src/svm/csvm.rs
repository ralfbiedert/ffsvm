use std::convert::TryFrom;
use std::marker::Sync;

use faster::{IntoPackedRefIterator, f64s};
use rand::random;
use itertools::zip;

use random::{random_vec, Randomize, Random};
use util::{find_max_index, set_all, sum_f64s, prefered_simd_size};
use svm::{SVM, Class,PredictProblem};
use svm::problem::Problem;
use parser::{ModelFile};
use kernel::RbfKernel;
use kernel::Kernel;



impl <'a, T> SVM<T> where T : Kernel + Random 
{
    
    /// Creates a new random CSVM
    pub fn random(num_classes: usize, num_sv_per_class: usize, num_attributes: usize) -> Self {

        let num_total_sv = num_classes * num_sv_per_class;
        let classes = (0..num_classes)
            .map(|class| {
                Class::with_parameters(num_classes, num_sv_per_class, num_attributes, class as u32).randomize()
            })
            .collect::<Vec<Class>>();

        SVM {
            num_total_sv,
            num_attributes,
            rho: random_vec(num_classes),
            kernel: T::new_random(),
            classes,
        }
    }



    /// Computes the kernel values for this problem
    fn compute_kernel_values(&self, problem: &mut Problem) {
        // Get current problem and decision values array
        let problem_features = &problem.features[..];


        // Compute kernel values per class
        for (i, class) in self.classes.iter().enumerate() {

            let kvalues = problem.kernel_values.get_vector_mut(i);

            self.kernel.compute(&class.support_vectors, problem_features, kvalues);
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


impl <'a, 'b, T> TryFrom<&'b ModelFile<'a>> for SVM<T> where T : Kernel + From<&'b ModelFile<'a>> 
{
    type Error = &'static str;

    /// Creates a SVM from the given raw model.
    fn try_from(raw_model: &'b ModelFile<'a>) -> Result<SVM<T>, &'static str> {
        let header = &raw_model.header;
        let vectors = &raw_model.vectors;

        // Get basic info
        let num_attributes = vectors[0].features.len();
        let num_classes = header.nr_class as usize;
        let num_total_sv = header.total_sv as usize;


        // Construct vector of classes
        let classes = (0..num_classes)
            .map(|class| {
                let label = header.label[class];
                let num_sv = header.nr_sv[class] as usize;
                Class::with_parameters(num_classes, num_sv , num_attributes, label)
            })
            .collect::<Vec<Class>>();



        // Allocate model
        let mut svm = SVM {
            num_total_sv,
            num_attributes,
            kernel: T::from(raw_model),
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

                    svm.classes[i].support_vectors.set(i_vector, attribute.index as usize, attribute.value);
                }

                // Set coefficients
                for (i_coefficient, coefficient) in vector.coefs.iter().enumerate() {
                    svm.classes[i].coefficients.set(i_coefficient, i_vector, *coefficient as f64);
                }
            }

            // Update last offset.
            start_offset = stop_offset;
        }

        // Return what we have
        return Result::Ok(svm);
    }

}


impl <'a, T> PredictProblem for SVM<T> where T : Kernel + Random + Sync
{
    
    // Predict the value for one problem.
    fn predict_value(&self, problem: &mut Problem) {
        // TODO: Dirty hack until faster allows us to operate on zipped, non-aligned arrays.
        assert_eq!(problem.features.len() % prefered_simd_size(3), 0);

        // Compute kernel, decision values and eventually the label
        self.compute_kernel_values(problem);
        self.compute_decision_values(problem);

        // Compute highest vote
        let highest_vote = find_max_index(&problem.vote);
        problem.label = self.classes[highest_vote].label;
    }
    
}