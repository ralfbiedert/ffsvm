
use faster::{IntoPackedRefIterator, f32s, PackedIterator };
use rand::{random};
use itertools::{zip};
use rayon::prelude::*;

use matrix::Matrix;
use parser::RawModel;
use types::{Feature};
use util::{sum_f32s, random_vec};

#[allow(unused_imports)] // TODO: Removing this causes 'unused import' warnings although it's being used.
use test::{Bencher};


#[derive(Debug)]
pub struct Problem {
    pub features: Vec<Feature>,
    pub kvalue: Vec<Feature>,
    pub vote: Vec<u32>,
    pub dec_values: Vec<f64>,
    pub label: u32
}

#[derive(Debug)]
pub struct CSVM {
    pub num_classes: usize,
    pub num_attributes: usize,
    pub gamma: f32,
    pub rho: Vec<f32>,
    pub total_support_vectors: usize,
    pub num_support_vectors: Vec<u32>,
    pub starts: Vec<u32>,
    pub labels: Vec<u32>,

    pub support_vectors: Matrix<Feature>,
    pub sv_coef: Matrix<Feature>,
}


impl Problem {
    
    pub fn new(total_sv: usize, num_classes: usize, num_attributes: usize) -> Problem {
        Problem {
            kvalue: vec![0.0; total_sv],
            features: vec![Default::default(); total_sv],
            vote: vec![0; num_classes],
            dec_values: vec![0.0; num_attributes],
            label: 0
        }
    }
    
    pub fn from_csvm(csvm: &CSVM) -> Problem {
        Problem::new(csvm.total_support_vectors, csvm.num_classes, csvm.num_attributes)           
    }

    pub fn from_csvm_with_random(csvm: &CSVM) -> Problem {
        let mut problem = Problem::from_csvm(csvm);
        problem.features = random_vec(csvm.num_attributes);
        problem
    }

}

impl CSVM {
    
    /// Creates a new random CSVM
    pub fn new_random(num_classes: usize, sv_per_class: usize, num_attributes: usize) -> CSVM {
        let mut starts = vec![0 as u32; num_classes];
        
        let total_sv = num_classes * sv_per_class;
        let sv = random_vec(total_sv * num_attributes);
        let sv_coef = random_vec(total_sv * (num_classes - 1));

        for i in 1 .. num_classes {
            starts[i] = starts[i-1] + sv_per_class as u32;
        }

        CSVM {
            num_classes,
            num_attributes,
            total_support_vectors: total_sv,
            gamma: random(),
            rho: random_vec(num_classes),
            labels: random_vec(num_classes),
            num_support_vectors: vec![sv_per_class as u32; num_classes],
            starts,
            support_vectors: Matrix::from_flat_vec(sv, total_sv, num_attributes ),
            sv_coef: Matrix::from_flat_vec(sv_coef, num_classes - 1, total_sv),
        }
    }



    pub fn from_raw_model(raw_model: &RawModel) -> Result<CSVM, &'static str> {
        // Get basic info
        let vectors = raw_model.header.total_sv as usize;
        let num_attributes = raw_model.vectors[0].features.len();
        let num_classes = raw_model.header.nr_class as usize;
        let total_support_vectors = raw_model.header.total_sv as usize;

        // Allocate model
        let mut csvm_model = CSVM {
            num_classes,
            num_attributes,
            total_support_vectors,
            gamma: raw_model.header.gamma,
            rho: raw_model.header.rho.clone(),
            labels: raw_model.header.label.clone(),
            num_support_vectors: raw_model.header.nr_sv.clone(),
            starts: vec![0; num_classes],
            support_vectors: Matrix::new(vectors, num_attributes, 0.0),
            sv_coef: Matrix::new(num_classes - 1, total_support_vectors, 0.0),
        };

        // Set support vector and coefficients
        for (i_vector, vector) in raw_model.vectors.iter().enumerate() {
            
            // Set support vectors
            for (i_attribute, attribute) in vector.features.iter().enumerate() {

                // Make sure we have a "sane" file.
                if attribute.index as usize != i_attribute {
                    return Result::Err("SVM support vector indices MUST range from [0 ... #(num_attributes - 1)].");
                }

                csvm_model.support_vectors.set(i_vector, attribute.index as usize, attribute.value);
            }

            // Set coefficients 
            for (i_coef, coef) in vector.coefs.iter().enumerate() {
                csvm_model.sv_coef.set(i_coef, i_vector, *coef);
            }
        }
        
        // Compute starts
        let mut next= 0;
        for (i, start) in csvm_model.starts.iter_mut().enumerate() {
            *start = next;
            next += csvm_model.num_support_vectors[i];
        }
        
        // Return what we have
        return Result::Ok(csvm_model);            
    }

    
    pub fn predict_value_one(&self, problem: &mut Problem) {
        // TODO: Surely there must be a better way to get SIMD width 
        let _temp = [0.0f32; 32];
        let simd_width = (&_temp[..]).simd_iter().width();


        // Get current problem and decision values array
        let dec_values = &mut problem.dec_values;
        let current_problem = &problem.features[..];


        // Compute kernel values for each support vector 
        for (i, kernel_value) in problem.kvalue.iter_mut().enumerate() {

            // Get current vector x (always same in this loop)
            let sv = self.support_vectors.get_vector(i);
            let mut simd_sum = f32s::splat(0.0f32);

            // SIMD computation of values 
            for (x, y) in zip(current_problem.simd_iter(), sv.simd_iter()) {
                simd_sum = simd_sum + (x - y) * (x - y);
            }

            // TODO: There must be a better function to do this ...
            let sum = sum_f32s(simd_sum, simd_width);

            // Compute k-value
            *kernel_value = (-self.gamma * sum).exp();
        }


        // Reset votes ... TODO: Is there a better way to do this?
        for vote in problem.vote.iter_mut() {
            *vote = 0;
        }

        // TODO: For some strange reason this code down here seems to have little performance impact ...
        let mut p = 0;
        for i in 0 .. self.num_classes {

            let s_i = self.starts[i] as usize;
            let nsv_i = self.num_support_vectors[i] as usize;

            for j in (i+1) .. self.num_classes {

                let s_j = self.starts[j] as usize;
                let nsv_j = self.num_support_vectors[j] as usize;

                let simd_sum1 = CSVM::simd_compute_partial_decision_value(self.sv_coef.get_vector(j - 1), &problem.kvalue, s_i, s_i + nsv_i);
                let simd_sum2 = CSVM::simd_compute_partial_decision_value(self.sv_coef.get_vector(i), &problem.kvalue, s_j, s_j + nsv_j);

                let sum = sum_f32s(simd_sum1, simd_width) + sum_f32s(simd_sum2, simd_width) - self.rho[p];

                dec_values[p] = sum as f64;

                if dec_values[p] > 0.0 {
                    problem.vote[i] += 1;
                } else {
                    problem.vote[j] += 1;
                }

                p += 1;
            }
        }

        let mut vote_max_idx = 0;

        for i in 1 .. self.num_classes {
            if problem.vote[i] > problem.vote[vote_max_idx] {
                vote_max_idx = i;
            }
        }
        
        problem.label = self.labels[vote_max_idx];

    }

    /// Creates a new CSVM from a raw model.
    pub fn predict_values(&self, problems: &mut [Problem]) {
          
        // Compute all problems ...
        problems.par_iter_mut().for_each( | problem| {
            self.predict_value_one(problem)            
        });
    }


    /// Computes our partial decision value 
    fn simd_compute_partial_decision_value(all_coef: &[f32], all_kvalue: &[f32], a: usize, b: usize) -> f32s {
        // TODO: WE MIGHT NEED TO REWORK THIS SINCE THIS 
        // TODO: SUM NEEDS HIGH PRECISION (f64) FROM THE LOOKS OF IT
        let coef = &all_coef[a..b];
        let kvalue = &all_kvalue[a..b];
        let mut simd_sum = f32s::splat(0.0f32);

        for (x, y) in zip(coef.simd_iter(), kvalue.simd_iter()) {
            simd_sum = simd_sum + x * y
        }

        simd_sum
    }

}



/// Produces a test case run for benchmarking
#[allow(dead_code)]
fn produce_testcase(classes: usize, sv_per_class: usize, attributes: usize, num_problems: usize) -> impl FnMut()
{
    let mut svm = CSVM::new_random(classes, sv_per_class, attributes);
    let mut problems = Vec::with_capacity(num_problems);
    
    for _i in 0 .. num_problems {
        let problem = Problem::from_csvm_with_random(&svm);
        problems.push(problem);
    }
    
    move || { (&mut svm).predict_values(&mut problems) } 
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
fn csvm_predict_sv1024_attr16_problems128(b: &mut Bencher) {
    b.iter(produce_testcase(2, 512, 16, 128));
}

#[bench]
fn csvm_predict_sv1024_attr1024_problems1(b: &mut Bencher) {
    b.iter(produce_testcase(2, 512, 1024, 1));
}



#[test]
fn test_something() {
    assert_eq!(4, 2+2);
}
