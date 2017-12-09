use faster::{IntoPackedRefIterator, f32s, f64s, PackedIterator };
use rand::{random};
use itertools::{zip};
use rayon::prelude::*;

use manyvectors::ManyVectors;
use parser::RawModel;
use data::{Class, Problem, SVM};
use util::{sum_f32s, sum_f64s, random_vec };


#[allow(unused_imports)] // TODO: Removing this causes 'unused import' warnings although it's being used.
use test::{Bencher};


pub struct RbfCData {
    gamma: f64,
} 

pub type RbfCSVM = SVM<RbfCData>;


impl RbfCSVM {
    
    /// Creates a new random CSVM
    pub fn random(num_classes: usize, sv_per_class: usize, num_attributes: usize) -> RbfCSVM {
        let mut starts = vec![0; num_classes];
        let mut classes = Vec::with_capacity(num_classes);
        
        let total_sv = num_classes * sv_per_class;

        for i in 1 .. num_classes {
            starts[i] = starts[i-1] + sv_per_class as u32;
        }

        for i in 0 .. num_classes {
            let mut class = Class {
                num_support_vectors: sv_per_class,
                label: i as u32,
                coefficients: ManyVectors::with_dimension(num_classes - 1, sv_per_class, Default::default()),
                support_vectors: ManyVectors::with_dimension(sv_per_class, num_attributes, Default::default())
            };    
            
            classes.push(class);
        }


        RbfCSVM {
            num_attributes,
            total_support_vectors: total_sv,
            rho: random_vec(num_classes),
            extra: RbfCData {
                gamma: random()
            },
            classes
        }
    }

    
    pub fn from_raw_model(raw_model: &RawModel) -> Result<RbfCSVM, &'static str> {
        let header = &raw_model.header;
        let vectors = &raw_model.vectors;
        
        // Get basic info
        let num_attributes = vectors[0].features.len();
        let num_classes = header.nr_class as usize;
        let total_support_vectors = header.total_sv as usize;

        // Allocate model
        let mut csvm_model = RbfCSVM {
            num_attributes,
            total_support_vectors,
            extra: RbfCData {
                gamma: header.gamma
            },
            rho: header.rho.clone(),
            classes: Vec::with_capacity(num_classes),
        };


        for i in 0 .. num_classes {
            let mut c = Class::with_dimensions(num_classes, header.nr_sv[i] as usize, num_attributes);
            c.label = header.label[i];
            
            csvm_model.classes.push(c);
        }
        
        
        let mut start = 0;
        
        for i in 0 .. header.nr_sv.len() {
            let n = &header.nr_sv[i];
            let stop = start + *n as usize;


            // Set support vector and coefficients
            for (i_vector, vector) in vectors[start .. stop].iter().enumerate() {
                
                // Set support vectors
                for (i_attribute, attribute) in vector.features.iter().enumerate() {

                    // Make sure we have a "sane" file.
                    if attribute.index as usize != i_attribute {
                        return Result::Err("SVM support vector indices MUST range from [0 ... #(num_attributes - 1)].");
                    }

                    csvm_model.classes[i].support_vectors.set(i_vector, attribute.index as usize, attribute.value);
                }

                // Set coefficients 
                for (i_coef, coef) in vector.coefs.iter().enumerate() {
                    csvm_model.classes[i].coefficients.set(i_coef,i_vector, *coef as f64);
                }
            }      
            
            start = stop;
        }

        // Return what we have
        return Result::Ok(csvm_model);            
    }

    
    pub fn predict_value_one(&self, problem: &mut Problem) {
        // TODO: Surely there must be a better way to get SIMD width 
        let _temp32 = [0.0f32; 32];
        let _temp64 = [0.0f64; 32];

        let simd_width_f32 = (&_temp32[..]).simd_iter().width();
        let simd_width_f64 = (&_temp64[..]).simd_iter().width();


        // Get current problem and decision values array
        let dec_values = &mut problem.decision_values;
        let current_problem = &problem.features[..];
        
        
        for i_class in 0 .. self.classes.len() {
            let class = &self.classes[i_class];
            let kvalues = problem.kernel_values.get_vector_mut(i_class);
            
            for i_sv in 0 .. class.support_vectors.vectors {
                let sv = class.support_vectors.get_vector(i_sv);

                let mut simd_sum = f32s::splat(0.0);


                // SIMD computation of values 
                for (x, y) in zip(current_problem.simd_iter(), sv.simd_iter()) {
                    let d = x - y;
                    simd_sum = simd_sum + d * d;
                }

                let sum = sum_f32s(simd_sum, simd_width_f32);
                let kvalue = (-self.extra.gamma * sum as f64).exp();

                kvalues[i_sv] = kvalue;
            }
        }

        // Reset votes ... TODO: Is there a better way to do this?
        for vote in problem.vote.iter_mut() {
            *vote = 0;
        }



        // TODO: For some strange reason this code down here seems to have little performance impact ...
        let mut p = 0;
        for i in 0 .. self.classes.len() {

//            let s_i = self.starts[i] as usize;
//            let nsv_i = self.num_support_vectors[i] as usize;

            for j in (i+1) .. self.classes.len() {

//                let s_j = self.starts[j] as usize;
//                let nsv_j = self.num_support_vectors[j] as usize;

                let mut simd_sum = f64s::splat(0.0);
                
                let class_1 = &self.classes[j-1];
                let class_2 = &self.classes[i];


                let sv_coef1 = class_1.coefficients.get_vector(i);

                let sv_coef2 = class_2.coefficients.get_vector(j-1);


                RbfCSVM::simd_compute_partial_decision_value(&mut simd_sum,sv_coef1, problem.kernel_values.get_vector(j-1));
                RbfCSVM::simd_compute_partial_decision_value(&mut simd_sum, sv_coef2, problem.kernel_values.get_vector(i));
                
                let sum = sum_f64s(simd_sum, simd_width_f64) - self.rho[p];

                dec_values[p] = sum ;

                if dec_values[p] > 0.0 {
                    problem.vote[i] += 1;
                } else {
                    problem.vote[j] += 1;
                }


                p += 1;
            }
        }


        let mut vote_max_idx = 0;

        for i in 1 .. self.classes.len() {
            if problem.vote[i] > problem.vote[vote_max_idx] {
                vote_max_idx = i;
            }
        }


        problem.label = self.classes[vote_max_idx].label;

    }

    /// Creates a new CSVM from a raw model.
    pub fn predict_values(&self, problems: &mut [Problem]) {
          
        // Compute all problems ...
        problems.par_iter_mut().for_each( | problem| {
            self.predict_value_one(problem)            
        });
    }


    /// Computes our partial decision value
    fn simd_compute_partial_decision_value(simd_sum: &mut f64s, coef: &[f64], kvalue: &[f64]) {
        // TODO: WE MIGHT NEED TO REWORK THIS SINCE THIS 
        // TODO: SUM NEEDS HIGH PRECISION (f64) FROM THE LOOKS OF IT
       
        for (x, y) in zip(coef.simd_iter(), kvalue.simd_iter()) {
            *simd_sum = *simd_sum + x * y;
        }
    }

}



/// Produces a test case run for benchmarking
#[allow(dead_code)]
fn produce_testcase(classes: usize, sv_per_class: usize, attributes: usize, num_problems: usize) -> impl FnMut()
{
    let mut svm = RbfCSVM::random(classes, sv_per_class, attributes);
    let mut problems = Vec::with_capacity(num_problems);
    
    for _i in 0 .. num_problems {
        let mut problem = Problem::from_svm(&svm);
        problem.features = random_vec(attributes);
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
