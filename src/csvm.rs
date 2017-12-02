
use matrix::Matrix;
use parser::RawModel;
use types::{Feature};
use faster::{IntoPackedRefIterator, f32s, PackedIterator };
use rand::{random, ChaChaRng, Rng};
use itertools::{zip};

#[allow(unused_imports)]
use test::{Bencher};


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
        let mut rng = ChaChaRng::new_unseeded();
        let mut problem = Problem::from_csvm(csvm);
        
        problem.features = rng.gen_iter().take(csvm.num_attributes).collect();
        
        problem
    }

}

impl CSVM {
    
    /// Creates a new random CSVM
    pub fn new_random(num_classes: usize, sv_per_class: usize, num_attributes: usize) -> CSVM {
        let mut rng = ChaChaRng::new_unseeded();
        let mut starts = vec![0 as u32; num_classes];
        let total_sv = num_classes * sv_per_class;
        
        for i in 1 .. num_classes {
            starts[i] = starts[i-1] + sv_per_class as u32;
        }
        
        CSVM {
            num_classes,
            num_attributes,
            total_support_vectors: total_sv,
            gamma: random(),
            rho: rng.gen_iter().take(num_classes).collect(),
            labels: rng.gen_iter().take(num_classes).collect(),
            num_support_vectors: vec![sv_per_class as u32; num_classes],
            starts,
            support_vectors: Matrix::new_random(total_sv, num_attributes),
            sv_coef: Matrix::new_random(num_classes - 1, total_sv),
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




    /// Creates a new CSVM from a raw model.
    pub fn predict_probability(&self, problems: &mut [Problem]) {
        
        // Get SIMD width: TODO: Surely there must be a better way to get SIMD width 
        let _temp = [0.0f32; 32];
        let simd_width = (&_temp[..]).simd_iter().width();
        
        //problem.par_iter_mut().for_each(||)
            
        // Compute all problems ...
        for problem in problems.iter_mut() {
            
            // Get current problem and decision values array
            let mut dec_values = &mut problem.dec_values;
            let current_problem = &problem.features[..];
            
      
            // Compute kernel values for each support vector 
            for (i, kvalue) in problem.kvalue.iter_mut().enumerate() {

                // Get current vector x (always same in this loop)
                let sv = self.support_vectors.get_vector(i);

                let mut sum = 0.0f32;
                let mut simd_sum = f32s::splat(0.0f32); 
                let mut simd_problem = current_problem.simd_iter();
                let mut simd_sv = sv.simd_iter();

                // SIMD compute of values 
                for (x, y) in zip(simd_problem, simd_sv) { 
                    simd_sum = simd_sum + (x - y) * (x - y);
                }

                // Sum components of our SIMD sum
                for i in 0 .. simd_width {
                    sum += simd_sum.extract(i as u32);
                }

                // Compute k-value
                *kvalue = (-self.gamma * sum).exp();
            }

            
            // Reset votes ... TODO: Is there a better way to do this?
            for vote in problem.vote.iter_mut() {
                *vote = 0;
            }

            let mut p = 0;
            for i in 0 .. self.num_classes {

                let si = self.starts[i];
                let ci = self.num_support_vectors[i];

                for j in (i+1) .. self.num_classes {
                    // Needs higher precision since we add lots of small values 
                    let mut sum: f64 = 0.0;

                    let sj = self.starts[j];
                    let cj = self.num_support_vectors[j];

                    let coef1 = self.sv_coef.get_vector(j-1);
                    let coef2 = self.sv_coef.get_vector(i);

                    for k in 0 .. ci {
                        let idx = (si+k) as usize;
                        sum += (coef1[idx] * problem.kvalue[idx]) as f64;
                    }

                    for k in 0 .. cj {
                        let idx = (sj+k) as usize;
                        sum += (coef2[idx] * problem.kvalue[idx]) as f64;
                    }


                    sum -= self.rho[p] as f64;
                    dec_values[p] = sum;

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
    
    move || { (&mut svm).predict_probability(&mut problems) } 
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
