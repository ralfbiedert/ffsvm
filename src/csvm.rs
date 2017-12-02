
use matrix::Matrix;
use parser::RawModel;
use types::{Feature, Label};
use rand::{random, ChaChaRng, Rng};

#[allow(unused_imports)]
use test::{Bencher};

const MAX_PROBLEM_SIZE: usize = 1024;

#[derive(Debug)]
pub struct Scratchpad {
    pub kvalue: Vec<Feature>,
    pub vote: Vec<u32>,
    pub dec_values: Matrix<f64>,
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

impl Scratchpad {
    
    pub fn new(total_sv: usize, num_classes: usize, max_problems: usize, num_attributes: usize) -> Scratchpad {
        Scratchpad {
            kvalue: vec![0.0; total_sv],
            vote: vec![0; num_classes],
            dec_values: Matrix::new(max_problems, num_attributes, 0.0),
        }
    }
    
    pub fn from_csvm(csvm: &CSVM) -> Scratchpad {
        Scratchpad::new(csvm.total_support_vectors, csvm.num_classes, MAX_PROBLEM_SIZE, csvm.num_attributes)           
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



    // TODO: SIMD
    //    faster SIMD goes here ...
    //    let x = (&feature_vector[..]);
    //    let mut mp = x.simd_iter().map(|vector| { f32s::splat(10.0) + vector.abs() });
    //    let c = mp.scalar_collect();
    /// Creates a new CSVM from a raw model.
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
    
    
    pub fn predict_probability(&mut self, scratchpad: &mut Scratchpad, problem: &Matrix<Feature>, solution: &mut Matrix<Label>) {
        
        for problem_index in 0 .. problem.vectors {
            
            let current_problem = problem.get_vector(problem_index);
            let mut dec_values = scratchpad.dec_values.get_vector_mut(problem_index);
            
            for (i, kvalue) in scratchpad.kvalue.iter_mut().enumerate() {

                // Get current vector x (always same in this loop)
                let sv = self.support_vectors.get_vector(i);
                let mut sum: Feature = 0.0;

                for (ix, x) in current_problem.iter().enumerate() {
                    let y = sv[ix];
                    let d = x - y;
                    sum += d * d;
                }

                *kvalue = (-self.gamma * sum).exp();
            }


            for vote in scratchpad.vote.iter_mut() {
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
                        let idx =(si+k) as usize;
                        sum += (coef1[idx] * scratchpad.kvalue[idx]) as f64;
                    }

                    for k in 0 .. cj {
                        let idx =(sj+k) as usize;
                        sum += (coef2[idx] * scratchpad.kvalue[idx]) as f64;
                    }


                    sum -= self.rho[p] as f64;
                    dec_values[p] = sum;

                    if dec_values[p] > 0.0 {
                        scratchpad.vote[i] += 1;
                    } else {
                        scratchpad.vote[j] += 1;
                    }

                    p += 1;
                }
            }

            let mut vote_max_idx = 0;

            for i in 1 .. self.num_classes {
                if scratchpad.vote[i] > scratchpad.vote[vote_max_idx] {
                    vote_max_idx = i;
                }
            }
            
            
            solution.set(0, problem_index, self.labels[vote_max_idx] );
        }
    }
}



/// Produces a test case run for benchmarking
#[allow(dead_code)]
fn produce_testcase(classes: usize, sv_per_class: usize, attributes: usize, problems: usize) -> impl FnMut()
{
    let mut svm = CSVM::new_random(classes, sv_per_class, attributes);
    let mut solution= Matrix::new_random(1, problems);
    let mut scratchpad = Scratchpad::from_csvm(&svm);
    let problem = Matrix::new_random(problems, attributes);
    
    move || { (&mut svm).predict_probability(&mut scratchpad, &problem, &mut solution) } 
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
