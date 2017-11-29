use matrix::Matrix;
use parser::RawModel;
use types::{Feature};

#[derive(Debug)]
struct Scratchpad {
    kvalue: Vec<Feature>,
    vote: Vec<u32>,
}


#[derive(Debug)]
pub struct CSVM {
    pub num_classes: usize,
    pub gamma: f32,
    pub rho: Vec<f32>,
    pub total_support_vectors: usize,
    pub num_support_vectors: Vec<u32>,
    pub starts: Vec<u32>,
    pub labels: Vec<u32>,

    pub support_vectors: Matrix<Feature>,
    pub sv_coef: Matrix<Feature>,

    // TODO: create as context object so that we are thread-safe
    // Struct used for all computations. 
    scratchpad: Scratchpad,
}


impl CSVM {
    
    pub fn new(raw_model: &RawModel) -> Result<CSVM, &'static str> {
        // Get basic info
        let vectors = raw_model.header.total_sv as usize;
        let attributes = raw_model.vectors[0].features.len();
        let num_classes = raw_model.header.nr_class as usize;
        let total_support_vectors = raw_model.header.total_sv as usize;

        // Allocate model
        let mut csvm_model = CSVM {
            num_classes,
            total_support_vectors,
            gamma: raw_model.header.gamma,
            rho: raw_model.header.rho.clone(),
            labels: raw_model.header.label.clone(),
            num_support_vectors: raw_model.header.nr_sv.clone(),
            starts: vec![0; num_classes],
            support_vectors: Matrix::new(vectors, attributes, 0.0),
            sv_coef: Matrix::new(num_classes - 1, total_support_vectors, 0.0),
            
            scratchpad: Scratchpad {
                kvalue: vec![0.0; vectors],
                vote: vec![0; num_classes],
            }
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
        // for(i=1;i<nr_class;i++)
        //      start[i] = start[i-1]+model->nSV[i-1];
        
        // Compute starts
        let mut next= 0;
        for (i, start) in csvm_model.starts.iter_mut().enumerate() {
            *start = next;
            next += csvm_model.num_support_vectors[i];
        }
        
        // Return what we have
        return Result::Ok(csvm_model);            
    }
    
    
    // TODO: SIMD
    //    faster SIMD goes here ...
    //    let x = (&feature_vector[..]);
    //    let mut mp = x.simd_iter().map(|vector| { f32s::splat(10.0) + vector.abs() });
    //    let c = mp.scalar_collect();

    // Re-implementation of: 
    // double svm_predict_values(const svm_model *model, const svm_node *x, double* dec_values)
    pub fn predict_probability_csvm(&mut self, problem: &Matrix<Feature>) -> u32 {
        
        // TODO: This should be parameter, but what is it?
        let mut dec_values = vec![0.0; problem.attributes];
        
        // int l = model->l;  -- l being total number of SV
        
        // for(i=0;i<l;i++)
        //      kvalue[i] = Kernel::k_function(x,model->SV[i],model->param);
        
        // TODO: We only compute for the first input vector 
        let current_problem = problem.get_vector(0);
        for (i, kvalue) in self.scratchpad.kvalue.iter_mut().enumerate() {
            
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
        
        // for(i=0;i<nr_class;i++) 
        //     vote[i] = 0;
        for vote in self.scratchpad.vote.iter_mut() {
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
                    sum += (coef1[idx] * self.scratchpad.kvalue[idx]) as f64;
                }

                for k in 0 .. cj {
                    let idx =(sj+k) as usize;
                    sum += (coef1[idx] * self.scratchpad.kvalue[idx]) as f64;
                }



                sum -= self.rho[p] as f64;
                dec_values[p] = sum;

                //println!("{:?} {:?} {:?} {:?} {:?}", i, j, sum, self.rho[p], dec_values[p]);


                if dec_values[p] > 0.0 {
                    self.scratchpad.vote[i] += 1;
                } else {
                    self.scratchpad.vote[j] += 1;
                }
                
                p += 1;
            }
        }

        let mut vote_max_idx = 0;
        for i in 1 .. self.num_classes {
            if self.scratchpad.vote[i] > self.scratchpad.vote[vote_max_idx] {
                vote_max_idx = i;
            }     
        }

        println!("{:?}", vote_max_idx);

        self.labels[vote_max_idx]

    }
}


