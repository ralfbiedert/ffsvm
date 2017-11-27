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
    pub total_support_vectors: usize,
    pub num_support_vectors: Vec<u32>,
    pub starts: Vec<u32>,

    pub support_vectors: Matrix<Feature>,
    pub sv_coef: Matrix<Feature>,

    // TODO: create as context object so that we are thread-safe
    // Struct used for all computations. 
    scratchpad: Scratchpad,
}


impl CSVM {
    
    pub fn new(model: &RawModel) -> Result<CSVM, &'static str> {
        // Get basic info
        let vectors = model.header.total_sv as usize;
        let attributes = model.vectors[0].features.len();
        let num_classes = model.header.nr_class as usize;
        let total_support_vectors = model.header.total_sv as usize;

        // Allocate model
        let mut rval = CSVM {
            num_classes,
            total_support_vectors,
            gamma: model.header.gamma,
            num_support_vectors: model.header.nr_sv.clone(),
            starts: vec![0; num_classes],
            support_vectors: Matrix::new(vectors, attributes, 0.0),
            sv_coef: Matrix::new(num_classes - 1, total_support_vectors, 0.0),
            
            scratchpad: Scratchpad {
                kvalue: vec![0.0; vectors],
                vote: vec![0; num_classes],
            }
        };

        // Set values in model
        for (i_vector, vector) in model.vectors.iter().enumerate() {
            for (i_attribute, attribute) in vector.features.iter().enumerate() {
                
                // Make sure we have a "sane" file.
                if attribute.index as usize != i_attribute {
                    return Result::Err("SVM support vector indices MUST range from [0 ... #(num_attributes - 1)].");
                }
                
                rval.support_vectors.set(i_vector, attribute.index as usize, attribute.value);
            }
        }
        
        // for(i=1;i<nr_class;i++)
        //      start[i] = start[i-1]+model->nSV[i-1];
        
        let mut next= 0;
        for (i, start) in rval.starts.iter_mut().enumerate() {
            *start = next;
            next += rval.num_support_vectors[i];
        }
        
        println!("{:?}", rval);

        // Return what we have
        return Result::Ok(rval);            
    }
    
    
    // TODO: SIMD
    //    faster SIMD goes here ...
    //    let x = (&feature_vector[..]);
    //    let mut mp = x.simd_iter().map(|vector| { f32s::splat(10.0) + vector.abs() });
    //    let c = mp.scalar_collect();

    // Re-implementation of: 
    // double svm_predict_values(const svm_model *model, const svm_node *x, double* dec_values)
    pub fn predict_probability_csvm(&mut self, problem: &Matrix<Feature>) {
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
        for vote in self.scratchpad.kvalue.iter_mut() {
            *vote = 0.0;
        }
        
        let mut p = 0;
        for i in 0 .. self.num_classes {
            
            let si = self.starts[i];
            let ci = self.num_support_vectors[i];

            for j in (i+1) .. self.num_classes {
                let mut sum = 0.0f32;

                let sj = self.starts[j];
                let cj = self.num_support_vectors[j];
                
             //   let coef1 = self.    
            }
        }
        //        int p=0;
        //        for(i=0;i<nr_class;i++)
        //        for(int j=i+1;j<nr_class;j++)
        //        {
        //        double sum = 0;
        //        int si = start[i];
        //        int sj = start[j];
        //        int ci = model->nSV[i];
        //        int cj = model->nSV[j];
        //
        //        int k;
        //        double *coef1 = model->sv_coef[j-1];
        //        double *coef2 = model->sv_coef[i];
        //        for(k=0;k<ci;k++)
        //        sum += coef1[si+k] * kvalue[si+k];
        //        for(k=0;k<cj;k++)
        //        sum += coef2[sj+k] * kvalue[sj+k];
        //        sum -= model->rho[p];
        //        dec_values[p] = sum;
        //
        //        if(dec_values[p] > 0)
        //        ++vote[i];
        //        else
        //        ++vote[j];
        //        p++;
        //        }
        let p = 0;


        // println!("{:?}", self.scratchpad);
    }
}


