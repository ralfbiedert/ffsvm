use faster::{IntoPackedRefIterator, f32s };
use itertools::{zip};
use manyvectors::{ManyVectors};
use data::{Kernel};
use util::{sum_f32s};

pub struct RbfKernel {
    pub gamma: f64,
}




impl Kernel for RbfKernel {
    
    fn compute(&self, vectors: &ManyVectors<f32>, feature: &[f32], kernel_values: &mut [f64]) {

        for (i, sv) in vectors.into_iter().enumerate() {
            println!("{:?} {:?} ", i, feature);

            let mut simd_sum = f32s::splat(0.0);

            // SIMD computation of values 
            for (x, y) in zip(sv.simd_iter(), feature.simd_iter()) {
                println!("{:?} {:?} {:?}", i, x, y);

                let d = x - y;
                simd_sum = simd_sum + d * d;
            }

            let sum = sum_f32s(simd_sum);
            kernel_values[i] = (-self.gamma * sum as f64).exp();
            
            println!("{:?} {:?}", i, kernel_values[i]);
        }

    }
    
}
