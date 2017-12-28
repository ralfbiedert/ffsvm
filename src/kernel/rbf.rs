use std::convert::From;
use faster::{IntoPackedRefIterator, f32s};
use itertools::zip;

use kernel::Kernel;
use parser::ModelFile;
use vectors::SimdOptimized;
use random::Random;
use rand::random;
use util;


pub struct RbfKernel {
    pub gamma: f32,
}


impl Kernel for RbfKernel {

    fn compute(&self, vectors: &SimdOptimized<f32>, feature: &[f32], kernel_values: &mut [f64]) {

        // According to Instruments, for realistic SVMs and problems, the VAST majority of our
        // CPU time is spent in this loop.
        for (i, sv) in vectors.into_iter().enumerate() {

            let mut simd_sum = f32s::splat(0.0);

            // SIMD computation of values
            for (x, y) in zip(sv.simd_iter(), feature.simd_iter()) {
                let d = x - y;
                simd_sum = simd_sum + d * d;
            }

            let sum = util::sum_f32s(simd_sum);
            println!("{:?}: {:?}", i, sum);
            // This seems to be the single-biggest CPU spike: saving back kernel_values,
            // and computing exp() (saving back seems to have 3x time impact over exp(),
            // but I might misread "Instruments" for that particular one).
            kernel_values[i] = f64::from((-self.gamma * sum).exp());
        }
    }
}


impl Random for RbfKernel {
    fn new_random() -> Self {
        RbfKernel { gamma: random() }
    }
}


impl <'a> From<&'a ModelFile<'a>> for RbfKernel {

    fn from(model: &'a ModelFile<'a>) -> Self {
        RbfKernel { gamma: model.header.gamma }
    }
}
