use faster::{f32s, IntoSIMDRefIterator, IntoSIMDZip, SIMDIterator, SIMDZippedIterator, Sum};
use std::convert::From;

use kernel::Kernel;
use parser::ModelFile;
use rand::random;
use random::Random;
use vectors::SimdOptimized;

pub struct RbfKernel {
    pub gamma: f32,
}

impl Kernel for RbfKernel {
    fn compute(&self, vectors: &SimdOptimized<f32>, feature: &[f32], kernel_values: &mut [f64]) {
        // According to Instruments, for realistic SVMs and problems, the VAST majority of our
        // CPU time is spent in this loop.
        for (i, sv) in vectors.into_iter().enumerate() {
            // TODO: This allocates a Vec internally, doesn't it?
            let sum: f32 = (sv.simd_iter(f32s(0.0f32)), feature.simd_iter(f32s(0.0f32)))
                .zip()
                .simd_map(|(a, b)| (a - b) * (a - b))
                .simd_reduce(f32s::splat(0f32), |a, v| a + v)
                .sum();

            // This seems to be the single-biggest CPU spike: saving back kernel_values,
            // and computing exp() (saving back seems to have 3x time impact over exp(),
            // but I might misread "Instruments" for that particular one).
            kernel_values[i] = f64::from((-self.gamma * sum).exp());
        }
    }
}

impl Random for RbfKernel {
    fn new_random() -> Self { RbfKernel { gamma: random() } }
}

impl<'a> From<&'a ModelFile<'a>> for RbfKernel {
    fn from(model: &'a ModelFile<'a>) -> Self {
        RbfKernel {
            gamma: model.header.gamma,
        }
    }
}
