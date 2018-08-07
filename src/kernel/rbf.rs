use std::convert::From;

use crate::kernel::Kernel;
use crate::parser::ModelFile;
use crate::random::Random;
use crate::simd_matrix::{f32s, Simd, SimdRows};

use rand::random;

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

#[derive(Copy, Clone, Debug, Default)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[doc(hidden)]
pub struct RbfKernel {
    pub gamma: f32,
}

impl Kernel for RbfKernel {
    fn compute(&self, vectors: &SimdRows<f32s>, feature: &[f32s], output: &mut [f64]) {
        // According to Instruments, for realistic SVMs and problems, the VAST majority of our
        // CPU time is spent in this loop.
        for (i, sv) in vectors.into_iter().enumerate() {
            let mut sum = f32s::splat(0.0);

            for (a, b) in sv.iter().zip(feature) {
                sum += (*a - *b) * (*a - *b);
            }

            // This seems to be the single-biggest CPU spike: saving back kernel_values,
            // and computing exp() (saving back seems to have 3x time impact over exp(),
            // but I might misread "Instruments" for that particular one).
            output[i] = f64::from((-self.gamma * sum.sum()).exp())
        }
    }
}

impl Random for RbfKernel {
    fn new_random() -> Self {
        RbfKernel { gamma: random() }
    }
}

impl<'a> From<&'a ModelFile<'a>> for RbfKernel {
    fn from(model: &'a ModelFile<'a>) -> Self {
        RbfKernel {
            gamma: model.header.gamma,
        }
    }
}
