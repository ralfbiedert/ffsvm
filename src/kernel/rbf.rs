use std::convert::From;

use crate::kernel::Kernel;
use crate::parser::ModelFile;
use crate::random::Random;
use crate::vectors::SimdVectorsf32;

use packed_simd::{f32x16, f32x4, f32x8};
use rand::random;

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

#[derive(Copy, Clone, Debug, Default)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[doc(hidden)]
pub struct RbfKernel {
    pub gamma: f32,
}

macro_rules! compute_kernel_impl {
    ($fn:ident, $vectype:ty) => {
        #[inline]
        fn $fn(sv: &[f32], feature: &[f32], gamma: f32) -> f64 {
            #[allow(non_camel_case_types)]
            type f32s = $vectype;

            let width = f32s::lanes();
            let steps = sv.len() / width;

            let mut sum = f32s::splat(0.0);

            for i in 0..steps {
                let a = unsafe { f32s::from_slice_aligned_unchecked(&sv[i * width..]) };
                let b = unsafe { f32s::from_slice_aligned_unchecked(&feature[i * width..]) };
                sum += (a - b) * (a - b);
            }

            // This seems to be the single-biggest CPU spike: saving back kernel_values,
            // and computing exp() (saving back seems to have 3x time impact over exp(),
            // but I might misread "Instruments" for that particular one).
            f64::from((-gamma * sum.sum()).exp())
        }
    };
}

#[inline]
fn compute_inner_kernel_scalar(sv: &[f32], feature: &[f32], gamma: f32) -> f64 {
    let sum = sv
        .iter()
        .zip(feature)
        .map(|(a, b)| (a - b) * (a - b))
        .sum::<f32>();

    f64::from((-gamma * sum).exp())
}

compute_kernel_impl!(compute_inner_kernel_simdf32x16, f32x16);
compute_kernel_impl!(compute_inner_kernel_simdf32x8, f32x8);
compute_kernel_impl!(compute_inner_kernel_simdf32x4, f32x4);

impl Kernel for RbfKernel {
    fn compute(&self, vectors: &SimdVectorsf32, feature: &[f32], kernel_values: &mut [f64]) {
        // According to Instruments, for realistic SVMs and problems, the VAST majority of our
        // CPU time is spent in this loop.
        for (i, sv) in vectors.into_iter().enumerate() {
            assert_eq!(sv.len(), feature.len());

            let len = sv.len();

            kernel_values[i] = match len {
                _ if len % 16 == 0 => compute_inner_kernel_simdf32x16(sv, feature, self.gamma),
                _ if len % 8 == 0 => compute_inner_kernel_simdf32x8(sv, feature, self.gamma),
                _ if len % 4 == 0 => compute_inner_kernel_simdf32x4(sv, feature, self.gamma),
                _ => compute_inner_kernel_scalar(sv, feature, self.gamma),
            };
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
