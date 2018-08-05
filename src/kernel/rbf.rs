use std::convert::From;

use crate::kernel::Kernel;
use crate::parser::ModelFile;
use crate::random::Random;
use crate::vectors::SimdOptimized;

use packed_simd::f32x8;
use rand::random;

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

#[derive(Copy, Clone, Debug, Default)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[doc(hidden)]
pub struct RbfKernel {
    pub gamma: f32,
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

#[inline]
fn compute_inner_kernel_simdf32x8(sv: &[f32], feature: &[f32], gamma: f32) -> f64 {
    #[allow(non_camel_case_types)]
    type f32s = f32x8;

    let width = f32s::lanes();
    let steps = sv.len() / width;

    let mut sum = f32s::splat(0.0);

    for i in 0..steps {
        // When benchmarking `csvm_predict_sv1024_attr1024_problems1` with AVX2:

        // 238,928 ns / iter
        let a = unsafe { f32s::from_slice_aligned_unchecked(&sv[i * width..]) };
        let b = unsafe { f32s::from_slice_aligned_unchecked(&feature[i * width..]) };

        // While I would love to have purely safe code, anything that's not `unchecked` will
        // reduce the overall classification speed by 50%! The problem I see with `packed_simd` is
        // that it treats every f32s independently, so it forces checks on every f32s load operation.
        // It would be much nicer having an API that does a sanity check once, and then just iterates.

        // 237,541 ns / iter
        // let a = unsafe { f32s::from_slice_unaligned_unchecked(&sv[i * width..]) };
        // let b = unsafe { f32s::from_slice_unaligned_unchecked(&feature[i * width..]) };

        // 343,970 ns / iter
        // let a = f32s::from_slice_aligned(&sv[i * width..]);
        // let b = f32s::from_slice_aligned(&feature[i * width..]);

        // 363,796 ns / iter
        // let a = f32s::from_slice_unaligned(&sv[i * width..]);
        // let b = f32s::from_slice_unaligned(&feature[i * width..]);
        sum += (a - b) * (a - b);
    }

    // This seems to be the single-biggest CPU spike: saving back kernel_values,
    // and computing exp() (saving back seems to have 3x time impact over exp(),
    // but I might misread "Instruments" for that particular one).
    f64::from((-gamma * sum.sum()).exp())
}

impl Kernel for RbfKernel {
    fn compute(&self, vectors: &SimdOptimized<f32>, feature: &[f32], kernel_values: &mut [f64]) {
        // According to Instruments, for realistic SVMs and problems, the VAST majority of our
        // CPU time is spent in this loop.
        for (i, sv) in vectors.into_iter().enumerate() {
            kernel_values[i] = compute_inner_kernel_simdf32x8(sv, feature, self.gamma);
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
