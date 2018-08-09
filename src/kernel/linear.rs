use std::convert::From;

use crate::kernel::Kernel;
use crate::parser::ModelFile;
use crate::random::Random;

use rand::random;
use simd_aligned::{f32s, RowOptimized, SimdMatrix, SimdVector};

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

#[derive(Copy, Clone, Debug, Default)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[doc(hidden)]
pub struct LinearKernel {}

impl Kernel for LinearKernel {
    fn compute(
        &self,
        vectors: &SimdMatrix<f32s, RowOptimized>,
        feature: &SimdVector<f32s>,
        output: &mut [f64],
    ) {
        for (i, sv) in vectors.row_iter().enumerate() {
            let mut sum = f32s::splat(0.0);
            let feature: &[f32s] = &feature;

            for (a, b) in sv.iter().zip(feature) {
                sum += *a * *b;
            }

            output[i] = f64::from(sum.sum());
        }
    }
}

impl Random for LinearKernel {
    fn new_random() -> Self {
        LinearKernel {}
    }
}

impl<'a> From<&'a ModelFile<'a>> for LinearKernel {
    fn from(model: &'a ModelFile<'a>) -> Self {
        LinearKernel {}
    }
}
