use std::convert::From;

use super::KernelDense;
use crate::{parser::ModelFile, random::Random};

use simd_aligned::{f32s, RowOptimized, SimdMatrix, SimdVector};

#[derive(Copy, Clone, Debug, Default)]
#[doc(hidden)]
pub struct Linear {}

impl KernelDense for Linear {
    fn compute(&self, vectors: &SimdMatrix<f32s, RowOptimized>, feature: &SimdVector<f32s>, output: &mut [f64]) {
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

impl Random for Linear {
    fn new_random() -> Self { Linear {} }
}

impl<'a> From<&'a ModelFile<'a>> for Linear {
    fn from(_model: &'a ModelFile<'a>) -> Self { Linear {} }
}
