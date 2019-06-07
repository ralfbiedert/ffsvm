use std::convert::From;

use super::{KernelDense, KernelSparse};
use crate::{
    f32s,
    parser::ModelFile,
    sparse::{SparseMatrix, SparseVector},
};

use simd_aligned::{MatrixD, Rows, VectorD};

#[derive(Copy, Clone, Debug, Default)]
#[doc(hidden)]
pub struct Linear {}

impl KernelDense for Linear {
    fn compute(&self, vectors: &MatrixD<f32s, Rows>, feature: &VectorD<f32s>, output: &mut [f64]) {
        for (i, sv) in vectors.row_iter().enumerate() {
            let mut sum = f32s::splat(0.0);
            let feature: &[f32s] = feature;

            for (a, b) in sv.iter().zip(feature) {
                sum += *a * *b;
            }

            output[i] = f64::from(sum.sum());
        }
    }
}

impl KernelSparse for Linear {
    fn compute(&self, vectors: &SparseMatrix<f32>, feature: &SparseVector<f32>, output: &mut [f64]) {
        for (i, sv) in vectors.row_iter().enumerate() {
            let mut sum = 0.0;
            let mut a_iter = sv.iter();
            let mut b_iter = feature.iter();

            let (mut a, mut b) = (a_iter.next(), b_iter.next());

            output[i] = loop {
                match (a, b) {
                    (Some((i_a, x)), Some((i_b, y))) if i_a == i_b => {
                        sum += x * y;

                        a = a_iter.next();
                        b = b_iter.next();
                    }
                    (Some((i_a, _)), Some((i_b, _))) if i_a < i_b => a = a_iter.next(),
                    (Some((i_a, _)), Some((i_b, _))) if i_a > i_b => b = b_iter.next(),
                    _ => break f64::from(sum),
                }
            }
        }
    }
}

impl<'a> From<&'a ModelFile<'a>> for Linear {
    fn from(_model: &'a ModelFile<'a>) -> Self { Self {} }
}
