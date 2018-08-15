use std::convert::{From, TryFrom};

use super::{KernelDense, KernelSparse};
use crate::{
    errors::Error,
    parser::ModelFile,
    sparse::{SparseMatrix, SparseVector},
};

use simd_aligned::{f32s, RowOptimized, SimdMatrix, SimdVector};

#[derive(Copy, Clone, Debug, Default)]
#[doc(hidden)]
pub struct Sigmoid {
    gamma: f32,
    coef0: f32,
}

impl KernelDense for Sigmoid {
    fn compute(&self, vectors: &SimdMatrix<f32s, RowOptimized>, feature: &SimdVector<f32s>, output: &mut [f64]) {
        for (i, sv) in vectors.row_iter().enumerate() {
            let mut sum = f32s::splat(0.0);
            let feature: &[f32s] = &feature;

            for (a, b) in sv.iter().zip(feature) {
                sum += *a * *b;
            }

            output[i] = (f64::from(self.gamma * sum.sum() + self.coef0)).tanh();
        }
    }
}

impl KernelSparse for Sigmoid {
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
                    _ => break (f64::from(self.gamma * sum + self.coef0)).tanh(),
                }
            }
        }
    }
}

impl<'a, 'b> TryFrom<&'a ModelFile<'b>> for Sigmoid {
    type Error = Error;

    fn try_from(raw_model: &'a ModelFile<'b>) -> Result<Sigmoid, Error> {
        let gamma = raw_model.header.gamma.ok_or(Error::NoGamma)?;
        let coef0 = raw_model.header.coef0.ok_or(Error::NoCoef0)?;

        Ok(Sigmoid { gamma, coef0 })
    }
}
