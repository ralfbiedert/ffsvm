use std::convert::{From, TryFrom};

use super::KernelDense;
use crate::{parser::ModelFile, random::Random, SVMError};

use rand::random;
use simd_aligned::{f32s, RowOptimized, SimdMatrix, SimdVector};

#[derive(Copy, Clone, Debug, Default)]
#[doc(hidden)]
pub struct Poly {
    degree: u32,
    gamma: f32,
    coef0: f32,
}

impl KernelDense for Poly {
    fn compute(&self, vectors: &SimdMatrix<f32s, RowOptimized>, feature: &SimdVector<f32s>, output: &mut [f64]) {
        for (i, sv) in vectors.row_iter().enumerate() {
            let mut sum = f32s::splat(0.0);
            let feature: &[f32s] = &feature;

            for (a, b) in sv.iter().zip(feature) {
                sum += *a * *b;
            }

            output[i] = crate::util::powi(f64::from(self.gamma * sum.sum() + self.coef0), self.degree);
        }
    }
}

impl Random for Poly {
    fn new_random() -> Self {
        Poly {
            gamma: random(),
            coef0: random(),
            degree: random(),
        }
    }
}

impl<'a, 'b> TryFrom<&'a ModelFile<'b>> for Poly {
    type Error = SVMError;

    fn try_from(raw_model: &'a ModelFile<'b>) -> Result<Poly, SVMError> {
        let gamma = raw_model.header.gamma.ok_or(SVMError::NoGamma)?;
        let coef0 = raw_model.header.coef0.ok_or(SVMError::NoCoef0)?;
        let degree = raw_model.header.degree.ok_or(SVMError::NoDegree)?;

        Ok(Poly { gamma, coef0, degree })
    }
}
