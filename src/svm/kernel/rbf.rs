use std::convert::{From, TryFrom};

use super::Kernel;
use crate::errors::SVMError;
use crate::parser::ModelFile;
use crate::random::Random;

use rand::random;
use simd_aligned::{f32s, RowOptimized, SimdMatrix, SimdVector};

#[derive(Copy, Clone, Debug, Default)]
#[doc(hidden)]
pub struct Rbf {
    pub gamma: f32,
}

impl Kernel for Rbf {
    fn compute(
        &self,
        vectors: &SimdMatrix<f32s, RowOptimized>,
        feature: &SimdVector<f32s>,
        output: &mut [f64],
    ) {
        // According to Instruments, for realistic SVMs and problems, the VAST majority of our
        // CPU time is spent in this loop.
        for (i, sv) in vectors.row_iter().enumerate() {
            let mut sum = f32s::splat(0.0);
            let feature: &[f32s] = &feature;

            for (a, b) in sv.iter().zip(feature) {
                sum += (*a - *b) * (*a - *b);
            }

            // This seems to be the single-biggest CPU spike: saving back kernel_values,
            // and computing exp() (saving back seems to have 3x time impact over exp(),
            // but I might misread "Instruments" for that particular one).
            output[i] = f64::from((-self.gamma * sum.sum()).exp());
        }
    }
}

impl Random for Rbf {
    fn new_random() -> Self {
        Rbf { gamma: random() }
    }
}

impl<'a, 'b> TryFrom<&'a ModelFile<'b>> for Rbf {
    type Error = SVMError;

    fn try_from(raw_model: &'a ModelFile<'b>) -> Result<Rbf, SVMError> {
        let gamma = raw_model.header.gamma.ok_or(SVMError::NoGamma)?;

        Ok(Rbf { gamma })
    }
}
