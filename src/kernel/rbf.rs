use std::convert::{From, TryFrom};

use crate::kernel::Kernel;
use crate::parser::ModelFile;
use crate::random::Random;
use crate::SVMError;

use rand::random;
use simd_aligned::{f32s, RowOptimized, SimdMatrix, SimdVector};

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

#[derive(Copy, Clone, Debug, Default)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[doc(hidden)]
pub struct RbfKernel {
    pub gamma: f32,
}

impl Kernel for RbfKernel {
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

impl Random for RbfKernel {
    fn new_random() -> Self {
        RbfKernel { gamma: random() }
    }
}

impl<'a, 'b> TryFrom<&'a ModelFile<'b>> for RbfKernel {
    type Error = SVMError;

    fn try_from(raw_model: &'a ModelFile<'b>) -> Result<RbfKernel, SVMError> {
        let gamma = raw_model.header.gamma.ok_or(SVMError::ModelMissingGamma)?;

        Ok(RbfKernel { gamma })
    }
}
