use std::convert::{From, TryFrom};

use super::{KernelDense, KernelSparse};
use crate::{
    errors::Error,
    f32s,
    parser::ModelFile,
    sparse::{SparseMatrix, SparseVector},
};

use simd_aligned::{MatrixD, Rows, VectorD};

#[derive(Copy, Clone, Debug, Default)]
#[doc(hidden)]
pub struct Rbf {
    pub gamma: f32,
}

#[inline]
fn compute_core(rbf: Rbf, vectors: &MatrixD<f32s, Rows>, feature: &VectorD<f32s>, output: &mut [f64]) {
    // According to Instruments, for realistic SVMs and problems, the VAST majority of our
    // CPU time is spent in this loop.
    for (i, sv) in vectors.row_iter().enumerate() {
        let mut sum = f32s::splat(0.0);
        let feature: &[f32s] = feature;

        for (a, b) in sv.iter().zip(feature) {
            sum += (*a - *b) * (*a - *b);
        }

        // This seems to be the single-biggest CPU spike: saving back kernel_values,
        // and computing exp() (saving back seems to have 3x time impact over exp(),
        // but I might misread "Instruments" for that particular one).
        output[i] = f64::from((-rbf.gamma * sum.sum()).exp());
    }
}

#[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
#[target_feature(enable = "avx")]
#[inline]
unsafe fn compute_avx(rbf: Rbf, vectors: &MatrixD<f32s, Rows>, feature: &VectorD<f32s>, output: &mut [f64]) { compute_core(rbf, vectors, feature, output); }

#[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
#[target_feature(enable = "avx2")]
#[inline]
unsafe fn compute_avx2(rbf: Rbf, vectors: &MatrixD<f32s, Rows>, feature: &VectorD<f32s>, output: &mut [f64]) { compute_core(rbf, vectors, feature, output); }

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
#[inline]
unsafe fn compute_neon(rbf: Rbf, vectors: &MatrixD<f32s, Rows>, feature: &VectorD<f32s>, output: &mut [f64]) { compute_core(rbf, vectors, feature, output); }

#[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
#[inline]
fn compute(rbf: Rbf, vectors: &MatrixD<f32s, Rows>, feature: &VectorD<f32s>, output: &mut [f64]) {
    if is_x86_feature_detected!("avx2") {
        unsafe { compute_avx2(rbf, vectors, feature, output) }
    } else if is_x86_feature_detected!("avx") {
        unsafe { compute_avx(rbf, vectors, feature, output) }
    } else {
        compute_core(rbf, vectors, feature, output)
    }
}

#[cfg(target_arch = "aarch64")]
#[inline]
fn compute(rbf: Rbf, vectors: &MatrixD<f32s, Rows>, feature: &VectorD<f32s>, output: &mut [f64]) {
    if is_aarch64_feature_detected!("neon") {
        unsafe { compute_neon(rbf, vectors, feature, output) }
    } else {
        compute_core(rbf, vectors, feature, output)
    }
}

#[cfg(not(any(target_arch = "x86_64", target_arch = "x86", target_arch = "aarch64")))]
#[inline]
fn compute(rbf: Rbf, vectors: &MatrixD<f32s, Rows>, feature: &VectorD<f32s>, output: &mut [f64]) { compute_core(rbf, vectors, feature, output) }

impl KernelDense for Rbf {
    fn compute(&self, vectors: &MatrixD<f32s, Rows>, feature: &VectorD<f32s>, output: &mut [f64]) { compute(*self, vectors, feature, output); }
}

impl KernelSparse for Rbf {
    fn compute(&self, vectors: &SparseMatrix<f32>, feature: &SparseVector<f32>, output: &mut [f64]) {
        for (i, sv) in vectors.row_iter().enumerate() {
            let mut sum = 0.0;
            let mut a_iter = sv.iter();
            let mut b_iter = feature.iter();

            let (mut a, mut b) = (a_iter.next(), b_iter.next());

            output[i] = loop {
                match (a, b) {
                    (Some((i_a, x)), Some((i_b, y))) if i_a == i_b => {
                        sum += (x - y) * (x - y);

                        a = a_iter.next();
                        b = b_iter.next();
                    }
                    (Some((i_a, _)), Some((i_b, _))) if i_a < i_b => a = a_iter.next(),
                    (Some((i_a, _)), Some((i_b, _))) if i_a > i_b => b = b_iter.next(),
                    _ => break f64::from((-self.gamma * sum).exp()),
                }
            }
        }
    }
}

impl<'a, 'b> TryFrom<&'a ModelFile<'b>> for Rbf {
    type Error = Error;

    fn try_from(raw_model: &'a ModelFile<'b>) -> Result<Self, Error> {
        let gamma = raw_model.header.gamma.ok_or(Error::NoGamma)?;

        Ok(Self { gamma })
    }
}
