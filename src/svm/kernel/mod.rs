mod linear;
mod poly;
mod rbf;
mod sigmoid;

pub use self::{linear::*, poly::*, rbf::*, sigmoid::*};
use crate::sparse::{SparseMatrix, SparseVector};
use simd_aligned::{arch::f32x8, MatSimd, Rows, VecSimd};

/// Base trait for kernels
#[doc(hidden)]
pub trait KernelDense
where
    Self: Send + Sync,
{
    fn compute(&self, vectors: &MatSimd<f32x8, Rows>, feature: &VecSimd<f32x8>, output: &mut [f64]);
}

/// Base trait for kernels
#[doc(hidden)]
pub trait KernelSparse
where
    Self: Send + Sync,
{
    fn compute(&self, vectors: &SparseMatrix<f32>, feature: &SparseVector<f32>, output: &mut [f64]);
}
