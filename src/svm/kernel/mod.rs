mod linear;
mod poly;
mod rbf;
mod sigmoid;

use crate::sparse::{SparseMatrix, SparseVector};
use crate::f32s;
use simd_aligned::{Rows, MatrixD, VectorD};

pub use self::{linear::*, poly::*, rbf::*, sigmoid::*};

/// Base trait for kernels
#[doc(hidden)]
pub trait KernelDense
where
    Self: Sync,
{
    fn compute(&self, vectors: &MatrixD<f32s, Rows>, feature: &VectorD<f32s>, output: &mut [f64]);
}

/// Base trait for kernels
#[doc(hidden)]
pub trait KernelSparse
where
    Self: Sync,
{
    fn compute(&self, vectors: &SparseMatrix<f32>, feature: &SparseVector<f32>, output: &mut [f64]);
}
