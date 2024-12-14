mod linear;
mod poly;
mod rbf;
mod sigmoid;

use crate::{
    sparse::{SparseMatrix, SparseVector},
};
use simd_aligned::{f32x8, MatD, Rows, VecD};

pub use self::{linear::*, poly::*, rbf::*, sigmoid::*};

/// Base trait for kernels
#[doc(hidden)]
pub trait KernelDense
where
    Self: Send + Sync,
{
    fn compute(&self, vectors: &MatD<f32x8, Rows>, feature: &VecD<f32x8>, output: &mut [f64]);
}

/// Base trait for kernels
#[doc(hidden)]
pub trait KernelSparse
where
    Self: Send + Sync,
{
    fn compute(&self, vectors: &SparseMatrix<f32>, feature: &SparseVector<f32>, output: &mut [f64]);
}
