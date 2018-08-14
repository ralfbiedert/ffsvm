crate mod class;
crate mod core;
crate mod kernel;
crate mod predict;
crate mod problem;

use self::kernel::{KernelDense, KernelSparse};
use crate::{
    sparse::{SparseMatrix, SparseVector},
    vectors::Triangular,
};

use simd_aligned::{f32s, f64s, RowOptimized, SimdMatrix, SimdVector};

#[derive(Clone, Debug, Default)]
crate struct Probabilities {
    crate a: Triangular<f64>,

    crate b: Triangular<f64>,
}

/// Classifier type.
pub enum SVMType {
    CSvc,
    NuSvc,
    ESvr,
    NuSvr,
}

pub type DenseSVM = core::SVMCore<dyn KernelDense, SimdMatrix<f32s, RowOptimized>, SimdVector<f32s>, SimdVector<f64s>>;

pub type SparseSVM = core::SVMCore<dyn KernelSparse, SparseMatrix<f32>, SparseVector<f32>, SparseVector<f64>>;
