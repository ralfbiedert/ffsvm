pub(crate) mod class;
pub(crate) mod core;
pub(crate) mod kernel;
pub(crate) mod predict;
pub(crate) mod problem;

use self::kernel::{KernelDense, KernelSparse};
use crate::{
    sparse::{SparseMatrix, SparseVector},
    vectors::Triangular,
};

use simd_aligned::{f32s, f64s, RowOptimized, SimdMatrix, SimdVector};

#[derive(Clone, Debug, Default)]
pub(crate) struct Probabilities {
    pub(crate) a: Triangular<f64>,

    pub(crate) b: Triangular<f64>,
}

/// Classifier type.
#[doc(hidden)]
pub enum SVMType {
    CSvc,
    NuSvc,
    ESvr,
    NuSvr,
}

/// **Start here** to classify dense models with highest performance.
pub type DenseSVM = core::SVMCore<dyn KernelDense, SimdMatrix<f32s, RowOptimized>, SimdVector<f32s>, SimdVector<f64s>>;

/// Use this to load any `libSVM` model with normal performance.
pub type SparseSVM = core::SVMCore<dyn KernelSparse, SparseMatrix<f32>, SparseVector<f32>, SparseVector<f64>>;
