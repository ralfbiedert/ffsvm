pub(crate) mod class;
pub(crate) mod core;
pub(crate) mod kernel;
pub(crate) mod predict;
pub(crate) mod problem;

use self::kernel::KernelSparse;
use crate::{sparse::SparseVector, vectors::Triangular};

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

pub use self::core::{dense::DenseSVM, sparse::SparseSVM};
