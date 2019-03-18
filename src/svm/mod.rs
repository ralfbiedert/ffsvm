pub(crate) mod class;
pub(crate) mod core;
pub(crate) mod kernel;
pub(crate) mod predict;
pub(crate) mod problem;

use crate::vectors::Triangular;

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
