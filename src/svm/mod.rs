pub mod class;
pub mod core;
pub mod features;
pub mod kernel;
pub mod predict;

use crate::vectors::Triangular;

#[derive(Clone, Debug, Default)]
pub struct Probabilities {
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
