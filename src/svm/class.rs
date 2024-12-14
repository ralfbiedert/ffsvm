use crate::sparse::SparseMatrix;
use simd_aligned::{
    arch::{f32x8, f64x4},
    MatSimd, Rows,
};

/// Represents one class of the SVM model.
#[derive(Clone, Debug)]
#[doc(hidden)]
pub struct Class<M32> {
    /// The label of this class
    pub(crate) label: i32,

    // /// The number of support vectors in this class
    // pub(crate) num_support_vectors: usize,
    /// Coefficients between this class and n-1 other classes.
    pub(crate) coefficients: MatSimd<f64x4, Rows>,

    /// All support vectors in this class.
    pub(crate) support_vectors: M32,
}

impl Class<MatSimd<f32x8, Rows>> {
    /// Creates a new class with the given parameters.
    pub fn with_parameters(classes: usize, support_vectors: usize, attributes: usize, label: i32) -> Self {
        Self {
            label,
            coefficients: MatSimd::with_dimension(classes - 1, support_vectors),
            support_vectors: MatSimd::with_dimension(support_vectors, attributes),
        }
    }
}

impl Class<SparseMatrix<f32>> {
    /// Creates a new class with the given parameters.
    pub fn with_parameters(classes: usize, support_vectors: usize, _attributes: usize, label: i32) -> Self {
        Self {
            label,
            coefficients: MatSimd::with_dimension(classes - 1, support_vectors),
            support_vectors: SparseMatrix::with(support_vectors),
        }
    }
}
