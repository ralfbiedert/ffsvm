use simd_aligned::{f32s, f64s, RowOptimized, SimdMatrix};

/// Represents one class of the SVM model.
#[derive(Clone, Debug)]
#[doc(hidden)]
crate struct Class<M32, M64> {
    /// The label of this class
    crate label: u32,

    /// The number of support vectors in this class
    crate num_support_vectors: usize,

    /// Coefficients between this class and n-1 other classes.
    crate coefficients: M64,

    /// All support vectors in this class.
    crate support_vectors: M32,
}

impl Class<SimdMatrix<f32s, RowOptimized>, SimdMatrix<f64s, RowOptimized>> {
    /// Creates a new class with the given parameters.
    pub fn with_parameters(classes: usize, support_vectors: usize, attributes: usize, label: u32) -> Class<SimdMatrix<f32s, RowOptimized>, SimdMatrix<f64s, RowOptimized>> {
        Class {
            label,
            num_support_vectors: support_vectors,
            coefficients: SimdMatrix::with_dimension(classes - 1, support_vectors),
            support_vectors: SimdMatrix::with_dimension(support_vectors, attributes),
        }
    }
}
