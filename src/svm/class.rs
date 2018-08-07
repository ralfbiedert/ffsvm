use crate::random::Randomize;
use crate::vectors::{SimdVectorsf32, SimdVectorsf64};

/// Represents one class of the SVM model.
#[derive(Clone, Debug)]
#[doc(hidden)]
pub struct Class {
    /// The label of this class
    pub label: u32,

    /// The number of support vectors in this class
    pub num_support_vectors: usize,

    /// Coefficients between this class and n-1 other classes.
    pub coefficients: SimdVectorsf64,

    /// All support vectors in this class.
    pub support_vectors: SimdVectorsf32,
}

impl Class {
    /// Creates a new class with the given parameters.
    pub fn with_parameters(
        classes: usize,
        support_vectors: usize,
        attributes: usize,
        label: u32,
    ) -> Class {
        Class {
            label,
            num_support_vectors: support_vectors,
            coefficients: SimdVectorsf64::with_dimension(classes - 1, support_vectors),
            support_vectors: SimdVectorsf32::with_dimension(support_vectors, attributes),
        }
    }
}

impl Randomize for Class {
    fn randomize(mut self) -> Self {
        self.coefficients = self.coefficients.randomize();
        self.support_vectors = self.support_vectors.randomize();
        self
    }
}
