pub mod crbf;
pub mod problem;

use vectors::flat::ManyVectors;

/// Core support vector machine
#[derive(Debug)]
pub struct SVM<T> {
    /// Total number of support vectors
    pub num_total_sv: usize,

    /// Number of attributes per support vector
    pub num_attributes: usize,

    pub rho: Vec<f64>,

    /// SVM specific data needed for classification
    pub kernel: T,

    /// All classes
    pub classes: Vec<Class>,
}



/// Represents one class of the SVM model.
#[derive(Debug)]
pub struct Class {
    /// The label of this class
    pub label: u32,

    /// The number of support vectors in this class
    pub num_support_vectors: usize,

    /// Coefficients between this class and n-1 other classes.
    pub coefficients: ManyVectors<f64>,

    /// All support vectors in this class.
    pub support_vectors: ManyVectors<f32>,
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
            coefficients: ManyVectors::with_dimension(
                classes - 1,
                support_vectors,
                Default::default(),
            ),
            support_vectors: ManyVectors::with_dimension(
                support_vectors,
                attributes,
                Default::default(),
            ),
        }
    }
}


