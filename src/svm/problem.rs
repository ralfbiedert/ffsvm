use crate::{random::Randomize, svm::core::SVMCore, vectors::Triangular};

use simd_aligned::{f32s, f64s, RowOptimized, SimdMatrix, SimdVector};

/// The result of a classification
#[derive(Copy, Debug, Clone, PartialEq)]
pub enum SVMResult {
    /// If classified this will hold the label.
    Label(u32),

    /// If regression was performed contains regression result.
    Value(f32),

    /// No operation was performed yet.
    None,
}

/// A single problem a [SVM] should classify.
///
/// # Creating a problem
///
/// Problems are created via the `Problem::from` method:
///
/// ```ignore
/// let mut problem = Problem::from(&svm); 
/// ```
///
/// # Classifying a problem
///
/// A problem is an instance of the SVM's problem domain. To be classified, all `features` need
/// to be set, for example by:
///
/// ```ignore
/// problem.features = vec![-0.55838, -0.157895, 0.581292, -0.221184, 0.135713, -0.874396, -0.563197, -1.0, -1.0]; 
/// ```
///
/// It can then be handed over to the [SVM] (via the [Predict] trait).
///
#[derive(Debug, Clone)]
pub struct Problem<V32, V64> {
    /// A vector of all features.
    crate features: V32,

    /// KernelDense values. A vector for each class.
    crate kernel_values: SimdMatrix<f64s, RowOptimized>,

    /// All votes for a given class label.
    crate vote: Vec<u32>,

    /// Decision values.
    crate decision_values: Triangular<f64>,

    /// Pairwise probabilities
    crate pairwise: SimdMatrix<f64s, RowOptimized>,

    /// Needed for multi-class probability estimates replicating libSVM.
    crate q: SimdMatrix<f64s, RowOptimized>,

    /// Needed for multi-class probability estimates replicating libSVM.
    crate qp: Vec<f64>,

    /// Probability estimates that will be updated after this problem was processed
    /// by `predict_probability` in [Predict] if the model supports it.
    crate probabilities: V64,

    /// Computed label that will be updated after this problem was processed by [Predict].
    crate result: SVMResult,
}

impl<T, R> Problem<T, R> {
    pub fn result(&self) -> SVMResult { self.result }
}

impl Problem<SimdVector<f32s>, SimdVector<f64s>> {
    /// Creates a new problem with the given parameters.
    crate fn with_dimension(total_sv: usize, num_classes: usize, num_attributes: usize) -> Problem<SimdVector<f32s>, SimdVector<f64s>> {
        Problem {
            features: SimdVector::with(0.0, num_attributes),
            kernel_values: SimdMatrix::with_dimension(num_classes, total_sv),
            pairwise: SimdMatrix::with_dimension(num_classes, num_classes),
            q: SimdMatrix::with_dimension(num_classes, num_classes),
            qp: vec![Default::default(); num_classes],
            decision_values: Triangular::with_dimension(num_classes, Default::default()),
            vote: vec![Default::default(); num_classes],
            probabilities: SimdVector::with(0.0, num_classes),
            result: SVMResult::None,
        }
    }

    pub fn features_mut(&mut self) -> &mut [f32] { self.features.flat_mut() }

    pub fn probabilities(&self) -> &[f64] { self.probabilities.flat() }

    pub fn probabilities_mut(&mut self) -> &mut [f64] { self.probabilities.flat_mut() }
}

impl<'a> From<&'a SVMCore<SimdMatrix<f64s, RowOptimized>, SimdMatrix<f32s, RowOptimized>, SimdVector<f32s>, SimdVector<f64s>>> for Problem<SimdVector<f32s>, SimdVector<f64s>> {
    fn from(svm: &SVMCore<SimdMatrix<f64s, RowOptimized>, SimdMatrix<f32s, RowOptimized>, SimdVector<f32s>, SimdVector<f64s>>) -> Self {
        Problem::with_dimension(svm.num_total_sv, svm.classes.len(), svm.num_attributes)
    }
}

impl Randomize for Problem<SimdVector<f32s>, SimdVector<f64s>> {
    fn randomize(mut self) -> Self {
        self.features = self.features.randomize();
        self
    }
}
