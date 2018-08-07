use crate::kernel::Kernel;
use crate::random::{random_vec, Randomize};
use crate::simd_matrix::{f32s, f64s, SimdRows};
use crate::svm::SVM;
use crate::vectors::Triangular;

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
/// # Classifiying a problem
///
/// A problem is an instance of the SVM's problem domain. To be classified, all `features` need
/// to be set, for example by:
///
/// ```ignore
/// problem.features = vec![
///     -0.55838, -0.157895, 0.581292, -0.221184, 0.135713, -0.874396, -0.563197, -1.0, -1.0,
/// ];
/// ```
///
/// It can then be handed over to the [SVM] (via the [PredictProblem] trait).
///
#[derive(Debug, Clone)]
pub struct Problem {
    /// A vector of all features.
    pub(crate) features: SimdRows<f32s>,

    /// Kernel values. A vector for each class.
    pub(crate) kernel_values: SimdRows<f64s>,

    /// All votes for a given class label.
    pub(crate) vote: Vec<u32>,

    /// Decision values.
    pub(crate) decision_values: Triangular<f64>,

    /// Pairwise probabilities
    pub(crate) pairwise: SimdRows<f64s>,

    /// Needed for multi-class probability estimates replicating libSVM.
    pub(crate) q: SimdRows<f64s>,

    /// Needed for multi-class probability estimates replicating libSVM.
    pub(crate) qp: Vec<f64>,

    /// Probability estimates that will be updated after this problem was processed
    /// by `predict_probability` in [PredictProblem] if the model supports it.
    pub(crate) probabilities: Vec<f64>,

    /// Computed label that will be updated after this problem was processed by [PredictProblem].
    pub(crate) label: u32,
}

impl Problem {
    /// Creates a new problem with the given parameters.
    pub(crate) fn with_dimension(
        total_sv: usize,
        num_classes: usize,
        num_attributes: usize,
    ) -> Problem {
        Problem {
            features: SimdRows::with_dimension(1, num_attributes),
            kernel_values: SimdRows::with_dimension(num_classes, total_sv),
            pairwise: SimdRows::with_dimension(num_classes, num_classes),
            q: SimdRows::with_dimension(num_classes, num_classes),
            qp: vec![Default::default(); num_classes],
            decision_values: Triangular::with_dimension(num_classes, Default::default()),
            vote: vec![Default::default(); num_classes],
            probabilities: vec![Default::default(); num_classes],
            label: 0,
        }
    }

    pub fn features_mut(&mut self) -> &mut [f32] {
        self.features.as_slice_mut()
    }

    pub fn probabilities(&self) -> &[f64] {
        &self.probabilities
    }

    pub fn label(&self) -> u32 {
        self.label
    }
}

impl<'a, T> From<&'a SVM<T>> for Problem
where
    T: Kernel,
{
    fn from(svm: &SVM<T>) -> Self {
        Problem::with_dimension(svm.num_total_sv, svm.classes.len(), svm.num_attributes)
    }
}

impl Randomize for Problem {
    fn randomize(mut self) -> Self {
        self.features = self.features.randomize();
        self
    }
}
