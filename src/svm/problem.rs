use crate::kernel::Kernel;
use crate::random::{random_vec, Randomize};
use crate::svm::SVM;
use crate::vectors::{SimdVectorsf64, Triangular};

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
    pub features: Vec<f32>,

    /// Kernel values. A vector for each class.
    pub(crate) kernel_values: SimdVectorsf64,

    /// All votes for a given class label.
    pub(crate) vote: Vec<u32>,

    /// Decision values.
    pub(crate) decision_values: Triangular<f64>,

    /// Pairwise probabilities
    pub(crate) pairwise: SimdVectorsf64,

    /// Needed for multi-class probability estimates replicating libSVM.
    pub(crate) q: SimdVectorsf64,

    /// Needed for multi-class probability estimates replicating libSVM.
    pub(crate) qp: Vec<f64>,

    /// Probability estimates that will be updated after this problem was processed
    /// by `predict_probability` in [PredictProblem] if the model supports it.
    pub probabilities: Vec<f64>,

    /// Computed label that will be updated after this problem was processed by [PredictProblem].
    pub label: u32,
}

impl Problem {
    /// Creates a new problem with the given parameters.
    pub(crate) fn with_dimension(
        total_sv: usize,
        num_classes: usize,
        num_attributes: usize,
    ) -> Problem {
        Problem {
            features: vec![Default::default(); num_attributes],
            kernel_values: SimdVectorsf64::with_dimension(num_classes, total_sv),
            pairwise: SimdVectorsf64::with_dimension(num_classes, num_classes),
            q: SimdVectorsf64::with_dimension(num_classes, num_classes),
            qp: vec![Default::default(); num_classes],
            decision_values: Triangular::with_dimension(num_classes, Default::default()),
            vote: vec![Default::default(); num_classes],
            probabilities: vec![Default::default(); num_classes],
            label: 0,
        }
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
        self.features = random_vec(self.features.len());
        self
    }
}
