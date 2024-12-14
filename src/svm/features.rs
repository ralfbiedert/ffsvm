use crate::{
    sparse::SparseVector,
    svm::{DenseSVM, SparseSVM},
    vectors::Triangular,
};

use simd_aligned::{
    arch::{f32x8, f64x4},
    MatSimd, Rows, VecSimd,
};

/// Feature vectors produced for [`DenseSVM`]s.
///
/// Also see [`FeatureVector`] for more methods for this type.
pub type DenseFeatures = FeatureVector<VecSimd<f32x8>>;

/// Feature vectors produced for [`SparseSVM`]s.
///
/// Also see [`FeatureVector`] for more methods for this type.
pub type SparseFeatures = FeatureVector<SparseVector<f32>>;

/// The result of a classification
#[derive(Copy, Debug, Clone, PartialEq)]
pub enum Label {
    /// If classified this will hold the label.
    Class(i32),

    /// If regression was performed contains regression result.
    Value(f32),

    /// No operation was performed yet.
    None,
}

/// A single feature vector ("problem") an SVM should classify.
///
/// # Creating a feature vector:
///
/// Feature vectors are created via the [`FeatureVector::from`] method and match the SVM type they were
/// created for, so their layout matches the SVM:
///
/// ```rust
/// use ffsvm::{Error, DenseSVM, FeatureVector};
/// # use ffsvm::SAMPLE_MODEL;
///
/// # fn main() -> Result<(), Error> {
/// let svm = DenseSVM::try_from(SAMPLE_MODEL)?;
/// let mut fv = FeatureVector::from(&svm);
/// # Ok(())
/// # }
/// ```
///
/// # Setting Features
///
/// A [`FeatureVector`] is an instance of the SVM's problem domain. Before it can be classified, all `features` need
/// to be set, for example by:
///
/// ```
/// use ffsvm::DenseFeatures;
///
/// fn set_features(problem: &mut DenseFeatures) {
///     let features = problem.features();
///     features[0] = -0.221184;
///     features[3] = 0.135713;
/// }
/// ```
///
/// It can then be classified via the [`Predict`](crate::Predict) trait.
#[derive(Debug, Clone)]
pub struct FeatureVector<T> {
    /// A vector of all features.
    pub(crate) features: T,

    /// KernelDense values. A vector for each class.
    pub(crate) kernel_values: MatSimd<f64x4, Rows>,

    /// All votes for a given class label.
    pub(crate) vote: Vec<u32>,

    /// Decision values.
    pub(crate) decision_values: Triangular<f64>,

    /// Pairwise probabilities
    pub(crate) pairwise: MatSimd<f64x4, Rows>,

    /// Needed for multi-class probability estimates replicating libSVM.
    pub(crate) q: MatSimd<f64x4, Rows>,

    /// Needed for multi-class probability estimates replicating libSVM.
    pub(crate) qp: Vec<f64>,

    /// Probability estimates that will be updated after this problem was processed
    /// by `predict_probability`.
    pub(crate) probabilities: VecSimd<f64x4>,

    /// Computed label that will be updated after this problem was processed.
    pub(crate) result: Label,
}

impl<T> FeatureVector<T> {
    /// After a [`Problem`](crate::FeatureVector) has been classified, this will hold the SVMs solution label.
    pub const fn label(&self) -> Label {
        self.result
    }

    /// Returns the probability estimates. Only really useful if the model was trained with probability estimates and you classified with them.
    pub fn probabilities(&self) -> &[f64] {
        self.probabilities.flat()
    }
}

impl FeatureVector<VecSimd<f32x8>> {
    /// Returns the features. You must set them first and classify the problem before you can get a solution.
    pub fn features(&mut self) -> &mut [f32] {
        self.features.flat_mut()
    }
}

impl FeatureVector<SparseVector<f32>> {
    /// Returns the features. You must set them first and classify the problem before you can get a solution.
    pub fn features(&mut self) -> &mut SparseVector<f32> {
        &mut self.features
    }
}

impl DenseFeatures {
    /// Creates a new problem with the given parameters.
    pub(crate) fn with_dimension(total_sv: usize, num_classes: usize, num_attributes: usize) -> Self {
        Self {
            features: VecSimd::with(0.0, num_attributes),
            kernel_values: MatSimd::with_dimension(num_classes, total_sv),
            pairwise: MatSimd::with_dimension(num_classes, num_classes),
            q: MatSimd::with_dimension(num_classes, num_classes),
            qp: vec![Default::default(); num_classes],
            decision_values: Triangular::with_dimension(num_classes, Default::default()),
            vote: vec![Default::default(); num_classes],
            probabilities: VecSimd::with(0.0, num_classes),
            result: Label::None,
        }
    }
}

impl SparseFeatures {
    /// Clears the [FeatureVector] when reusing it between calls. Only needed for [SparseSVM] problems.
    pub fn clear(&mut self) {
        self.features.clear();
    }

    /// Creates a new problem with the given parameters.
    pub(crate) fn with_dimension(total_sv: usize, num_classes: usize, _num_attributes: usize) -> Self {
        Self {
            features: SparseVector::new(),
            kernel_values: MatSimd::with_dimension(num_classes, total_sv),
            pairwise: MatSimd::with_dimension(num_classes, num_classes),
            q: MatSimd::with_dimension(num_classes, num_classes),
            qp: vec![Default::default(); num_classes],
            decision_values: Triangular::with_dimension(num_classes, Default::default()),
            vote: vec![Default::default(); num_classes],
            probabilities: VecSimd::with(0.0, num_classes),
            result: Label::None,
        }
    }
}

impl From<&DenseSVM> for DenseFeatures {
    fn from(svm: &DenseSVM) -> Self {
        Self::with_dimension(svm.num_total_sv, svm.classes.len(), svm.num_attributes)
    }
}

impl From<&SparseSVM> for SparseFeatures {
    fn from(svm: &SparseSVM) -> Self {
        Self::with_dimension(svm.num_total_sv, svm.classes.len(), svm.num_attributes)
    }
}
