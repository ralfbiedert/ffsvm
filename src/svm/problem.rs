use std::ops::{Index, IndexMut};

use crate::{
    sparse::SparseVector,
    svm::{DenseSVM, SparseSVM},
    vectors::Triangular,
};

use simd_aligned::{f32s, f64s, RowOptimized, SimdMatrix, SimdVector};

/// Problems produced for [`DenseSVM`]s.
///
/// Also see [`Problem`] for more methods for this type.
pub type DenseProblem = Problem<SimdVector<f32s>>;

/// Problems produced for [`SparseSVM`]s.
///
/// Also see [`Problem`] for more methods for this type.
pub type SparseProblem = Problem<SparseVector<f32>>;

/// The result of a classification
#[derive(Copy, Debug, Clone, PartialEq)]
pub enum Solution {
    /// If classified this will hold the label.
    Label(i32),

    /// If regression was performed contains regression result.
    Value(f32),

    /// No operation was performed yet.
    None,
}

#[derive(Debug, Clone)]
pub struct Features<V32> {
    data: V32,
}

/// A single problem a SVM should classify.
///
/// # Creating a `Problem`
///
/// Problems are created via the `Problem::from` method and match the SVM type they were created for:
///
/// ```rust
/// #![feature(try_from)]
///
/// use ffsvm::*;
/// use std::convert::TryFrom;
///
/// fn main() -> Result<(), Error> {
///     let svm = DenseSVM::try_from(SAMPLE_MODEL)?;
///
///     let mut problem = Problem::from(&svm);
///
///     Ok(())
/// }
/// ```
///
/// # Setting Features
///
/// A `Problem` is an instance of the SVM's problem domain. Before it can be classified, all `features` need
/// to be set, for example by:
///
/// ```
/// use ffsvm::*;
///
/// fn set_features(problem: &mut DenseProblem) {
///     let features = problem.features();
///     features[0] = -0.221184;
///     features[3] = 0.135713;
/// }
/// ```
///
/// It can then be classified via the [`Predict`](crate::Predict) trait.
///
#[derive(Debug, Clone)]
pub struct Problem<V32> {
    /// A vector of all features.
    pub(crate) features: Features<V32>,

    /// KernelDense values. A vector for each class.
    pub(crate) kernel_values: SimdMatrix<f64s, RowOptimized>,

    /// All votes for a given class label.
    pub(crate) vote: Vec<u32>,

    /// Decision values.
    pub(crate) decision_values: Triangular<f64>,

    /// Pairwise probabilities
    pub(crate) pairwise: SimdMatrix<f64s, RowOptimized>,

    /// Needed for multi-class probability estimates replicating libSVM.
    pub(crate) q: SimdMatrix<f64s, RowOptimized>,

    /// Needed for multi-class probability estimates replicating libSVM.
    pub(crate) qp: Vec<f64>,

    /// Probability estimates that will be updated after this problem was processed
    /// by `predict_probability`.
    pub(crate) probabilities: SimdVector<f64s>,

    /// Computed label that will be updated after this problem was processed.
    pub(crate) result: Solution,
}

impl<T> Problem<T> {
    /// After a [`Problem`](crate::Problem) has been classified, this will hold the SVMs solution.
    pub fn solution(&self) -> Solution { self.result }

    /// Returns the probability estimates. Only really useful if the model was trained with probability estimates and you classified with them.
    pub fn probabilities(&self) -> &[f64] { self.probabilities.flat() }

    /// Returns the features. You must set them first and classifiy the problem before you can get a solution.
    pub fn features(&mut self) -> &mut Features<T> { &mut self.features }
}

impl DenseProblem {
    /// Creates a new problem with the given parameters.
    pub(crate) fn with_dimension(total_sv: usize, num_classes: usize, num_attributes: usize) -> Problem<SimdVector<f32s>> {
        Problem {
            features: Features {
                data: SimdVector::with(0.0, num_attributes),
            },
            kernel_values: SimdMatrix::with_dimension(num_classes, total_sv),
            pairwise: SimdMatrix::with_dimension(num_classes, num_classes),
            q: SimdMatrix::with_dimension(num_classes, num_classes),
            qp: vec![Default::default(); num_classes],
            decision_values: Triangular::with_dimension(num_classes, Default::default()),
            vote: vec![Default::default(); num_classes],
            probabilities: SimdVector::with(0.0, num_classes),
            result: Solution::None,
        }
    }
}

impl SparseProblem {
    /// Clears the [Problem] when reusing it between calls. Only needed for [SparseSVM] problems.
    pub fn clear(&mut self) { self.features.data.clear(); }

    /// Creates a new problem with the given parameters.
    pub(crate) fn with_dimension(total_sv: usize, num_classes: usize, _num_attributes: usize) -> Problem<SparseVector<f32>> {
        Problem {
            features: Features { data: SparseVector::new() },
            kernel_values: SimdMatrix::with_dimension(num_classes, total_sv),
            pairwise: SimdMatrix::with_dimension(num_classes, num_classes),
            q: SimdMatrix::with_dimension(num_classes, num_classes),
            qp: vec![Default::default(); num_classes],
            decision_values: Triangular::with_dimension(num_classes, Default::default()),
            vote: vec![Default::default(); num_classes],
            probabilities: SimdVector::with(0.0, num_classes),
            result: Solution::None,
        }
    }
}

impl<'a> From<&'a DenseSVM> for DenseProblem {
    fn from(svm: &DenseSVM) -> Self { Problem::<SimdVector<f32s>>::with_dimension(svm.num_total_sv, svm.classes.len(), svm.num_attributes) }
}

impl<'a> From<&'a SparseSVM> for SparseProblem {
    fn from(svm: &SparseSVM) -> Self { Problem::<SparseVector<f32>>::with_dimension(svm.num_total_sv, svm.classes.len(), svm.num_attributes) }
}

impl<V32> Features<V32> {
    pub fn as_raw(&self) -> &V32 { &self.data }
}

impl Features<SimdVector<f32s>> {
    pub fn as_slice_mut(&mut self) -> &mut [f32] { self.data.flat_mut() }
}

impl Index<usize> for Features<SimdVector<f32s>> // where
{
    type Output = f32;

    fn index(&self, index: usize) -> &f32 { &self.data.flat()[index] }
}

impl IndexMut<usize> for Features<SimdVector<f32s>> {
    fn index_mut(&mut self, index: usize) -> &mut f32 { &mut self.data.flat_mut()[index] }
}

impl Index<usize> for Features<SparseVector<f32>> // where
{
    type Output = f32;

    fn index(&self, index: usize) -> &f32 { &self.data[index] }
}

impl IndexMut<usize> for Features<SparseVector<f32>> {
    fn index_mut(&mut self, index: usize) -> &mut f32 { &mut self.data[index] }
}
