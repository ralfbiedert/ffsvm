use std::{
    marker::PhantomData,
    ops::{Index, IndexMut},
};

use crate::{
    sparse::{SparseMatrix, SparseVector},
    svm::{
        core::SVMCore,
        kernel::{KernelDense, KernelSparse},
    },
    vectors::Triangular,
};

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

#[derive(Debug, Clone)]
pub struct Features<V32> {
    data: V32,
}

/// A single problem a [DenseSVM] or [SparseSVM] should classify.
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
pub struct Problem<V32> {
    /// A vector of all features.
    crate features: Features<V32>,

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
    crate probabilities: SimdVector<f64s>,

    /// Computed label that will be updated after this problem was processed by [Predict].
    crate result: SVMResult,
}

impl<T> Problem<T> {
    pub fn result(&self) -> SVMResult { self.result }

    pub fn probabilities(&self) -> &[f64] { self.probabilities.flat() }

    pub fn probabilities_mut(&mut self) -> &mut [f64] { self.probabilities.flat_mut() }

    pub fn features(&mut self) -> &mut Features<T> { &mut self.features }
}

impl Problem<SimdVector<f32s>> {
    /// Creates a new problem with the given parameters.
    crate fn with_dimension(total_sv: usize, num_classes: usize, num_attributes: usize) -> Problem<SimdVector<f32s>> {
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
            result: SVMResult::None,
        }
    }
}

impl Problem<SparseVector<f32>> {
    /// Creates a new problem with the given parameters.
    crate fn with_dimension(total_sv: usize, num_classes: usize, num_attributes: usize) -> Problem<SparseVector<f32>> {
        Problem {
            features: Features { data: SparseVector::new() },
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
}

impl<'a> From<&'a SVMCore<KernelDense, SimdMatrix<f32s, RowOptimized>, SimdVector<f32s>, SimdVector<f64s>>> for Problem<SimdVector<f32s>> {
    fn from(svm: &SVMCore<KernelDense, SimdMatrix<f32s, RowOptimized>, SimdVector<f32s>, SimdVector<f64s>>) -> Self {
        Problem::<SimdVector<f32s>>::with_dimension(svm.num_total_sv, svm.classes.len(), svm.num_attributes)
    }
}

impl<'a> From<&'a SVMCore<KernelSparse, SparseMatrix<f32>, SparseVector<f32>, SparseVector<f64>>> for Problem<SparseVector<f32>> {
    fn from(svm: &SVMCore<KernelSparse, SparseMatrix<f32>, SparseVector<f32>, SparseVector<f64>>) -> Self {
        Problem::<SparseVector<f32>>::with_dimension(svm.num_total_sv, svm.classes.len(), svm.num_attributes)
    }
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
