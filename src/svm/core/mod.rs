mod dense;
mod sparse;

use simd_aligned::{f32s, f64s, RowOptimized, SimdMatrix, SimdVector};
use std::{convert::TryFrom, marker::PhantomData};

use crate::{
    errors::SVMError,
    parser::ModelFile,
    random::*,
    svm::{
        class::Class,
        kernel::{KernelDense, Linear, Poly, Rbf, Sigmoid},
        predict::Predict,
        problem::{Problem, SVMResult},
        Probabilities, SVMType,
    },
    util::{find_max_index, set_all, sigmoid_predict},
    vectors::Triangular,
};

/// Generic support vector machine, template for [RbfSVM].
///
/// The SVM holds a kernel, class information and all other numerical data read from
/// the [ModelFile]. It implements [Predict] to predict [Problem] instances.
///
/// # Creating a SVM
///
/// The only SVM currently implemented is the [RbfSVM]. It can be constructed from a
/// [ModelFile] like this:
///
/// ```ignore
/// let svm = RbfSVM::try_from(&model)!;
/// ```
///
pub struct SVMCore<K, M64, M32, V32, V64>
where
    K: ?Sized,
{
    /// Total number of support vectors
    crate num_total_sv: usize,

    /// Number of attributes per support vector
    crate num_attributes: usize,

    crate rho: Triangular<f64>,

    crate probabilities: Option<Probabilities>,

    crate svm_type: SVMType,

    /// SVM specific data needed for classification
    crate kernel: Box<K>,

    /// All classes
    crate classes: Vec<Class<M32, M64>>,

    phantomV32: PhantomData<V32>,

    phantomV64: PhantomData<V64>,
}

impl<K, M64, M32, V32, V64> SVMCore<K, M64, M32, V32, V64> {
    /// Finds the class index for a given label.
    ///
    /// # Description
    ///
    /// This method takes a `label` as defined in the libSVM training model
    /// and returns the internal `index` where this label resides. The index
    /// equals the [Problem]'s `.probabilities` index where that label's
    /// probability can be found.
    ///
    /// # Returns
    ///
    /// If the label was found its index returned in the [Option]. Otherwise `None`
    /// is returned.
    ///
    pub fn class_index_for_label(&self, label: u32) -> Option<usize> {
        for (i, class) in self.classes.iter().enumerate() {
            if class.label != label {
                continue;
            }

            return Some(i);
        }

        None
    }

    /// Returns the class label for a given index.
    ///
    /// # Description
    ///
    /// The inverse of [SVM::class_index_for_label], this function returns the class label
    /// associated with a certain internal index. The index equals the [Problem]'s
    /// `.probabilities` index where a label's probability can be found.
    ///
    /// # Returns
    ///
    /// If the index was found it is returned in the [Option]. Otherwise `None`
    /// is returned.
    pub fn class_label_for_index(&self, index: usize) -> Option<u32> {
        if index >= self.classes.len() {
            None
        } else {
            Some(self.classes[index].label)
        }
    }

    /// Returns number of attributes, reflecting the libSVM model.
    pub fn attributes(&self) -> usize { self.num_attributes }

    /// Returns number of classes, reflecting the libSVM model.
    pub fn classes(&self) -> usize { self.classes.len() }
}
