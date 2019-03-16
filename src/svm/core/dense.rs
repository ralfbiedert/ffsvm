use simd_aligned::{f32s, f64s, RowOptimized, SimdMatrix, SimdVector};
use std::{convert::TryFrom, marker::PhantomData};

use crate::{
    errors::Error,
    parser::ModelFile,
    svm::{
        class::Class,
        kernel::{KernelDense, Linear, Poly, Rbf, Sigmoid},
        predict::Predict,
        problem::{Problem, Solution},
        Probabilities, SVMType,
    },
    util::{find_max_index, set_all, sigmoid_predict},
    vectors::Triangular,
};

/// A SVM using [SIMD](https://en.wikipedia.org/wiki/SIMD) intrinsics optimized for speed.
///
///
/// # Creating a SVM
///
/// This SVM can be created by passing a [`ModelFile`](crate::ModelFile) into `try_from`, or a `&str`:
///
/// ```
/// use ffsvm::*;
/// use std::convert::TryFrom;
///
/// let svm = DenseSVM::try_from("...");
/// ```
pub struct DenseSVM {
    /// Total number of support vectors
    pub(crate) num_total_sv: usize,

    /// Number of attributes per support vector
    pub(crate) num_attributes: usize,

    pub(crate) rho: Triangular<f64>,

    pub(crate) probabilities: Option<Probabilities>,

    pub(crate) svm_type: SVMType,

    /// SVM specific data needed for classification
    pub(crate) kernel: Box<dyn KernelDense>,

    /// All classes
    pub(crate) classes: Vec<Class<SimdMatrix<f32s, RowOptimized>>>,
}

impl DenseSVM {
    /// Finds the class index for a given label.
    ///
    /// # Description
    ///
    /// This method takes a `label` as defined in the libSVM training model
    /// and returns the internal `index` where this label resides. The index
    /// equals [`Problem::probabilities`] index where that label's
    /// probability can be found.
    ///
    /// # Returns
    ///
    /// If the label was found its index returned in the [`Option`]. Otherwise `None`
    /// is returned.
    pub fn class_index_for_label(&self, label: i32) -> Option<usize> {
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
    /// The inverse of [`DenseSVM::class_index_for_label`], this function returns the class label
    /// associated with a certain internal index. The index equals the [`Problem::probabilities`]
    /// index where a label's probability can be found.
    ///
    /// # Returns
    ///
    /// If the index was found it is returned in the [`Option`]. Otherwise `None`
    /// is returned.
    pub fn class_label_for_index(&self, index: usize) -> Option<i32> {
        if index >= self.classes.len() {
            None
        } else {
            Some(self.classes[index].label)
        }
    }

    /// Computes the kernel values for this problem
    pub(crate) fn compute_kernel_values(&self, problem: &mut Problem<SimdVector<f32s>>) {
        // Get current problem and decision values array
        let features = &problem.features;
        let kernel_values = &mut problem.kernel_values;

        // Compute kernel values per class
        for (i, class) in self.classes.iter().enumerate() {
            let kvalues = kernel_values.row_as_flat_mut(i);

            self.kernel.compute(&class.support_vectors, features.as_raw(), kvalues);
        }
    }

    // This is pretty much copy-paste of `multiclass_probability` from libSVM which we need
    // to be compatibly for predicting probability for multiclass SVMs. The method is in turn
    // based on Method 2 from the paper "Probability Estimates for Multi-class
    // Classification by Pairwise Coupling", Journal of Machine Learning Research 5 (2004) 975-1005,
    // by Ting-Fan Wu, Chih-Jen Lin and Ruby C. Weng.
    pub(crate) fn compute_multiclass_probabilities(
        &self,
        problem: &mut Problem<SimdVector<f32s>>,
    ) -> Result<(), Error> {
        compute_multiclass_probabilities_impl!(self, problem)
    }

    /// Based on kernel values, computes the decision values for this problem.
    pub(crate) fn compute_classification_values(&self, problem: &mut Problem<SimdVector<f32s>>) {
        compute_classification_values_impl!(self, problem)
    }

    /// Based on kernel values, computes the decision values for this problem.
    pub(crate) fn compute_regression_values(&self, problem: &mut Problem<SimdVector<f32s>>) {
        let class = &self.classes[0];
        let coef = class.coefficients.row(0);
        let kvalues = problem.kernel_values.row(0);

        let mut sum = coef.iter().zip(kvalues).map(|(a, b)| (*a * *b).sum()).sum::<f64>();

        sum -= self.rho[0];

        problem.result = Solution::Value(sum as f32);
    }

    /// Returns number of attributes, reflecting the libSVM model.
    pub fn attributes(&self) -> usize { self.num_attributes }

    /// Returns number of classes, reflecting the libSVM model.
    pub fn classes(&self) -> usize { self.classes.len() }
}


impl Predict<SimdVector<f32s>, SimdVector<f64s>> for DenseSVM {
    fn predict_probability(&self, problem: &mut Problem<SimdVector<f32s>>) -> Result<(), Error> {
        predict_probability_impl!(self, problem)
    }

    // Predict the value for one problem.
    fn predict_value(&self, problem: &mut Problem<SimdVector<f32s>>) -> Result<(), Error> {
        match self.svm_type {
            SVMType::CSvc | SVMType::NuSvc => {
                // Compute kernel, decision values and eventually the label
                self.compute_kernel_values(problem);
                self.compute_classification_values(problem);

                // Compute highest vote
                let highest_vote = find_max_index(&problem.vote);
                problem.result = Solution::Label(self.classes[highest_vote].label);

                Ok(())
            }
            SVMType::ESvr | SVMType::NuSvr => {
                self.compute_kernel_values(problem);
                self.compute_regression_values(problem);
                Ok(())
            }
        }
    }
}

impl<'a, 'b> TryFrom<&'a str> for DenseSVM {
    type Error = Error;

    fn try_from(input: &'a str) -> Result<DenseSVM, Error> {
        let raw_model = ModelFile::try_from(input)?;
        Self::try_from(&raw_model)
    }
}

impl<'a, 'b> TryFrom<&'a ModelFile<'b>> for DenseSVM {
    type Error = Error;

    fn try_from(raw_model: &'a ModelFile<'_>) -> Result<DenseSVM, Error> {
        let (mut svm, nr_sv) = prepare_svm!(raw_model, dyn KernelDense, SimdMatrix<f32s, RowOptimized>, DenseSVM);

        let vectors = &raw_model.vectors;

        // Things down here are a bit ugly as the file format is a bit ugly ...
        // Now read all vectors and decode stored information
        let mut start_offset = 0;

        // In the raw file, support vectors are grouped by class
        for (i, num_sv_per_class) in nr_sv.iter().enumerate() {
            let stop_offset = start_offset + *num_sv_per_class as usize;

            // Set support vector and coefficients
            for (i_vector, vector) in vectors[start_offset .. stop_offset].iter().enumerate() {
                let mut last_attribute = None;

                // Set support vectors
                for (i_attribute, attribute) in vector.features.iter().enumerate() {
                    if let Some(last) = last_attribute {
                        // In case we have seen an attribute already, this one must be strictly
                        // the successor attribute
                        if attribute.index != last + 1 {
                            return Result::Err(Error::AttributesUnordered {
                                index: attribute.index,
                                value: attribute.value,
                                last_index: last,
                            });
                        }
                    };

                    let mut support_vectors = svm.classes[i].support_vectors.flat_mut();
                    support_vectors[(i_vector, i_attribute)] = attribute.value;

                    last_attribute = Some(attribute.index);
                }

                // Set coefficients
                for (i_coefficient, coefficient) in vector.coefs.iter().enumerate() {
                    let mut coefficients = svm.classes[i].coefficients.flat_mut();
                    coefficients[(i_coefficient, i_vector)] = f64::from(*coefficient);
                }
            }

            // Update last offset.
            start_offset = stop_offset;
        }

        // Return what we have
        Result::Ok(svm)
    }
}

#[cfg(test)]
mod tests {
    use crate::*;
    use std::convert::TryFrom;

    #[test]
    fn class_operations() -> Result<(), Error> {
        let svm = DenseSVM::try_from(SAMPLE_MODEL)?;

        assert_eq!(None, svm.class_index_for_label(0));
        assert_eq!(Some(1), svm.class_index_for_label(42));

        Ok(())
    }

}
