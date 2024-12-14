use crate::{
    errors::Error,
    parser::ModelFile,
    svm::{
        class::Class,
        features::{FeatureVector, Label},
        kernel::{KernelDense, Linear, Poly, Rbf, Sigmoid},
        predict::Predict,
        Probabilities, SVMType,
    },
    util::{find_max_index, set_all, sigmoid_predict},
    vectors::Triangular,
};
use simd_aligned::{arch::f32x8, traits::Simd, MatSimd, Rows, VecSimd};
use std::convert::TryFrom;

/// An SVM using [SIMD](https://en.wikipedia.org/wiki/SIMD) intrinsics optimized for speed.
///
///
/// # Creating an SVM
///
/// This SVM can be created by passing a [`ModelFile`](crate::ModelFile) or [`&str`] into [`ModelFile::try_from`]:
///
/// ```
/// use ffsvm::DenseSVM;
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
    pub(crate) classes: Vec<Class<MatSimd<f32x8, Rows>>>,
}

impl DenseSVM {
    /// Finds the class index for a given label.
    ///
    /// # Description
    ///
    /// This method takes a `label` as defined in the libSVM training model
    /// and returns the internal `index` where this label resides. The index
    /// equals [`FeatureVector::probabilities`] index where that label's
    /// probability can be found.
    ///
    /// # Returns
    ///
    /// If the label was found its index returned in the [`Option`], otherwise `None`
    /// is returned.
    #[must_use]
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
    /// associated with a certain internal index. The index equals the [`FeatureVector::probabilities`]
    /// index where a label's probability can be found.
    ///
    /// # Returns
    ///
    /// If the index was found it is returned in the [`Option`], otherwise `None`
    /// is returned.
    #[must_use]
    pub fn class_label_for_index(&self, index: usize) -> Option<i32> {
        if index >= self.classes.len() {
            None
        } else {
            Some(self.classes[index].label)
        }
    }

    /// Computes the kernel values for this problem
    pub(crate) fn compute_kernel_values(&self, problem: &mut FeatureVector<VecSimd<f32x8>>) {
        // Get current problem and decision values array
        let features = &problem.features;
        let kernel_values = &mut problem.kernel_values;

        // Compute kernel values per class
        for (i, class) in self.classes.iter().enumerate() {
            let kvalues = kernel_values.row_as_flat_mut(i);

            self.kernel.compute(&class.support_vectors, features, kvalues);
        }
    }

    // This is pretty much copy-paste of `multiclass_probability` from libSVM which we need
    // to be compatibly for predicting probability for multiclass SVMs. The method is in turn
    // based on Method 2 from the paper "Probability Estimates for Multi-class
    // Classification by Pairwise Coupling", Journal of Machine Learning Research 5 (2004) 975-1005,
    // by Ting-Fan Wu, Chih-Jen Lin and Ruby C. Weng.
    pub(crate) fn compute_multiclass_probabilities(&self, problem: &mut FeatureVector<VecSimd<f32x8>>) -> Result<(), Error> {
        compute_multiclass_probabilities_impl!(self, problem)
    }

    /// Based on kernel values, computes the decision values for this problem.
    pub(crate) fn compute_classification_values(&self, problem: &mut FeatureVector<VecSimd<f32x8>>) {
        compute_classification_values_impl!(self, problem)
    }

    /// Based on kernel values, computes the decision values for this problem.
    pub(crate) fn compute_regression_values(&self, problem: &mut FeatureVector<VecSimd<f32x8>>) {
        let class = &self.classes[0];
        let coef = class.coefficients.row(0);
        let kvalues = problem.kernel_values.row(0);

        let mut sum = coef.iter().zip(kvalues).map(|(a, b)| (*a * *b).sum()).sum::<f64>();

        sum -= self.rho[0];

        problem.result = Label::Value(sum as f32);
    }

    /// Returns number of attributes, reflecting the libSVM model.
    #[must_use]
    pub const fn attributes(&self) -> usize {
        self.num_attributes
    }

    /// Returns number of classes, reflecting the libSVM model.
    #[must_use]
    pub fn classes(&self) -> usize {
        self.classes.len()
    }
}

impl Predict<VecSimd<f32x8>> for DenseSVM {
    // Predict the value for one problem.
    fn predict_value(&self, fv: &mut FeatureVector<VecSimd<f32x8>>) -> Result<(), Error> {
        match self.svm_type {
            SVMType::CSvc | SVMType::NuSvc => {
                // Compute kernel, decision values and eventually the label
                self.compute_kernel_values(fv);
                self.compute_classification_values(fv);

                // Compute the highest vote
                let highest_vote = find_max_index(&fv.vote);
                fv.result = Label::Class(self.classes[highest_vote].label);

                Ok(())
            }
            SVMType::ESvr | SVMType::NuSvr => {
                self.compute_kernel_values(fv);
                self.compute_regression_values(fv);
                Ok(())
            }
        }
    }

    fn predict_probability(&self, problem: &mut FeatureVector<VecSimd<f32x8>>) -> Result<(), Error> {
        predict_probability_impl!(self, problem)
    }
}

impl<'a> TryFrom<&'a str> for DenseSVM {
    type Error = Error;

    fn try_from(input: &'a str) -> Result<Self, Error> {
        let raw_model = ModelFile::try_from(input)?;
        Self::try_from(&raw_model)
    }
}

impl<'a> TryFrom<&'a ModelFile<'_>> for DenseSVM {
    type Error = Error;

    fn try_from(raw_model: &'a ModelFile<'_>) -> Result<Self, Error> {
        let (mut svm, nr_sv) = prepare_svm!(raw_model, dyn KernelDense, MatSimd<f32x8, Rows>, Self);

        let vectors = &raw_model.vectors();

        // Things down here are a bit ugly as the file format is a bit ugly ...
        // Now read all vectors and decode stored information
        let mut start_offset = 0;

        // In the raw file, support vectors are grouped by class
        for (i, num_sv_per_class) in nr_sv.iter().enumerate() {
            let stop_offset = start_offset + *num_sv_per_class as usize;

            // Set support vector and coefficients
            for (i_vector, vector) in vectors[start_offset..stop_offset].iter().enumerate() {
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
        Ok(svm)
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
