use crate::{errors::Error, svm::features::FeatureVector};

/// Implemented by [`DenseSVM`](crate::DenseSVM) and [`SparseSVM`](crate::SparseSVM) to predict a [`FeatureVector`].
///
/// # Predicting a label
///
/// To predict a label, first make sure the [`FeatureVector`](crate::FeatureVector) has all features set. Then calling
/// ```
/// use ffsvm::{DenseFeatures, DenseSVM, Predict};
///
/// fn set_features(svm: &DenseSVM, problem: &mut DenseFeatures) {
///     // Predicts the value.
///     svm.predict_value(problem);
/// }
/// ```
/// will update the [`FeatureVector::label`] to correspond to the class label with the highest likelihood.
///
/// # Predicting a label and obtaining probability estimates.
///
/// If the libSVM model was trained with probability estimates FFSVM can not only predict the
/// label, but it can also give information about the likelihood distribution of all classes.
/// This can be helpful if you want to consider alternatives.
///
/// Probabilities are estimated like this:
///
/// ```
/// use ffsvm::{DenseFeatures, DenseSVM, Predict};
///
/// fn set_features(svm: &DenseSVM, features: &mut DenseFeatures) {
///     // Predicts the value.
///     svm.predict_probability(features);
/// }
/// ```
///
/// Predicting probabilities automatically predicts the best label. In addition, [`FeatureVector::probabilities`]
/// will be updated accordingly. The class labels for each probablity entry can be obtained
/// by the SVM's `class_label_for_index` and `class_index_for_label` methods.
pub trait Predict<T>
where
    Self: Sync,
{
    /// Predict a single value for a [`FeatureVector`].
    ///
    /// The problem needs to have all features set. Once this method returns,
    /// the [`FeatureVector::label`] will be set.
    ///
    /// # Errors
    ///
    /// Can fail if the feature vector didn't match the original SVM configuration.
    fn predict_value(&self, problem: &mut FeatureVector<T>) -> Result<(), Error>;

    /// Predict a probability value for a problem.
    ///
    /// The problem needs to have all features set. Once this method returns,
    /// both [`FeatureVector::label`] will be set, and all [`FeatureVector::probabilities`] will
    /// be available accordingly.
    ///
    /// # Errors
    ///
    /// Can fail if the model was not trained with probability estimates support.
    fn predict_probability(&self, problem: &mut FeatureVector<T>) -> Result<(), Error>;
}
