use crate::{errors::SVMError, svm::problem::Problem};

/// Implemented by [SVM]s to predict a [Problem].
///
/// # Predicting a label
///
/// To predict a label, first make sure the [Problem] has all features set. Then calling
/// ```ignore
/// svm.predict_value(&mut problem)!
/// ```
/// will update the `.label` field to correspond to the class label with the highest likelihood.
///
/// # Predicting a label and obtaining probability estimates.
///
/// If the libSVM model was trained with probability estimates FFSVM can not only predict the
/// label, but it can also give information about the likelihood distribution of all classes.
/// This can be helpful if you want to consider alternatives.
///
/// Probabilities are estimated like this:
///
/// ```ignore
/// svm.predict_probability(&mut problem)!
/// ```
///
/// Predicting probabilities automatically predicts the best label. In addition `.probabilities`
/// will be updated accordingly. The class labels for each `.probabilities` entry can be obtained
/// by [SVM]'s `class_label_for_index` and `class_index_for_label` methods.
///
pub trait Predict<V32, V64>
where
    Self: Sync,
{
    /// Predict a single value for a [Problem].
    ///
    /// The problem needs to have all `.features` set. Once this method returns,
    /// the [Problem]'s field `.label` will be set.
    fn predict_value(&self, _: &mut Problem<V32, V64>) -> Result<(), SVMError>;

    /// Predict a probability value for a problem.
    ///
    /// The problem needs to have all `.features` set. Once this method returns,
    /// both the [Problem]'s field `.label` will be set, and all `.probabilities` will
    /// be set accordingly.
    fn predict_probability(&self, _: &mut Problem<V32, V64>) -> Result<(), SVMError>;
}
