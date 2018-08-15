use crate::{errors::Error, svm::problem::Problem};

/// Implemented by [DenseSVM] and [SparseSVM] to predict a [Problem].
///
/// # Predicting a label
///
/// To predict a label, first make sure the [Problem] has all features set. Then calling
/// ```
/// use ffsvm::*;
///
/// fn set_features(svm: &DenseSVM, problem: &mut DenseProblem) {
///     // Predicts the value.
///     svm.predict_value(problem);
/// }
/// ```
/// will update the [Problem::solution] to correspond to the class label with the highest likelihood.
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
/// use ffsvm::*;
///
/// fn set_features(svm: &DenseSVM, problem: &mut DenseProblem) {
///     // Predicts the value.
///     svm.predict_probability(problem);
/// }
/// ```
///
/// Predicting probabilities automatically predicts the best label. In addition [Problem::probabilities]
/// will be updated accordingly. The class labels for each probablity entry can be obtained
/// by the [SVMCore::class_label_for_index] and [SVMCore::class_index_for_label] methods.
///
pub trait Predict<V32, V64>
where
    Self: Sync,
{
    /// Predict a single value for a [Problem].
    ///
    /// The problem needs to have all features set. Once this method returns,
    /// the [Problem::solution] will be set.
    fn predict_value(&self, problem: &mut Problem<V32>) -> Result<(), Error>;

    /// Predict a probability value for a problem.
    ///
    /// The problem needs to have all features set. Once this method returns,
    /// both [Problem::solution] will be set, and all [Problem::probabilities] will
    /// be available accordingly.
    fn predict_probability(&self, problem: &mut Problem<V32>) -> Result<(), Error>;
}
