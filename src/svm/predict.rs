use crate::{
    errors::SVMError,
    svm::{csvm::CSVM, problem::Problem},
    util::{find_max_index, sigmoid_predict},
};

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
pub trait Predict
where
    Self: Sync,
{
    /// Predict a single value for a [Problem].
    ///
    /// The problem needs to have all `.features` set. Once this method returns,
    /// the [Problem]'s field `.label` will be set.
    fn predict_value(&self, _: &mut Problem) -> Result<(), SVMError>;

    /// Predict a probability value for a problem.
    ///
    /// The problem needs to have all `.features` set. Once this method returns,
    /// both the [Problem]'s field `.label` will be set, and all `.probabilities` will
    /// be set accordingly.
    fn predict_probability(&self, _: &mut Problem) -> Result<(), SVMError>;
}

impl Predict for CSVM {
    fn predict_probability(&self, problem: &mut Problem) -> Result<(), SVMError> {
        const MIN_PROB: f64 = 1e-7;

        // Ensure we have probabilities set. If not, somebody used us the wrong way
        if self.probabilities.is_none() {
            return Err(SVMError::NoProbabilities);
        }

        let num_classes = self.classes.len();
        let probabilities = self.probabilities.as_ref().unwrap();

        // First we need to predict the problem for our decision values
        self.predict_value(problem)?;

        let mut pairwise = problem.pairwise.flat_mut();

        // Now compute probability values
        for i in 0 .. num_classes {
            for j in i + 1 .. num_classes {
                let decision_value = problem.decision_values[(i, j)];
                let a = probabilities.a[(i, j)];
                let b = probabilities.b[(i, j)];

                let sigmoid =
                    sigmoid_predict(decision_value, a, b).max(MIN_PROB).min(1f64 - MIN_PROB);

                pairwise[(i, j)] = sigmoid;
                pairwise[(j, i)] = 1f64 - sigmoid;
            }
        }

        if num_classes == 2 {
            problem.probabilities[0] = pairwise[(0, 1)];
            problem.probabilities[1] = pairwise[(1, 0)];
        } else {
            self.compute_multiclass_probabilities(problem)?;
        }

        let max_index = find_max_index(problem.probabilities.as_slice());
        problem.label = self.classes[max_index].label;

        Ok(())
    }

    // Predict the value for one problem.
    fn predict_value(&self, problem: &mut Problem) -> Result<(), SVMError> {
        // Compute kernel, decision values and eventually the label
        self.compute_kernel_values(problem);
        self.compute_decision_values(problem);

        // Compute highest vote
        let highest_vote = find_max_index(&problem.vote);
        problem.label = self.classes[highest_vote].label;

        Ok(())
    }
}
