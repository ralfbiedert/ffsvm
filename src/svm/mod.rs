mod class;
mod csvm;
mod problem;

pub use self::{class::Class, problem::Problem};

use kernel::{Kernel, RbfKernel};
use vectors::Triangular;

pub type RbfCSVM = SVM<RbfKernel>;

#[derive(Debug)]
pub enum SVMError {
    /// All attributes must be in order 0, 1, 2, ..., n. If they are not, this
    /// error will be emitted.
    SvmAttributesUnordered {
        index: u32,
        value: f32,
        last_index: u32,
    },

    ModelDoesNotSupportProbabilities,

    MaxIterationsExceededPredictingProbabilities,
}

#[derive(Clone, Debug, Default)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct Probabilities {
    a: Triangular<f64>,

    b: Triangular<f64>,
}

/// Core support vector machine
#[derive(Clone, Debug, Default)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct SVM<T>
where
    T: Kernel,
{
    /// Total number of support vectors
    pub num_total_sv: usize,

    /// Number of attributes per support vector
    pub num_attributes: usize,

    pub rho: Triangular<f64>,

    pub probabilities: Option<Probabilities>,

    /// SVM specific data needed for classification
    pub kernel: T,

    /// All classes
    pub classes: Vec<Class>,
}

impl<T> SVM<T>
where
    T: Kernel,
{
    /// Finds the class index for a given label.
    pub fn class_index_for_label(&self, label: u32) -> Option<usize> {
        for (i, class) in self.classes.iter().enumerate() {
            if class.label != label {
                continue;
            }

            return Some(i);
        }

        None
    }
}

/// Predict a problem.
pub trait PredictProblem
where
    Self: Sync,
{
    /// Predict a single value for a problem.
    fn predict_value(&self, &mut Problem) -> Result<(), SVMError>;

    /// Predict a probability value for a problem.
    fn predict_probability(&self, &mut Problem) -> Result<(), SVMError>;
}
