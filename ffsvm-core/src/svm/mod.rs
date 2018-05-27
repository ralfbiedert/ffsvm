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
    pub (crate) num_total_sv: usize,

    /// Number of attributes per support vector
    pub (crate) num_attributes: usize,

    pub (crate) rho: Triangular<f64>,

    pub (crate) probabilities: Option<Probabilities>,

    /// SVM specific data needed for classification
    pub (crate) kernel: T,

    /// All classes
    pub (crate) classes: Vec<Class>,
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

    /// TODO
    pub fn class_label_for_index(&self, index: usize) -> Option<u32> {
        
        return if index >= self.classes.len() {
            None
        } else {
            Some(self.classes[index].label)
        }
    }
    
    /// Returns number of attributes
    pub fn attributes(&self) -> usize {
        self.num_attributes        
    }

    /// Returns number of classes
    pub fn classes(&self) -> usize {
        self.classes.len()
    }
    
   
}

/// Predict a problem.
#[doc(hidden)]
pub trait PredictProblem
where
    Self: Sync,
{
    /// Predict a single value for a problem.
    fn predict_value(&self, &mut Problem) -> Result<(), SVMError>;

    /// Predict a probability value for a problem.
    fn predict_probability(&self, &mut Problem) -> Result<(), SVMError>;
}
