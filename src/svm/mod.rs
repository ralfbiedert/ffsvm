mod csvm;
mod problem;
mod class;

pub use self::problem::Problem;
pub use self::class::Class;

use rayon::prelude::*;
use std::marker::Sync;
use vectors::Triangular;
use kernel::{Kernel, RbfKernel};


pub type RbfCSVM = SVM<RbfKernel>;


/// Core support vector machine
#[derive(Debug)]
pub struct SVM<T> where
T : Kernel
{
    /// Total number of support vectors
    pub num_total_sv: usize,

    /// Number of attributes per support vector
    pub num_attributes: usize,

    pub rho: Triangular<f64>,

    /// SVM specific data needed for classification
    pub kernel: T,

    /// All classes
    pub classes: Vec<Class>,
}



/// Predict a problem. 
pub trait PredictProblem where Self : Sync
{
    /// Predict a single value for a problem.
    fn predict_value(&self, &mut Problem);

    
    /// Predicts all values for a set of problems.
    fn predict_values(&self, problems: &mut [Problem]) {

        // Compute all problems ...
        problems.par_iter_mut().for_each(|problem|
            self.predict_value(problem)
        );
    }
}



