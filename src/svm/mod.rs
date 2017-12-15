mod crbf;
mod problem;
mod class;

pub use self::crbf::RbfCSVM;
pub use self::problem::Problem;
pub use self::class::Class;

use rayon::prelude::*;
use std::marker::Sync;
use kernel::Kernel;


/// Core support vector machine
#[derive(Debug)]
pub struct SVM<T> where
T : Kernel
{
    /// Total number of support vectors
    pub num_total_sv: usize,

    /// Number of attributes per support vector
    pub num_attributes: usize,

    pub rho: Vec<f64>,

    /// SVM specific data needed for classification
    pub kernel: T,

    /// All classes
    pub classes: Vec<Class>,
}


// Maybe not, that would be a very special trait ... Should probably be part of 
// the struct impl instead. 
// 
// Maybe the whole random thing should be one though.
// HOWEVER, That would be problematic, since it would need special arguments (e.g., size),
// which would be hard to reflect in a generic trait.
//trait FromLibSvmModel {
//    fn form(xxx) -> Self;    
//}
//


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



