#![feature(try_from)]

extern crate faster;
#[macro_use]
extern crate nom;
extern crate rand;
#[cfg(feature = "serde")]
#[macro_use]
extern crate serde;

mod kernel;
mod parser;
mod random;
mod svm;
pub mod util;
mod vectors;

pub use kernel::RbfKernel;
pub use parser::ModelFile;
pub use random::{Random, Randomize};
pub use svm::{Class, PredictProblem, Problem, RbfCSVM, SVM};
pub use vectors::SimdOptimized;
