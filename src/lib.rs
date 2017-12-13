#![feature(toowned_clone_into)]
#![feature(test)]
#![feature(repr_simd)]

extern crate faster;
extern crate itertools;
#[macro_use] extern crate nom;
extern crate rand;
extern crate rayon;
extern crate test;

mod vectors;
mod kernel;
mod parser;
mod svm;
mod random;
pub mod util;

pub use vectors::flat::ManyVectors;
pub use kernel::rbf::RbfKernel;
pub use svm::problem::Problem;
pub use svm::{SVM, Class};
pub use svm::crbf::RbfCSVM;
pub use parser::model::RawModel;
pub use random::Randomize;
