// TODO impls:
// Try?From<&str> for parser
// Index for SIMDXX ...
// FromModel for Kernel
// crbf to csvm<K>


#![feature(toowned_clone_into)]
#![feature(test)]
#![feature(repr_simd)]
#![feature(try_from)]

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

pub use kernel::RbfKernel;
pub use svm::{SVM, Class, Problem, RbfCSVM, PredictProblem};
pub use parser::{ModelFile};
pub use vectors::SimdOptimized;
pub use random::Randomize;
