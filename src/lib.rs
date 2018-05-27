// TODO impls:
// when process an array, take iterable trait, not slice type
// Problem: Introduce option for label, features and probabilites

#![feature(toowned_clone_into)]
#![feature(test)]
#![feature(repr_simd)]
#![feature(try_from)]
#![feature(libc)]

extern crate faster;
#[macro_use]
extern crate nom;
extern crate libc;
extern crate rand;
extern crate test;
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
