// TODO impls:
// when process an array, take iterable trait, not slice type
// Problem: Introduce option for label, features and probabilites


#![feature(toowned_clone_into)]
#![feature(test)]
#![feature(repr_simd)]
#![feature(try_from)]
#![feature(libc)]

//#![cfg_attr(feature="clippy", feature(plugin))]
//#![cfg_attr(feature="clippy", plugin(clippy))]

extern crate faster;
#[macro_use] extern crate nom;
extern crate rand;
extern crate rayon;
extern crate test;
extern crate libc;

mod vectors;
mod kernel;
mod parser;
mod svm;
mod random;
pub mod ffi;
pub mod util;
pub mod profiling;

pub use kernel::RbfKernel;
pub use svm::{SVM, Class, Problem, RbfCSVM, PredictProblem};
pub use parser::{ModelFile};
pub use vectors::SimdOptimized;
pub use random::{Randomize, Random};
