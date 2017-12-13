#![feature(toowned_clone_into)]
#![feature(test)]
#![feature(repr_simd)]

extern crate faster;
extern crate itertools;
#[macro_use] extern crate nom;
extern crate rand;
extern crate rayon;
extern crate test;

mod manyvectors;
mod triangularmatrix;
mod rbfcsvm;
mod parser;
mod data;
mod randomization;
mod rbfkernel;
pub mod util;

pub use manyvectors::ManyVectors;
pub use rbfcsvm::RbfCSVM;
pub use parser::{RawModel};
pub use data::{Class, Kernel, Problem, SVM};
pub use randomization::Randomize;
//pub use triangularmatrix::TriangularMatrix;
