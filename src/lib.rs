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
mod rbfcsvm;
mod parser;
mod data;
mod util;
mod randomization;
mod rbfkernel;

pub use manyvectors::ManyVectors;
pub use rbfcsvm::RbfCSVM;
pub use parser::{RawModel};
pub use data::{Class, Kernel, Problem, SVM};
pub use randomization::Randomize;
