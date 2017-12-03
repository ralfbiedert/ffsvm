// TODO:
// * One call classify multiple problems
// * Use SIMD
// * Use parallelism
#![feature(toowned_clone_into)]
#![feature(test)]
#![feature(conservative_impl_trait)]    // to "return impl FnMut" 
#![feature(repr_simd)]

#[macro_use] extern crate nom;
extern crate faster;
extern crate rand;
extern crate test;
extern crate itertools;
extern crate rayon;

pub mod types;
pub mod matrix;
pub mod csvm;
pub mod parser;
mod util;


//