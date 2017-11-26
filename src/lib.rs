// TODO:
// * One call classify multiple problems
// * Use SIMD
// * Use parallelism

#[macro_use] extern crate nom;
extern crate faster;


pub mod types;
pub mod parser;
pub mod classifier;

//