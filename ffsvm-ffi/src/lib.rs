#![feature(try_from)]
#![feature(libc)]

extern crate libc;
extern crate ffsvm;
extern crate rayon;

use ffsvm::{Problem, RbfCSVM, PredictProblem};
use rayon::prelude::*;

pub mod ffi;

/// Predicts all values for a set of problems.
pub fn predict_values(svm: &RbfCSVM, problems: &mut [Problem]) {
    // Compute all problems ...
    problems
        .par_iter_mut()
        .for_each(|problem| { svm.predict_value(problem); });
}

/// Predicts all probabilities for a set of problems.
pub fn predict_probabilities(svm: &RbfCSVM, problems: &mut [Problem]) {
    // Compute all problems ...
    problems
        .par_iter_mut()
        .for_each(|problem| { svm.predict_probability(problem); });
}