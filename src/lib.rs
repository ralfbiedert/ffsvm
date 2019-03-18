//! FFSVM stands for "Really Fast Support Vector Machine", a
//! [libSVM](https://www.csie.ntu.edu.tw/~cjlin/libsvm/) compatible classifier.
//! It allows you to load models trained by libSVM's `svm-train`, and use them from your Rust
//! code.
//!
//! # Background
//! [Support Vector Machines](https://en.wikipedia.org/wiki/Support_Vector_Machine) (SVMs) are a
//! class of relatively simple and fast machine learning algorithms. They have
//! * few parameters (making them easy to tune),
//! * good generalization properties (making them good learners with limited data) and
//! * overall good classification accuracy.
//!
//! [LibSVM](https://www.csie.ntu.edu.tw/~cjlin/libsvm/) is a relatively portable, general purpose
//! SVM implementation written in C++ that includes tools for training, as well as tools and code
//! for classification.
//!
//! FFSVM is a library that can load such models trained by [libSVM](https://www.csie.ntu.edu.tw/~cjlin/libsvm/)'s
//! `svm-train` and offers a number of benefits:
//!
//! # Features
//!
//! FFSVM
//! * loads almost all [libSVM](https://github.com/cjlin1/libsvm) types (C-SVC, ν-SVC, ε-SVR,  ν-SVR) and kernels (linear, poly, RBF and sigmoid)
//! * produces practically same classification results as libSVM
//! * optimized for [SIMD](https://github.com/rust-lang/rfcs/pull/2366) and can be mixed seamlessly with [Rayon](https://github.com/rayon-rs/rayon)
//! * written in 100% Rust, but can be loaded from any language (via FFI)
//! * allocation-free during classification for dense SVMs
//! * **2.5x - 14x faster than libSVM for dense SVMs**
//! * extremely low classification times for small models (e.g., 128 SV, 16 dense attributes, linear ~ 500ns)
//! * successfully used in **Unity and VR** projects (Windows & Android)
//! * free of `unsafe` code ;)
//!
//! FFSVM is not, however, a full libSVM replacement. Instead, it assumes you use `svm-train`
//! *at home* (see [Usage](#usage) below), and ship a working model with your library or application.
//!
//! # Usage
//!
//! ### If you have a libSVM model
//!
//! In this example we assume you already have a libSVM that was trained with
//! `svm-train`. If you haven't created a model yet, [check out the FAQ on how to get started](https://github.com/ralfbiedert/ffsvm-rust/blob/master/docs/FAQ.md).
//!
//! ```rust
//! use ffsvm::*;
//! use std::convert::TryFrom;
//!
//! fn main() -> Result<(), Error> {
//!     // Replace `SAMPLE_MODEL` with a `&str` to your model.
//!     let svm = DenseSVM::try_from(SAMPLE_MODEL)?;
//!
//!     let mut problem = Problem::from(&svm);
//!     let features = problem.features();
//!
//!     features[0] = 0.55838;
//!     features[1] = -0.157895;
//!     features[2] = 0.581292;
//!     features[3] = -0.221184;
//!
//!     svm.predict_value(&mut problem)?;
//!
//!     assert_eq!(problem.solution(), Solution::Label(42));
//!
//!     Ok(())
//! }
//! ```
//!
//! # Performance Tips
//!
//! * For a ~50% performance boost consider compiling your application with more aggressive CPU flags (e.g., `export RUSTFLAGS="-C target-feature=+avx2"` in case you run on a modern x86 CPU).
//! * For a further x-fold performance increase, create a number of [`Problem`] structures, and process them with [Rayon's](https://docs.rs/rayon/1.0.3/rayon/) `par_iter`.

#![feature(try_trait)]
#![warn(clippy::all)] // Enable ALL the warnings ...
#![warn(clippy::nursery)]
#![warn(clippy::pedantic)]
#![warn(clippy::cargo)]
#![allow(clippy::cast_possible_truncation)] // All our casts are in a range where this doesn't matter.
#![allow(clippy::cast_precision_loss)]
#![allow(clippy::cast_possible_wrap)]
#![allow(clippy::module_name_repetitions)] // We do that way too often
#![allow(clippy::doc_markdown)] // Mainly for `libSVM` in the docs.

mod errors;
mod parser;
mod sparse;
mod svm;
mod util;
mod vectors;

#[doc(hidden)]
pub static SAMPLE_MODEL: &str = include_str!("sample.model");

pub use crate::{
    errors::Error,
    parser::{Attribute, Header, ModelFile, SupportVector},
    svm::{
        kernel::{KernelDense, KernelSparse, Linear, Poly, Rbf, Sigmoid},
        predict::Predict,
        problem::{DenseProblem, Problem, Solution, SparseProblem},
        DenseSVM, SVMType, SparseSVM,
    },
};
