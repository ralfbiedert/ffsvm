//! FFSVM stands for "Really Fast Support Vector Machine", a
//! [libSVM](https://www.csie.ntu.edu.tw/~cjlin/libsvm/) compatible classifier.
//! It allows you to load models trained by libSVM's `svm-train`, and use them from your Rust
//! code.

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
//! `svm-train`. If you don't, skip to the next section.
//!
//! ```rust
//! #![feature(try_from)]
//!
//! use ffsvm::*;
//! use std::convert::TryFrom;
//!
//! fn main() -> Result<(), SVMError> {
//!     // Replace `SAMPLE_MODEL` with a `&str` to your model.
//!     let svm = DenseSVM::try_from(SAMPLE_MODEL)?;
//!
//!     let mut problem = Problem::from(&svm);
//!     let mut features = problem.features();
//!
//!     features[0] = 0.55838;
//!     features[1] = -0.157895;
//!     features[2] = 0.581292;
//!     features[3] = -0.221184;
//!
//!     svm.predict_value(&mut problem)?;
//!
//!     assert_eq!(problem.result(), SVMResult::Label(42));
//!
//!     Ok(())
//! }
//! ```
//!
//! ### Creating a libSVM model
//!
//! Although FFSVM is 100% Rust code without any native dependencies, creating a model for use in
//! this library requires the `libSVM` tools for your current platform:
//!
//! * On **Windows** see the [official builds](https://github.com/cjlin1/libsvm/tree/master/windows)
//! * For **MacOS** use [Homebrew](https://brew.sh/) and run `brew install libsvm`,
//! * **Linux** users need to check with their distro
//!
//! Then make sure you have labeled training data in a libSVM compatible file format:
//!
//! ```ignore
//! > cat ./my.training-data
//! +1 1:0.708333 2:1 3:1 4:-0.320755 5:-0.105023 6:-1 7:1 8:-0.419847
//! -1 1:0.583333 2:-1 3:0.333333 4:-0.603774 5:1 6:-1 7:1 8:0.358779
//! +1 1:0.166667 2:1 3:-0.333333 4:-0.433962 5:-0.383562 6:-1 7:-1 8:0.0687023
//! -1 1:0.458333 2:1 3:1 4:-0.358491 5:-0.374429 6:-1 7:-1 8:-0.480916
//! ```
//!
//! Because FFSVM only supports dense SVMs you **must make sure** all attributes
//! for each sample are present and **all attributes are numbered in sequential, increasing**
//! order!
//!
//! Next, run `svm-train` on your data:
//!
//! ```ignore
//! svm-train ./my.training-data ./my.model
//! ```
//!
//! This will create the file `my.model` you can then include in the example above.
//!
//! For more advanced use cases and best classification accuracy, you should consider to run
//! grid search before you train your model. LibSVM comes with a tool `tools/grid.py` that you
//! can run:
//!
//! ```ignore
//! > python3 grid.py ./my.training-data
//! [local] 5 -7 0.0 (best c=32.0, g=0.0078125, rate=0.0)
//! [local] -1 -7 0.0 (best c=0.5, g=0.0078125, rate=0.0)
//! [local] 5 -1 0.0 (best c=0.5, g=0.0078125, rate=0.0)
//! [local] -1 -1 0.0 (best c=0.5, g=0.0078125, rate=0.0)
//! ...
//! ```
//!
//! The best parameters (in this case `c=0.5`, `g=0.0078125`) can then be used on `svm-train`. The
//! optional paramter `-b 1` allows the model to also predict probabilty estimates for its
//! classification.
//!
//! ```ignore
//! > svm-train -c 0.5 -g 0.0078125 -b 1 ./my.training-data ./my.model
//! ```
//!
//! For more information how to use libSVM to generate the best models, see the
//! [Practical Guide to SVM Classification](https://www.csie.ntu.edu.tw/%7Ecjlin/papers/guide/guide.pdf)
//! and the [libSVM FAQ](https://www.csie.ntu.edu.tw/%7Ecjlin/libsvm/faq.html).
//!

// Opt in to unstable features expected for Rust 2018
#![feature(try_from, stdsimd, rust_2018_preview, try_trait)]
#![warn(rust_2018_idioms)]

mod errors;
mod parser;
mod sparse;
mod svm;
mod util;
mod vectors;

pub static SAMPLE_MODEL: &str = include_str!("sample.model");

pub use crate::{
    errors::SVMError,
    parser::ModelFile,
    svm::{
        kernel::{KernelDense, Linear, Poly, Rbf, Sigmoid},
        predict::Predict,
        problem::{Problem, SVMResult},
        DenseSVM, SVMType, SparseSVM,
    },
};
