//! FFSVM stands for "Really Fast Support Vector Machine", a
//! [libSVM](https://www.csie.ntu.edu.tw/~cjlin/libsvm/) compatible classifier for dense RBF C-SVMs.
//! It allows you to load models trained by libSVM's `svm-train`, and use them from your Rust
//! code.

//! # Background
//! [Support Vector Machines](https://en.wikipedia.org/wiki/Support_Vector_Machine) (SVMs) are a
//! class of relatively simple and fast machine learning algorithms. They have
//! * few parameters (making them easy to tune),
//! * good generalization properties (making them good learners with limited data) and
//! * overall good classification accuracy.
//!
//! An RBF C-SVM is a special type of SVM, and probably the most frequently used one for a wide
//! range of problems.
//!
//! A *dense* SVM does not have any 'empty' features. In other words, for each
//! problem, you know all of that problems's value during training and classification. An example
//! could be classifying images, where each pixel value is known.
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
//! FFSVM is:
//! * **2.5x - 14x faster** than libSVM [according to our benchmarks](https://github.com/ralfbiedert/ffsvm-rust/blob/master/docs/performance.adoc).
//! * **allocation free** once the model is loaded
//! * highly **cache and SIMD** optimized
//! * trivially composable with [Rayon](https://github.com/rayon-rs/rayon) for even more performance
//! * ideally suited for real-time applications such as **games and VR**
//! * lightweight, and only comes with a few dependencies ([faster](https://github.com/AdamNiederer/faster) for SIMD, [nom](https://github.com/Geal/nom) for parsing libSVM models).
//!
//! For small to medium-sized problems FFSVM's execution speed is best measured in **nano-
//! or microseconds** on modern architectures (e.g., AVX2), not milliseconds.
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
//! ```rust,ignore
//! // Get this model from somewhere, e.g., a file.
//! let model_str: &str = include_str!("my.model");
//!
//! // Parse the model, and create an actual SVM instance
//! let model = ModelFile::try_from(model_str)!;
//! let svm = RbfSVM::try_from(&model)!;
//!
//! // Create a 'problem'. It will hold your features and, once classified, the label. Problems
//! // are meant to be reused between calls (e.g., each frame in games).
//! let mut problem = Problem::from(&svm);
//!
//! // Next we set all features, based on your real-world problem.
//! problem.features = vec![
//!     -0.55838, -0.157895, 0.581292, -0.221184, 0.135713, -0.874396, -0.563197, -1.0, -1.0,
//! ];
//!
//! // Here we ask the SVM to classify the problem. This will update the `.label` field.
//! svm.predict_value(&mut problem)!
//!
//! // Assume the `.label` is what we expect.
//! assert_eq!(problem.label, 123);
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
mod random;
mod svm;
mod util;
mod vectors;

pub use crate::{
    errors::SVMError,
    random::{Random, RandomSVM, Randomize},
    svm::{
        kernel::{Kernel, Linear, Poly, Rbf, Sigmoid},
        predict::Predict,
        problem::{Problem, SVMResult},
        SVMType, SVM,
    },
};
