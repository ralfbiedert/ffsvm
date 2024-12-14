//! [![crates.io-badge]][crates.io-url]
//! [![docs.rs-badge]][docs.rs-url]
//! ![license-badge]
//! [![rust-version-badge]][rust-version-url]
//! [![rust-build-badge]][rust-build-url]
//!
//! # In One Sentence
//!
//! You trained an SVM using [libSVM](https://github.com/cjlin1/libsvm), now you want the highest possible performance during (real-time) classification, like games or VR.
//!
//! # Highlights
//!
//! * loads almost all [libSVM](https://github.com/cjlin1/libsvm) types (C-SVC, ν-SVC, ε-SVR,  ν-SVR) and kernels (linear, poly, RBF and sigmoid)
//! * produces practically same classification results as libSVM
//! * optimized for [SIMD](https://github.com/rust-lang/rfcs/pull/2366) and can be mixed seamlessly with [Rayon](https://github.com/rayon-rs/rayon)
//! * written in 100% safe Rust
//! * allocation-free during classification for dense SVMs
//! * **2.5x - 14x faster than libSVM for dense SVMs**
//! * extremely low classification times for small models (e.g., 128 SV, 16 dense attributes, linear ~ 500ns)
//! * successfully used in **Unity and VR** projects (Windows & Android)
//!
//! # Usage
//!
//! Train with [libSVM](https://github.com/cjlin1/libsvm) (e.g., using the tool `svm-train`), then classify with `ffsvm-rust`.
//!
//! From Rust:
//!
//! ```rust
//! # use std::convert::TryFrom;
//! # use ffsvm::{DenseSVM, Predict, FeatureVector, SAMPLE_MODEL, Label};
//! # fn main() -> Result<(), ffsvm::Error> {
//! // Replace `SAMPLE_MODEL` with a `&str` to your model.
//! let svm = DenseSVM::try_from(SAMPLE_MODEL)?;
//!
//! let mut fv = FeatureVector::from(&svm);
//! let features = fv.features();
//!
//! features[0] = 0.55838;
//! features[1] = -0.157895;
//! features[2] = 0.581292;
//! features[3] = -0.221184;
//!
//! svm.predict_value(&mut fv)?;
//!
//! assert_eq!(fv.label(), Label::Class(42));
//! # Ok(())
//! # }
//! ```
//!
//! # Status
//! * **December 14, 2024**: **After 7+ years, finally ported to stable**.<sup>🎉</sup><sup>🎉</sup><sup>🎉</sup>
//! * **March 10, 2023**: Reactivated for latest Rust nightly.
//! * **June 7, 2019**: Gave up on 'no `unsafe`', but gained runtime SIMD selection.
//! * **March 10, 2019**: As soon as we can move away from nightly we'll go beta.
//! * **Aug 5, 2018**: Still in alpha, but finally on crates.io.
//! * **May 27, 2018**: We're in alpha. Successfully used internally on Windows, Mac, Android and Linux
//!   on various machines and devices. Once SIMD stabilizes and we can cross-compile to WASM
//!   we'll move to beta.
//! * **December 16, 2017**: We're in pre-alpha. It will probably not even work on your machine.
//!
//!
//! # Performance
//!
//! ![performance](https://raw.githubusercontent.com/ralfbiedert/ffsvm-rust/master/docs/performance_relative.v3.png)
//!
//! All performance numbers reported for the `DenseSVM`. We also have support for `SparseSVM`s, which are slower
//! for "mostly dense" models, and faster for "mostly sparse" models (and generally on the performance level of libSVM).
//!
//! [See here for details.](https://github.com/ralfbiedert/ffsvm-rust/blob/master/docs/performance.md)
//!
//!
//! ### Tips
//!
//! * Compile your project with `target-cpu=native` for a massive speed boost (e.g., check our `.cargo/config.toml` how
//!   you can easily do that for your project). Note, due to how Rust works, this is only used for application
//!   (or dynamic FFI libraries), not library crates wrapping us.
//! * For an x-fold performance increase, create a number of `Problem` structures, and process them with [Rayon's](https://docs.rs/rayon/1.0.3/rayon/) `par_iter`.
//!
//! # FAQ
//!
//! [See here for details.](https://github.com/ralfbiedert/ffsvm-rust/blob/master/docs/FAQ.md)
//!
//! [crates.io-badge]: https://img.shields.io/crates/v/ffsvm.svg
//! [crates.io-url]: https://crates.io/crates/ffsvm
//! [license-badge]: https://img.shields.io/badge/license-BSD2-blue.svg
//! [docs.rs-badge]: https://docs.rs/ffsvm/badge.svg
//! [docs.rs-url]: https://docs.rs/ffsvm/
//! [rust-version-badge]: https://img.shields.io/badge/rust-1.83%2B-blue.svg?maxAge=3600
//! [rust-version-url]: https://github.com/ralfbiedert/ffsvm
//! [rust-build-badge]: https://github.com/ralfbiedert/ffsvm/actions/workflows/rust.yml/badge.svg
//! [rust-build-url]: https://github.com/ralfbiedert/ffsvm/actions/workflows/rust.yml
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
        features::{DenseFeatures, FeatureVector, Label, SparseFeatures},
        kernel::{KernelDense, KernelSparse, Linear, Poly, Rbf, Sigmoid},
        predict::Predict,
        DenseSVM, SVMType, SparseSVM,
    },
};
