
# FAQ

## General

### Why `ffsvm`? What is the problem with `libSVM`?

First, in many cases there is nothing wrong with [libSVM](https://github.com/cjlin1/libSVM). If extreme classification performance is not an issue, it is probably the more flexible choice.

However, when using `libSVM` in real-time applications (e.g., VR games), a number of problems become noticeable:

* it does lots of small allocations per classification call
* non-optimal cache locality
* lots of cycles per vector processed

`ffsvm` tries to address that by:

* being zero-allocation during classification
* packing all data SIMD-friendly, and using SIMD intrinsics where it makes sense
* safe and parallelization friendly API
* being designed and measured, from day 1, for speed


With this in mind, `libSVM` still has nice, portable tools for training and grid search. The ultimate plan for `ffsvm` is not to replace these, but to use their output.


## Usage

### How can I use a trained `libSVM` model?

Since version 0.6 we should be able to load practically all `libSVM` models. Two caveats:

* For "regular speed" classification with any model use the provided `SparseSVM`.
* For "high speed" classification you can use `DenseSVM`. However, then all attributes must start with index `0`, have the same length and there must be no "holes".


## Development

### How do I enable AVX2 support?

Set the `RUSTFLAGS` environment variable to `-C target-feature=+avx2`.
