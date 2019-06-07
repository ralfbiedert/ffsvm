
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


### How can I train a model?

Although FFSVM is 100% Rust code without any native dependencies, creating a model for use in
this library requires the `libSVM` tools for your current platform:

* On **Windows** see the [official builds](https://github.com/cjlin1/libsvm/tree/master/windows)
* For **MacOS** use [Homebrew](https://brew.sh/) and run `brew install libsvm`,
* **Linux** users need to check with their distro

Then make sure you have labeled training data in a libSVM compatible file format:

```ignore
> cat ./my.training-data
+1 0:0.708333 1:1 2:1 3:-0.320755 4:-0.105023 5:-1 6:1 7:-0.419847
-1 0:0.583333 1:-1 2:0.333333 3:-0.603774 4:1 5:-1 6:1 7:0.358779
+1 0:0.166667 1:1 2:-0.333333 3:-0.433962 4:-0.383562 5:-1 6:-1 7:0.0687023
-1 0:0.458333 1:1 2:1 3:-0.358491 4:-0.374429 5:-1 6:-1 7:-0.480916

```

* If you want to use a `DenseSVM` you **must make sure** all attributes
for each sample are present, and **all attributes are numbered in sequential, increasing order starting with `0`**! For `SparseSVM`s these restrictions don't apply.
* In any case, make sure your **data is scaled**. That means each attribute is **in the range \[0; 1\], or \[-1; 1\]** respectively. If you do not scale your data, you will get poor accuracy and lots of "obviously wrong" classification results. Whatever scaling you apply, don't forget you have to apply the same scaling when you then classify with ffsvm.


Next, run `svm-train` on your data:

```ignore
svm-train ./my.training-data ./my.model
```

This will create the file `my.model` you can then include in the example above.
For more advanced use cases and best classification accuracy, you should consider to run
grid search before you train your model. LibSVM comes with a tool `tools/grid.py` that you
can run:

```ignore
> python3 grid.py ./my.training-data
[local] 5 -7 0.0 (best c=32.0, g=0.0078125, rate=0.0)
[local] -1 -7 0.0 (best c=0.5, g=0.0078125, rate=0.0)
[local] 5 -1 0.0 (best c=0.5, g=0.0078125, rate=0.0)
[local] -1 -1 0.0 (best c=0.5, g=0.0078125, rate=0.0)
...
```

The best parameters (in this case `c=0.5`, `g=0.0078125`) can then be used on `svm-train`. The
optional paramter `-b 1` allows the model to also predict probabilty estimates for its
classification.

```ignore
> svm-train -c 0.5 -g 0.0078125 -b 1 ./my.training-data ./my.model
```

For more information how to use libSVM to generate the best models, see the
[Practical Guide to SVM Classification](https://www.csie.ntu.edu.tw/%7Ecjlin/papers/guide/guide.pdf)
and the [libSVM FAQ](https://www.csie.ntu.edu.tw/%7Ecjlin/libsvm/faq.html).


### How can I use a trained `libSVM` model?

Since version 0.6 we should be able to load practically all `libSVM` models. Two caveats:

* For "regular speed" classification with any model use the provided `SparseSVM`.
* For "high speed" classification you can use `DenseSVM`. However, then all attributes must start with index `0`, have the same length and there must be no "holes".

