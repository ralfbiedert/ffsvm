
# Performance History

TODO: Paste Excel numbers here



# Questions & TODO                                               


#### Performance

 * Is going full `f32` worth it, and what about classification accuracy?


#### Building

 * How to better enable `avx2` from Cargo without resorting to `RUSTFLAGS`?


#### Idiomatic Rust

 * Implement [common traits](https://doc.rust-lang.org/1.0.0/style/features/traits/common.html). 
 * How should I name number variables consistently? `num_vectors`? `vectors`? `n_vectors`?   
 * How to name constructors that take multiple arguments? `with_a_b_c`, `with_something`, `with`?
 * How to implement common method for `Vec<T>` and `&[T]`?   


#### SIMD / Faster 

 * How to sum a f32s to a single scalar with [faster](https://github.com/AdamNiederer/faster)?
 * How to get rid of [itertools](https://github.com/bluss/rust-itertools) `zip` with [faster](https://github.com/AdamNiederer/faster)?
 * How to get rid of `SIMD_F32_WIDTH`?
