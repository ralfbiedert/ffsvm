
# Performance History

TODO: Paste Excel numbers here



# Open Questions                                               


#### Performance

 * Is going full `f32` worth it, and what about classification accuracy?


#### Building

 * How to better enable `avx2` from Cargo without resorting to `RUSTFLAGS`?


#### Idiomatic Rust

 * How should I name number variables consistently? `num_vectors`? `vectors`? `n_vectors`?   
 * How to name constructors that take multiple arguments? `with_a_b_c`, `with_something`, `with`?   


#### SIMD / Faster 

 * How to sum a f32s to a single scalar with [faster](https://github.com/AdamNiederer/faster)?
 * How to get rid of [itertools](https://github.com/bluss/rust-itertools) `zip` with [faster](https://github.com/AdamNiederer/faster)?
 * How to get rid of `SIMD_F32_WIDTH`?
