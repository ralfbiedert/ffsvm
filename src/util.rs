use faster::{f32s, f64s};
use std::marker::Copy;
use std::cmp::PartialOrd;


// This is the worst possible hack. We have to find a proper way to get
// the word size from faster, AND also sum a vector to a scalar.
//
// Nothing relying on these two should exist in this code.
//
const SIMD_F32_WIDTH: usize = 8;
const SIMD_F64_WIDTH: usize = 4;


/// Sets all items of a mutable vector to the given value.
pub fn set_all<T>(vector: &mut Vec<T>, value: T)
where
    T: Copy,
{
    for item in vector.iter_mut() {
        *item = value;
    }
}

/// Finds the item with the maximum index.
pub fn find_max_index<T>(array: &[T]) -> usize
where
    T: PartialOrd,
{
    let mut vote_max_idx = 0;

    for i in 1..array.len() {
        if array[i] > array[vote_max_idx] {
            vote_max_idx = i;
        }
    }

    vote_max_idx
}



/// Sum elements of a f32s ...
#[inline]
pub fn sum_f32s(v: f32s) -> f32 {
    let mut sum = 0.0f32;

    for i in 0..SIMD_F32_WIDTH {
        sum += v.extract(i as u32);
    }

    sum
}

/// Sum elements of a f64s ...
#[inline]
pub fn sum_f64s(v: f64s) -> f64 {
    let mut sum = 0.0;

    for i in 0..SIMD_F64_WIDTH {
        sum += v.extract(i as u32);
    }

    sum
}



/// Computes our prefered SIMD size for vectors.
pub fn prefered_simd_size(size: usize) -> usize {
    if size % SIMD_F32_WIDTH == 0 {
        size
    } else {
        ((size / SIMD_F32_WIDTH) + 1) * SIMD_F32_WIDTH
    }
}
