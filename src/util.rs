use faster::{f32s,f64s};
use std::marker::Copy;

const SIMD_F32_WIDTH: usize = 4;
const SIMD_F64_WIDTH: usize = 2;


/// Sets all items of a mutable vector to the given value.
pub fn set_all<T>(vector: &mut Vec<T>, value: T) where T: Copy {
    for item in vector.iter_mut() {
        *item = value;
    }
}



/// Sum elements of a f32s ... 
#[inline]
pub fn sum_f32s(v: f32s) -> f32 {
    let mut sum = 0.0f32;
    
    for i in 0 .. SIMD_F32_WIDTH {
        sum += v.extract(i as u32);
    }
    
    sum
}

#[inline]
pub fn sum_f64s(v: f64s) -> f64 {
    let mut sum = 0.0;

    for i in 0 .. SIMD_F64_WIDTH {
        sum += v.extract(i as u32);
    }

    sum
}



/// Computes our prefered SIMD size for vectors. 
pub fn prefered_simd_size(size: usize) -> usize {
    const ALIGN: usize = 4;
    if size % ALIGN == 0 {
        size 
    } else {
        ((size / ALIGN) + 1) * ALIGN
    }
}

