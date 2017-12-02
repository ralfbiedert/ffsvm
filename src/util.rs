use faster::{ f32s };

/// Sum elements of a f32s ...
#[inline]
pub fn sum_elements_f32(v: f32s, simd_width: usize) -> f32 {
    let mut sum = 0.0f32;
    
    for i in 0 .. simd_width {
        sum += v.extract(i as u32);
    }
    
    sum
}