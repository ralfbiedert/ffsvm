use faster::{ f32s };
//use stdsimd::{ f32x4};

/// Sum elements of a f32s ...
#[inline]
pub fn sum_f32s(v: f32s, simd_width: usize) -> f32 {
    let mut sum = 0.0f32;
    
    for i in 0 .. simd_width {
        sum += v.extract(i as u32);
    }
    
    sum
}

trait SimdSum {
    fn sss() -> Self;
}

impl SimdSum for f32s {
    
    fn sss() -> Self {
        f32s::splat(0.0f32)
    }
}