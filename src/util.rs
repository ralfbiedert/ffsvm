use faster::{f32s,f64s};
use rand::{ChaChaRng, Rng, Rand};

/// Sum elements of a f32s ... 
#[inline]
pub fn sum_f32s(v: f32s, simd_width: usize) -> f32 {
    let mut sum = 0.0f32;
    
    for i in 0 .. simd_width {
        sum += v.extract(i as u32);
    }
    
    sum
}

#[inline]
pub fn sum_f64s(v: f64s, simd_width: usize) -> f64 {
    let mut sum = 0.0;

    for i in 0 .. simd_width {
        sum += v.extract(i as u32);
    }

    sum
}

/// Creates a vector of random 
pub fn random_vec<T>(size: usize) -> Vec<T> 
where T: Rand
{
    let mut rng = ChaChaRng::new_unseeded();
    rng.gen_iter().take(size).collect()
}



/// Computes our prefered SIMD size for vectors. 
pub fn prefered_simd_size(size: usize) -> usize {
    const ALIGN: usize = 4;
    if size % ALIGN == 0 { 
        size 
    } else  { 
        ((size / ALIGN) + 1) * ALIGN 
    }
}

