use std::marker::Copy;
use std::cmp::PartialOrd;


////// TODO: REMOVE ME WHEN FASTER HAS WORKING PackedZipMap::next_partial!!!!!11

const SIMD_F32_WIDTH: usize = 8;

/// Computes our prefered SIMD size for vectors.
pub fn prefered_simd_size(size: usize) -> usize {
    if size % SIMD_F32_WIDTH == 0 {
        size
    } else {
        ((size / SIMD_F32_WIDTH) + 1) * SIMD_F32_WIDTH
    }
}

////////////////////////////////////////////////////////////////////////////////////


/// Sets all items of a mutable vector to the given value.
pub fn set_all<T>(vector: &mut [T], value: T)
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


/// As implemented in `libsvm`.  
pub fn sigmoid_predict(decision_value: f64, a: f64, b: f64) -> f64 {
    
    let fapb = decision_value * a + b;

    // Citing from the original libSVM implementation:
    // "1-p used later; avoid catastrophic cancellation"
    if fapb >= 0f64 {
        (-fapb).exp() / (1f64 + (-fapb).exp())        
    } else {
        1f64 / (1f64 + fapb.exp())
    }
    
}

