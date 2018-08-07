mod rbf;

use crate::vectors::SimdVectorsf32;

pub use self::rbf::RbfKernel;

/// Base trait for kernels
#[doc(hidden)]
pub trait Kernel {
    fn compute(&self, vectors: &SimdVectorsf32, feature: &[f32], output: &mut [f64]);
}
