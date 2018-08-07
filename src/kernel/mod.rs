mod rbf;

use crate::{f32s, SimdRows};

pub use self::rbf::RbfKernel;

/// Base trait for kernels
#[doc(hidden)]
pub trait Kernel {
    fn compute(&self, vectors: &SimdRows<f32s>, feature: &[f32s], output: &mut [f64]);
}
