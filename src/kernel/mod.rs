mod rbf;

use vectors::SimdOptimized;

pub use self::rbf::RbfKernel;




/// Base trait for kernels
pub trait Kernel {
    fn compute(&self, vectors: &SimdOptimized<f32>, feature: &[f32], output: &mut [f64]);
}
