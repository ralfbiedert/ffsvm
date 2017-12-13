pub mod rbf;

use vectors::flat::ManyVectors;


/// Base trait for kernels
pub trait Kernel {
    fn compute(&self, vectors: &ManyVectors<f32>, feature: &[f32], kvalues: &mut [f64]);
}
