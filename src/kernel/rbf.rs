use std::convert::From;
use faster::{IntoPackedRefIterator, IntoPackedZip, PackedZippedIterator, PackedIterator, Packed, f32s};

use kernel::Kernel;
use parser::ModelFile;
use vectors::SimdOptimized;
use random::Random;
use rand::random;


pub struct RbfKernel {
    pub gamma: f32,
}


impl Kernel for RbfKernel {

    fn compute(&self, vectors: &SimdOptimized<f32>, feature: &[f32], kernel_values: &mut [f64]) {

        // According to Instruments, for realistic SVMs and problems, the VAST majority of our
        // CPU time is spent in this loop.
        for (i, sv) in vectors.into_iter().enumerate() {

            // TODO: This allocates a Vec internally, doesn't it?
            let sum: f32 = (sv.simd_iter(), feature.simd_iter()).zip()
                .simd_map((f32s::splat(0f32), f32s::splat(0f32)), |(a,b)| (a - b) * (a - b))
                .simd_reduce(f32s::splat(0f32), f32s::splat(0f32), |a, v| a + v)
                .sum();
          
            // This seems to be the single-biggest CPU spike: saving back kernel_values,
            // and computing exp() (saving back seems to have 3x time impact over exp(),
            // but I might misread "Instruments" for that particular one).
            kernel_values[i] = f64::from((-self.gamma * sum).exp());
        }
    }
}


impl Random for RbfKernel {
    fn new_random() -> Self {
        RbfKernel { gamma: random() }
    }
}


impl <'a> From<&'a ModelFile<'a>> for RbfKernel {

    fn from(model: &'a ModelFile<'a>) -> Self {
        RbfKernel { gamma: model.header.gamma }
    }
}




#[cfg(test)]
mod tests {
    use faster::{IntoPackedRefIterator, IntoPackedZip, PackedZippedIterator, PackedIterator, Packed, f64s};

    #[test]
    fn test_faster_simd() {

        let a = vec![ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 ];
        let b = vec![ 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0 ];

        let slice_a = &a[..];
        let slice_b = &b[..];

        let sum: f64 = (slice_a.simd_iter(), slice_b.simd_iter()).zip()
            .simd_map((f64s::splat(0f64), f64s::splat(0f64)), |(a,b)| (a - b) * (a - b))
            .simd_reduce(f64s::splat(0f64), f64s::splat(0f64), |a, v| a + v)
            .sum();

    }
}

