mod iter;
mod matrix;
mod nsfw; // `unsafe` was already taken ...
mod rows;

pub use packed_simd::*;

pub use self::rows::SimdRows;

/// This is copy-paste from `packed_simd`, where this trait is unfortunately
/// sealed right now. In the future this might come from `std::simd`.
pub trait Simd {
    /// Element type of the SIMD vector
    type Element;
    /// The number of elements in the SIMD vector.
    const LANES: usize;
    /// The type: `[u32; Self::N]`.
    type LanesType;
}

/// The "best know" `f32` type for this platform.
pub type f32s = f32x16;

/// The "best know" `f64` type for this platform.
pub type f64s = f64x8;

macro_rules! impl_simd {
    ($simd:ty, $element:ty, $lanes:expr, $lanestype:ty) => {
        impl Simd for $simd {
            type Element = $element;
            const LANES: usize = $lanes;
            type LanesType = $lanestype;
        }
    };
}

impl_simd!(f32x16, f32, 16, [f32; 16]);
impl_simd!(f32x8, f32, 8, [f32; 8]);
impl_simd!(f32x4, f32, 4, [f32; 4]);
impl_simd!(f64x8, f64, 8, [f64; 8]);
impl_simd!(f64x4, f64, 4, [f64; 4]);
impl_simd!(f64x2, f64, 2, [f64; 2]);

#[cfg(test)]
mod test {
    use super::rows::SimdRows;
    use super::{f32s, f32x16, Simd};

    #[test]
    fn f32x8() {
        let mut x = SimdRows::<f32s>::with_dimension(10, 10);
        let mut m = x.as_matrix_mut();
        m[(0, 1)] = 0.4;

        assert!((x[0][0].sum() - 0.4).abs() < 0.001);
    }
}
