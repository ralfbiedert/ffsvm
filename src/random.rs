use rand::{distributions, random};

use simd_aligned::*;

/// Randomizes a data structure
#[doc(hidden)]
pub trait Randomize {
    /// Randomizes data in a structure (mostly its vectors) within the structure's parameters.
    fn randomize(self) -> Self;
}

#[doc(hidden)]
pub trait Random {
    /// Creates a new random thing.
    fn new_random() -> Self;
}

impl<T, O> Randomize for SimdMatrix<T, O>
where
    T: Simd + Sized + Copy + Default,
    T::Element: Default + Clone,
    O: OptimizationStrategy,
    distributions::Standard: distributions::Distribution<T::Element>,
{
    fn randomize(mut self) -> Self {
        let (h, w) = self.dimension();
        let mut matrix = self.flat_mut();

        for y in 0..h {
            for x in 0..w {
                matrix[(y, x)] = random();
            }
        }

        self
    }
}

impl<T> Randomize for SimdVector<T>
where
    T: Simd + Sized + Copy + Default,
    T::Element: Default + Clone,
    distributions::Standard: distributions::Distribution<T::Element>,
{
    fn randomize(mut self) -> Self {
        let flat = self.flat_mut();

        for v in flat {
            *v = random();
        }

        self
    }
}
