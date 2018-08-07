use rand::{distributions, random};

use crate::simd_matrix::{Simd, SimdRows};

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

impl<T> Randomize for SimdRows<T>
where
    T: Simd + Sized + Copy + Default,
    T::Element: Default + Clone,
    distributions::Standard: distributions::Distribution<T::Element>,
{
    fn randomize(mut self) -> Self {
        let rows = self.rows();
        let row_length = self.row_length;
        let mut matrix = self.as_matrix_mut();

        for y in 0..rows {
            for x in 0..row_length {
                matrix[(y, x)] = random();
            }
        }

        self
    }
}
