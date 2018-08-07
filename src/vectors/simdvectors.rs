use std::{
    fmt,
    iter::IntoIterator,
    marker::{Copy, Sized},
    ops::Range,
    ops::{Index, IndexMut},
};

use packed_simd::{f32x16, f32x4, f32x8, f64x2, f64x4, f64x8};
use rand::distributions;

use crate::random::{random_vec, Randomize};

pub type SimdVectorsf32 = SimdVectorsf32x16;
pub type SimdVectorsf64 = SimdVectorsf64x8;

pub type f32s = f32x16;
pub type f64s = f64x8;

macro_rules! simd_vector_impl {
    ($name:ident, $iter:ident, $vector:ident, $base:ty, $width:expr) => {
        #[derive(Clone, Debug)]
        pub struct $name {
            /// Number of rows this matrix has. Each row consists of a number
            /// of SIMD vectors, which when looked at "flattened" form that row.
            rows: usize,

            /// Number of attributes this matrix has per element.
            attributes: usize,

            /// Number of SIMD vectors used per row.
            vectors_per_row: usize,

            /// We store all data in one giant array for performance reasons (caching)
            data: Vec<$vector>,
        }

        impl $name {
            /// Creates a new empty Matrix.
            pub fn with_dimension(rows: usize, attributes: usize) -> $name {
                let vectors_per_row = match (attributes / $vector::lanes(), attributes % $vector::lanes()) {
                    (x, 0) => x,
                    (x, _) => x + 1,
                };

                $name {
                    rows,
                    attributes,
                    vectors_per_row,
                    data: vec![$vector::splat(0.0); vectors_per_row * rows],
                }
            }

            #[inline]
            fn row_range(&self, row: usize) -> Range<usize> {
                let start_offset = self.row_start_offset(row);
                start_offset..start_offset + self.vectors_per_row
            }

            fn row_as_flat_slice(&self, row: usize) -> &[$base] {
                let range = self.row_range(row);
                let vector_slice = &self.data[range];

                assert!(self.attributes <= $vector::lanes() * self.vectors_per_row);

                // This "should be safe(tm)" since:
                //
                // 1) a slice of `N x f32x8` elements are transformed into a slice of
                // `attributes * f32` elements, where `attributes <=  N * 8`, as ensured by
                // the assert.
                //
                // 2) The lifetime of the returned value should automatically match the self borrow.
                //
                // Having said this, as soon as `std::simd` (or similar) provides a safe way of handling
                // that for us, these lines should be removed!
                unsafe { std::slice::from_raw_parts(vector_slice.as_ptr() as *const $base, self.attributes) }
            }

            fn row_as_flat_slice_mut(&mut self, row: usize) -> &mut [$base] {
                let range = self.row_range(row);
                let vector_slice = &mut self.data[range];

                assert!(self.attributes <= $vector::lanes() * self.vectors_per_row);

                // See comment above.
                unsafe {
                    std::slice::from_raw_parts_mut(vector_slice.as_mut_ptr() as *mut $base, self.attributes)
                }
            }

            /// Computes an offset for a vector and attribute.
            #[inline]
            fn row_start_offset(&self, row: usize) -> usize {
                row * self.vectors_per_row
            }

            /// Sets a vector with the given data.
            pub fn set_row(&mut self, row_index: usize, data: &[$base]) {
                self.row_as_flat_slice_mut(row_index).clone_from_slice(data);
            }

            /// For a given row, return all SIMD vectors.
            pub fn simd_row(&self, row_index: usize) -> &[$vector] {
                &self.data[self.row_range(row_index)]
            }
        }

        /// Basic iterator struct to go over matrix
        #[derive(Clone, Debug)]
        pub struct $iter<'a> {
            /// Reference to the matrix we iterate over.
            matrix: &'a $name,

            /// Current index of vector iteration.
            index: usize,
        }

        impl Index<(usize, usize)> for $name {
            type Output = $base;

            #[inline]
            fn index(&self, index: (usize, usize)) -> &Self::Output {
                let slice = self.row_as_flat_slice(index.0);
                &slice[index.1]
            }
        }

        impl IndexMut<(usize, usize)> for $name {
            #[inline]
            fn index_mut(&mut self, index: (usize, usize)) -> &mut Self::Output {
                let slice = self.row_as_flat_slice_mut(index.0);
                &mut slice[index.1]
            }
        }

        impl Index<usize> for $name {
            type Output = [$base];

            #[inline]
            fn index(&self, index: usize) -> &Self::Output {
                self.row_as_flat_slice(index)
            }
        }

        impl IndexMut<usize> for $name {
            #[inline]
            fn index_mut(&mut self, index: usize) -> &mut Self::Output {
                self.row_as_flat_slice_mut(index)
            }
        }

        impl IntoIterator for &'a $name {
            type Item = &'a [$base];
            type IntoIter = $iter<'a>;

            fn into_iter(self) -> Self::IntoIter {
                $iter {
                    matrix: self,
                    index: 0,
                }
            }
        }

        impl<'a> Iterator for $iter<'a> {
            type Item = &'a [$base];

            #[inline]
            fn next(&mut self) -> Option<Self::Item> {
                if self.index >= self.matrix.rows {
                    None
                } else {
                    self.index += 1;
                    Some(&self.matrix[self.index - 1])
                }
            }
        }

        impl Randomize for $name {
            fn randomize(mut self) -> Self {
                for i in 0..self.rows {
                    let vector = random_vec(self.attributes);

                    self.set_row(i, vector.as_slice());
                }

                self
            }
        }
    };
}

simd_vector_impl!(SimdVectorsf32x4, IterSimdVectorsf32x4, f32x4, f32, 4);
simd_vector_impl!(SimdVectorsf32x8, IterSimdVectorsf32x8, f32x8, f32, 8);
simd_vector_impl!(SimdVectorsf32x16, IterSimdVectorsf32x16, f32x16, f32, 16);
simd_vector_impl!(SimdVectorsf64x2, IterSimdVectorsf64x2, f64x2, f64, 2);
simd_vector_impl!(SimdVectorsf64x4, IterSimdVectorsf64x4, f64x4, f64, 4);
simd_vector_impl!(SimdVectorsf64x8, IterSimdVectorsf64x8, f64x8, f64, 8);

#[cfg(test)]
mod test {
    use super::SimdVectorsf32x8;
    use crate::random::Randomize;
    use packed_simd::{f32x8, f64x4};

    #[test]
    fn f32x8() {
        let width = f32x8::lanes();
        let mut matrix = SimdVectorsf32x8::with_dimension(width + 1, width + 1);

        // SIMD length is valid
        assert_eq!(matrix.simd_row(0).len(), 2);

        // Basic indexing and SIMD access match
        matrix[(1, 1)] = 1.0;
        assert!((matrix.simd_row(1)[0].sum() - 1.0).abs() < 0.000001);

        // Basic indexing works
        matrix[(width, width)] = 1.0;
        assert!((matrix[width][width] - 1.0).abs() < 0.000001);

        // Randomization works
        let mut matrix = matrix.randomize();
        assert!((matrix[width][width] - 1.0).abs() > 0.000001);
        assert!(matrix[0][0].abs() > 0.000001);
    }
}
