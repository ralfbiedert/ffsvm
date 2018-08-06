use std::{
    fmt,
    iter::IntoIterator,
    marker::{Copy, Sized},
    ops::Range,
    ops::{Index, IndexMut},
};

use packed_simd::f32x8;
use rand::distributions;

use crate::random::{random_vec, Randomize};

#[derive(Clone, Debug)]
struct SimdVectorsf32x8 {
    /// Number of rows this matrix has. Each row consists of a number
    /// of SIMD vectors, which when looked at "flattened" form that row.
    rows: usize,

    /// Number of attributes this matrix has per element.
    attributes: usize,

    /// Number of SIMD vectors used per row.
    vectors_per_row: usize,

    /// We store all data in one giant array for performance reasons (caching)
    data: Vec<f32x8>,
}

impl SimdVectorsf32x8 {
    /// Creates a new empty Matrix.
    pub fn with_dimension(rows: usize, attributes: usize) -> SimdVectorsf32x8 {
        let vectors_per_row = match (attributes / f32x8::lanes(), attributes % f32x8::lanes()) {
            (x, 0) => x,
            (x, _) => x + 1,
        };

        SimdVectorsf32x8 {
            rows,
            attributes,
            vectors_per_row,
            data: vec![f32x8::splat(0.0); vectors_per_row * rows],
        }
    }

    #[inline]
    fn row_range(&self, row: usize) -> Range<usize> {
        let start_offset = self.row_start_offset(row);
        start_offset..start_offset + self.vectors_per_row
    }

    fn row_as_flat_slice(&self, row: usize) -> &[f32] {
        let range = self.row_range(row);
        let vector_slice = &self.data[range];

        assert!(self.attributes <= 8 * self.vectors_per_row);

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
        unsafe { std::slice::from_raw_parts(vector_slice.as_ptr() as *const f32, self.attributes) }
    }

    fn row_as_flat_slice_mut(&mut self, row: usize) -> &mut [f32] {
        let range = self.row_range(row);
        let vector_slice = &mut self.data[range];

        assert!(self.attributes <= 8 * self.vectors_per_row);

        // See comment above.
        unsafe {
            std::slice::from_raw_parts_mut(vector_slice.as_mut_ptr() as *mut f32, self.attributes)
        }
    }

    /// Computes an offset for a vector and attribute.
    #[inline]
    fn row_start_offset(&self, row: usize) -> usize {
        row * self.vectors_per_row
    }

    /// Sets a vector with the given data.
    pub fn set_row(&mut self, row_index: usize, data: &[f32]) {
        self.row_as_flat_slice_mut(row_index).clone_from_slice(data);
    }

    /// For a given row, return all SIMD vectors.
    pub fn simd_row(&self, row_index: usize) -> &[f32x8] {
        &self.data[self.row_range(row_index)]
    }
}

/// Basic iterator struct to go over matrix
#[derive(Clone, Debug)]
pub struct IterSimdVectorsf32x8<'a> {
    /// Reference to the matrix we iterate over.
    matrix: &'a SimdVectorsf32x8,

    /// Current index of vector iteration.
    index: usize,
}

impl Index<(usize, usize)> for SimdVectorsf32x8 {
    type Output = f32;

    #[inline]
    fn index(&self, index: (usize, usize)) -> &f32 {
        let slice = self.row_as_flat_slice(index.0);
        &slice[index.1]
    }
}

impl IndexMut<(usize, usize)> for SimdVectorsf32x8 {
    #[inline]
    fn index_mut(&mut self, index: (usize, usize)) -> &mut f32 {
        let slice = self.row_as_flat_slice_mut(index.0);
        &mut slice[index.1]
    }
}

impl Index<usize> for SimdVectorsf32x8 {
    type Output = [f32];

    #[inline]
    fn index(&self, index: usize) -> &[f32] {
        self.row_as_flat_slice(index)
    }
}

impl IntoIterator for &'a SimdVectorsf32x8 {
    type Item = &'a [f32];
    type IntoIter = IterSimdVectorsf32x8<'a>;

    fn into_iter(self) -> Self::IntoIter {
        IterSimdVectorsf32x8 {
            matrix: self,
            index: 0,
        }
    }
}

impl<'a> Iterator for IterSimdVectorsf32x8<'a> {
    type Item = &'a [f32];

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

impl Randomize for SimdVectorsf32x8 {
    fn randomize(mut self) -> Self {
        for i in 0..self.rows {
            let vector = random_vec(self.attributes);

            self.set_row(i, vector.as_slice());
        }

        self
    }
}

#[cfg(test)]
mod tests {
    use super::SimdVectorsf32x8;
    use crate::random::Randomize;

    #[test]
    fn test_simdvectorsf32x8() {
        let mut matrix = SimdVectorsf32x8::with_dimension(9, 9);

        // SIMD length is valid
        assert_eq!(matrix.simd_row(0).len(), 2);

        // Basic indexing and SIMD access match
        matrix[(1, 1)] = 1.0;
        assert!((matrix.simd_row(1)[0].sum() - 1.0).abs() < 0.000001);

        // Basic indexing works
        matrix[(8, 8)] = 8.0;
        assert!((matrix[8][8] - 8.0).abs() < 0.000001);

        // Randomization works
        let mut matrix = matrix.randomize();
        assert!((matrix[8][8] - 8.0).abs() > 0.000001);
        assert!(matrix[0][0].abs() > 0.000001);
    }

}
