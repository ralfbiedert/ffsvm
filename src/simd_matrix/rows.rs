use std::{
    fmt,
    iter::IntoIterator,
    marker::{Copy, Sized},
    ops::Range,
    ops::{Index, IndexMut},
};

use super::iter::SimdRowsIter;
use super::matrix::{SimdMatrix, SimdMatrixMut};
use super::nsfw::simd_vector_to_flat_slice_mut;
use super::Simd;

#[derive(Clone, Debug)]
pub struct SimdRows<SimdType>
where
    SimdType: Simd + Default + Clone,
{
    crate rows: usize,
    crate row_length: usize,
    crate vectors_per_row: usize,
    crate data: Vec<SimdType>,
}

impl<SimdType> SimdRows<SimdType>
where
    SimdType: Simd + Default + Clone,
{
    pub fn with_dimension(rows: usize, row_length: usize) -> SimdRows<SimdType> {
        let vectors_per_row = match (row_length / SimdType::LANES, row_length % SimdType::LANES) {
            (x, 0) => x,
            (x, _) => x + 1,
        };

        SimdRows {
            rows,
            row_length,
            vectors_per_row,
            data: vec![SimdType::default(); vectors_per_row * rows],
        }
    }

    pub fn rows(&self) -> usize {
        self.rows
    }

    pub fn row_length(&self) -> usize {
        self.row_length
    }

    /// Returns a flat matrix view of this Simd rows collection type
    pub fn as_matrix(&self) -> SimdMatrix<'_, SimdType> {
        SimdMatrix { simd_rows: &self }
    }

    /// Returns a mutable matrix view of this Simd rows collection type
    pub fn as_matrix_mut(&mut self) -> SimdMatrixMut<'_, SimdType> {
        SimdMatrixMut {
            simd_rows: &mut *self,
        }
    }

    pub fn as_slice_mut(&mut self) -> &mut [SimdType::Element] {
        // This function only makes sense if we have exactly 1 row, otherwise there will be weird gaps.
        assert_eq!(self.rows, 1);

        simd_vector_to_flat_slice_mut(&mut self.data, self.row_length)
    }

    /// Computes an offset for a vector and attribute.
    #[inline]
    crate fn row_start_offset(&self, row: usize) -> usize {
        row * self.vectors_per_row
    }

    /// Returns the range of SIMD vectors for the given row.
    #[inline]
    crate fn range_for_row(&self, row: usize) -> Range<usize> {
        let start = self.row_start_offset(row);
        let end = start + self.vectors_per_row;
        start..end
    }
}

impl<SimdType> Index<usize> for SimdRows<SimdType>
where
    SimdType: Simd + Default + Clone,
{
    type Output = [SimdType];

    #[inline]
    fn index(&self, index: usize) -> &Self::Output {
        let range = self.range_for_row(index);
        &self.data[range]
    }
}

impl<SimdType> IndexMut<usize> for SimdRows<SimdType>
where
    SimdType: Simd + Default + Clone,
{
    #[inline]
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        let range = self.range_for_row(index);
        &mut self.data[range]
    }
}

impl<SimdType> IntoIterator for &'a SimdRows<SimdType>
where
    SimdType: Simd + Default + Clone,
{
    type Item = &'a [SimdType];
    type IntoIter = SimdRowsIter<'a, SimdType>;

    fn into_iter(self) -> Self::IntoIter {
        SimdRowsIter {
            simd_rows: self,
            index: 0,
        }
    }
}
