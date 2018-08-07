use std::ops::{Index, IndexMut};

use super::nsfw::{simd_vector_to_flat_slice, simd_vector_to_flat_slice_mut};
use super::rows::SimdRows;
use super::Simd;

#[derive(Debug)]
pub struct SimdMatrix<'a, SimdType: 'a>
where
    SimdType: Simd + Default + Clone,
{
    crate simd_rows: &'a SimdRows<SimdType>,
}

#[derive(Debug)]
pub struct SimdMatrixMut<'a, SimdType: 'a>
where
    SimdType: Simd + Default + Clone,
{
    crate simd_rows: &'a mut SimdRows<SimdType>,
}

impl<'a, SimdType: 'a> SimdMatrix<'a, SimdType>
where
    SimdType: Simd + Default + Clone,
{
    fn row(&self, row: usize) -> &[SimdType::Element] {
        let range = self.simd_rows.range_for_row(row);
        let vector_slice = &self.simd_rows.data[range];

        simd_vector_to_flat_slice(vector_slice, self.simd_rows.row_length)
    }
}

impl<'a, SimdType: 'a> SimdMatrixMut<'a, SimdType>
where
    SimdType: Simd + Default + Clone,
{
    pub fn row(&self, row: usize) -> &[SimdType::Element] {
        let range = self.simd_rows.range_for_row(row);
        let vector_slice = &self.simd_rows.data[range];

        simd_vector_to_flat_slice(vector_slice, self.simd_rows.row_length)
    }

    pub fn row_mut(&mut self, row: usize) -> &mut [SimdType::Element] {
        let range = self.simd_rows.range_for_row(row);
        let vector_slice = &mut self.simd_rows.data[range];

        simd_vector_to_flat_slice_mut(vector_slice, self.simd_rows.row_length)
    }
}

impl<SimdType> Index<(usize, usize)> for SimdMatrix<'a, SimdType>
where
    SimdType: Simd + Default + Clone,
{
    type Output = SimdType::Element;

    #[inline]
    fn index(&self, index: (usize, usize)) -> &Self::Output {
        let slice = self.row(index.0);
        &slice[index.1]
    }
}

impl<SimdType> Index<(usize, usize)> for SimdMatrixMut<'a, SimdType>
where
    SimdType: Simd + Default + Clone,
{
    type Output = SimdType::Element;

    #[inline]
    fn index(&self, index: (usize, usize)) -> &Self::Output {
        let slice = self.row(index.0);
        &slice[index.1]
    }
}

impl<SimdType> IndexMut<(usize, usize)> for SimdMatrixMut<'a, SimdType>
where
    SimdType: Simd + Default + Clone,
{
    #[inline]
    fn index_mut(&mut self, index: (usize, usize)) -> &mut Self::Output {
        let slice = self.row_mut(index.0);
        &mut slice[index.1]
    }
}
