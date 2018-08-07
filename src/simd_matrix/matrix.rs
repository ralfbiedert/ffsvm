use std::ops::{Index, IndexMut};

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
        let ptr = vector_slice.as_ptr() as *const SimdType::Element;

        // This "should be safe(tm)" since:
        //
        // 1) a slice of `N x f32x8` elements are transformed into a slice of
        // `attributes * f32` elements, where `attributes <=  N * 8`.
        //
        // 2) The lifetime of the returned value should automatically match the self borrow.
        //
        // Having said this, as soon as `std::simd` (or similar) provides a safe way of handling
        // that for us, these lines should be removed!
        unsafe { std::slice::from_raw_parts(ptr, self.simd_rows.row_length) }
    }
}

impl<'a, SimdType: 'a> SimdMatrixMut<'a, SimdType>
where
    SimdType: Simd + Default + Clone,
{
    pub fn row(&self, row: usize) -> &[SimdType::Element] {
        let range = self.simd_rows.range_for_row(row);
        let vector_slice = &self.simd_rows.data[range];
        let ptr = vector_slice.as_ptr() as *const SimdType::Element;

        // See comment above.
        unsafe { std::slice::from_raw_parts(ptr, self.simd_rows.row_length) }
    }

    pub fn row_mut(&mut self, row: usize) -> &mut [SimdType::Element] {
        let range = self.simd_rows.range_for_row(row);
        let vector_slice = &mut self.simd_rows.data[range];
        let ptr = vector_slice.as_mut_ptr() as *mut SimdType::Element;

        // See comment above.
        unsafe { std::slice::from_raw_parts_mut(ptr, self.simd_rows.row_length) }
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
