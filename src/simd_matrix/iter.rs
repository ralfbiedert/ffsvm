use super::rows::SimdRows;
use super::Simd;

/// Basic iterator struct to go over matrix
#[derive(Clone, Debug)]
pub struct SimdRowsIter<'a, SimdType: 'a>
where
    SimdType: Simd + Default + Clone,
{
    /// Reference to the matrix we iterate over.
    crate simd_rows: &'a SimdRows<SimdType>,

    /// Current index of vector iteration.
    crate index: usize,
}

impl<SimdType> Iterator for SimdRowsIter<'a, SimdType>
where
    SimdType: Simd + Default + Clone,
{
    type Item = &'a [SimdType];

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        if self.index >= self.simd_rows.rows {
            None
        } else {
            let range = self.simd_rows.range_for_row(self.index);
            self.index += 1;
            Some(&self.simd_rows.data[range])
        }
    }
}
