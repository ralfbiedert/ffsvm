use std::{
    collections::{btree_map::Iter, BTreeMap},
    ops::{Index, IndexMut},
};

#[derive(Clone, Debug)]
pub struct SparseVector<T>
where
    T: Clone,
{
    entries: BTreeMap<usize, T>,
}

impl<T> SparseVector<T>
where
    T: Clone,
{
    pub fn new() -> Self { SparseVector { entries: BTreeMap::new() } }

    pub fn iter(&self) -> Iter<usize, T> { self.entries.iter() }
}

impl<T> Index<usize> for SparseVector<T>
where
    T: Copy + Sized,
{
    type Output = T;

    fn index(&self, index: usize) -> &T { &self.entries[&index] }
}

impl<T> IndexMut<usize> for SparseVector<T>
where
    T: Copy + Sized + Default,
{
    fn index_mut(&mut self, index: usize) -> &mut T {
        if !self.entries.contains_key(&index) {
            self.entries.insert(index, T::default());
        }

        // If the index were wrong we would panic anyway, so unwrapping here
        // does not really change anything.
        // println!("Getting {} for {} entries", index, self.entries.len());
        self.entries.get_mut(&index).unwrap()
    }
}

#[derive(Clone, Debug)]
pub struct SparseMatrix<T>
where
    T: Clone,
{
    vectors: Vec<SparseVector<T>>,
}

impl<T> SparseMatrix<T>
where
    T: Clone,
{
    pub fn with(rows: usize) -> Self {
        SparseMatrix {
            vectors: vec![SparseVector::new(); rows],
        }
    }

    pub fn row(&self, row: usize) -> &SparseVector<T> { &self.vectors[row] }

    #[inline]
    pub fn row_iter(&self) -> SparseMatrixIter<'_, T> { SparseMatrixIter { matrix: &self, index: 0 } }
}

impl<T> Index<(usize, usize)> for SparseMatrix<T>
where
    T: Copy + Sized,
{
    type Output = T;

    fn index(&self, index: (usize, usize)) -> &T { &self.vectors[index.0][index.1] }
}

impl<T> IndexMut<(usize, usize)> for SparseMatrix<T>
where
    T: Copy + Sized + Default,
{
    fn index_mut(&mut self, index: (usize, usize)) -> &mut T { &mut self.vectors[index.0][index.1] }
}

/// Basic iterator struct to go over matrix
#[derive(Clone, Debug)]
pub struct SparseMatrixIter<'a, T: 'a>
where
    T: Clone,
{
    /// Reference to the matrix we iterate over.
    crate matrix: &'a SparseMatrix<T>,

    /// Current index of vector iteration.
    crate index: usize,
}

impl<'a, T> Iterator for SparseMatrixIter<'a, T>
where
    T: Clone,
{
    type Item = &'a SparseVector<T>;

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        if self.index >= self.matrix.vectors.len() {
            None
        } else {
            self.index += 1;
            Some(&self.matrix.vectors[self.index - 1])
        }
    }
}
