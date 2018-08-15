use std::ops::{Index, IndexMut};

#[derive(Clone, Debug)]
struct Entry<T>
where
    T: Copy + Clone + Default,
{
    index: u32,
    value: T,
}

#[derive(Clone, Debug, Default)]
pub struct SparseVector<T>
where
    T: Clone + Copy + Default,
{
    entries: Vec<Entry<T>>,
}

impl<T> SparseVector<T>
where
    T: Clone + Copy + Default,
{
    pub fn new() -> Self { SparseVector { entries: Vec::new() } }

    pub fn clear(&mut self) { self.entries.clear(); }

    pub fn iter(&self) -> SparseVectorIter<'_, T> { SparseVectorIter { vector: self, index: 0 } }
}

/// Basic iterator struct to go over matrix
#[derive(Clone, Debug)]
pub struct SparseVectorIter<'a, T: 'a>
where
    T: Clone + Copy + Default,
{
    /// Reference to the matrix we iterate over.
    crate vector: &'a SparseVector<T>,

    /// Current index of vector iteration.
    crate index: usize,
}

impl<'a, T> Iterator for SparseVectorIter<'a, T>
where
    T: Clone + Copy + Default,
{
    type Item = (u32, T);

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        if self.index >= self.vector.entries.len() {
            None
        } else {
            let entry = &self.vector.entries[self.index];
            self.index += 1;
            Some((entry.index, entry.value))
        }
    }
}

impl<T> Index<usize> for SparseVector<T>
where
    T: Copy + Sized + Default,
{
    type Output = T;

    fn index(&self, index: usize) -> &T {
        // TODO: Beautify me

        for e in &self.entries {
            if e.index == index as u32 {
                return &e.value;
            }
        }

        // We can panic here since a regular index out of bounds would also panic.
        panic!("Index out of bounds.");
    }
}

impl<T> IndexMut<usize> for SparseVector<T>
where
    T: Copy + Sized + Default,
{
    fn index_mut(&mut self, index: usize) -> &mut T {
        // TODO: Beautify me

        let highest_so_far: i32 = match self.entries.last() {
            None => -1,
            Some(x) => x.index as i32,
        };

        if index as i32 <= highest_so_far {
            unimplemented!("We still need to implement unsorted insertion. As of today, you need to insert element in strictly ascending order.");
        }

        self.entries.push(Entry {
            index: index as u32,
            value: T::default(),
        });

        // `unwrap` should be safe since we just inserted that value.
        &mut self.entries.last_mut().unwrap().value
    }
}

#[derive(Clone, Debug)]
pub struct SparseMatrix<T>
where
    T: Clone + Copy + Default,
{
    vectors: Vec<SparseVector<T>>,
}

impl<T> SparseMatrix<T>
where
    T: Clone + Copy + Default,
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
    T: Copy + Sized + Default,
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
    T: Clone + Copy + Default,
{
    /// Reference to the matrix we iterate over.
    crate matrix: &'a SparseMatrix<T>,

    /// Current index of vector iteration.
    crate index: usize,
}

impl<'a, T> Iterator for SparseMatrixIter<'a, T>
where
    T: Clone + Copy + Default,
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
