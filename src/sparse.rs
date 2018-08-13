use std::collections::BTreeMap;

#[derive(Clone)]
crate struct SparseVector<T>
where
    T: Clone,
{
    entries: BTreeMap<u32, T>,
}

impl<T> SparseVector<T>
where
    T: Clone,
{
    fn new() -> Self { SparseVector { entries: BTreeMap::new() } }
}

#[derive(Clone)]
crate struct SparseMatrix<T>
where
    T: Clone,
{
    vectors: Vec<SparseVector<T>>,
}

impl<T> SparseMatrix<T>
where
    T: Clone,
{
    fn with(rows: usize) -> Self {
        SparseMatrix {
            vectors: vec![SparseVector::new(); rows],
        }
    }
}
