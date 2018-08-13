struct Entry<T> {
    key: u32,
    value: T,
}

struct SparseVector<T> {
    entries: Vec<Entry<T>>,
}

struct SparseMatrix<T> {
    vectors: Vec<Vec<Entry<T>>>,
}
