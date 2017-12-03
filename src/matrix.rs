use std;
use std::fmt;
use std::iter::{IntoIterator};


/// Basic "matrix' we use for fast SIMD and parallel operations.
/// 
/// Note: Right now we use a Matrix mostly as a vector of vectors and is mostly 
/// intended for read operations.
pub struct Matrix<T> where
    T : std::marker::Copy,
    T : std::marker::Sized,
    T : std::clone::Clone,
{
    /// Number of vectors this matrix has 
    pub vectors: usize,
    
    /// Number of attributes this matrix has
    pub attributes: usize,
    
    /// We store all data in one giant array for performance reasons (caching)
    pub data: Vec<T>
}


/// Basic iterator struct to go over matrix 
pub struct IterMatrix<'a, T: 'a> where
    T : std::marker::Sized,
    T : std::marker::Copy,
    T : std::clone::Clone,
{
    pub matrix: &'a Matrix<T>,
    pub index: usize,
}



impl<T> Matrix<T> where
    T : std::marker::Sized,
    T : std::marker::Copy,
    T : std::clone::Clone,
{
    /// Creates a new emptry Matrix.
    pub fn new(vectors: usize, attributes: usize, default: T) -> Matrix<T> {
        Matrix::<T> {
            vectors,
            attributes,
            data: vec![default; vectors * attributes],
        }
    }
    
    /// Given a flat vec and dimensions, set the matrix with the given dimensions 
    pub fn from_flat_vec(vector: Vec<T>, vectors: usize, attributes: usize) -> Matrix<T> {
        Matrix::<T> {
            vectors,
            attributes,
            data: vector
        } 
    }

    #[inline]
    pub fn get_vector(&self, index_vector: usize) -> &[T] {
        let start_index = self.offset(index_vector, 0);
        &self.data[start_index..start_index + self.attributes]
    }

    #[inline]
    pub fn get_vector_mut(&mut self, index_vector: usize) -> &mut [T] {
        let start_index = self.offset(index_vector, 0);
        &mut self.data[start_index..start_index + self.attributes]
    }
    
    #[inline]
    pub fn set_vector(&mut self, index_vector: usize, vector: &[T]) {
        let start_index = self.offset(index_vector, 0);
        for i in 0 .. self.attributes {
            self.data[start_index + i] = vector[i];    
        }
    }
    
    #[inline]
    pub fn offset(&self, index_vector: usize, index_attribute: usize) -> usize {
        ((index_vector * self.attributes) + index_attribute)
    }
    
    #[inline]
    pub fn set(&mut self, index_vector: usize, index_attribute: usize, value: T) {
        let  index = self.offset(index_vector, index_attribute);
        self.data[index] = value;
    }

    #[inline]
    pub fn get(&self, index_vector: usize, index_attribute: usize) -> T {
        let  index = self.offset(index_vector, index_attribute);
        self.data[index]
    }
}


impl <T> fmt::Debug for Matrix<T> where
    T : std::marker::Sized,
    T : std::marker::Copy,
    T : std::clone::Clone,
{
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "({}, {}, [data])", self.vectors, self.attributes)
    }
}



impl <'a, T> IntoIterator for &'a Matrix<T> where
    T : std::marker::Sized,
    T : std::marker::Copy,
    T : std::clone::Clone,
{
    type Item = &'a [T];
    type IntoIter = IterMatrix<'a, T>;

    fn into_iter(self) -> Self::IntoIter {
        IterMatrix{ matrix: self, index: 0 }
    }
}



impl <'a, T> Iterator for IterMatrix<'a, T> where
    T : std::marker::Sized,
    T : std::marker::Copy,
    T : std::clone::Clone,
{
    type Item = &'a [T];
    
    fn next(&mut self) -> Option<Self::Item> {
        if self.index >= self.matrix.vectors { 
            None 
        } else {
            self.index += 1;
            Some(self.matrix.get_vector(self.index-1))
        }
    }
}



#[test]
fn test_iter() {
    let matrix = Matrix::new(10, 5, 0);
    for x in &matrix {
        assert_eq!(x[0], 0);
    }
}
