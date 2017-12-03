use std;
use std::iter::{IntoIterator};
use rand::{ChaChaRng, Rng, Rand};

/// Basic Matrix we use for fast SIMD and parallel operations
#[derive(Debug)]
pub struct Matrix<T> where
    T : std::fmt::Debug,
    T : std::marker::Copy,
    T : std::marker::Sized,
    T : std::clone::Clone,
    T: Rand,
{
    pub vectors: usize,
    pub attributes: usize,

    pub data: Vec<T>
}


impl<T> Matrix<T> where
    T : std::fmt::Debug,
    T : std::marker::Sized,
    T : std::marker::Copy,
    T : std::clone::Clone,
    T: Rand,
{
    /// Creates a new Matrix
    pub fn new(vectors: usize, attributes: usize, default: T) -> Matrix<T> {
        Matrix::<T> {
            vectors,
            attributes,
            data: vec![default; vectors * attributes],
        }
    }

    /// Creates a new Matrix
    pub fn new_random(vectors: usize, attributes: usize) -> Matrix<T> {
        let size = vectors * attributes;
        let mut rng = ChaChaRng::new_unseeded();
        
        Matrix::<T> {
            vectors,
            attributes,
            data: rng.gen_iter().take(size).collect(),
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


pub struct IterMatrix<'a, T: 'a> where
    T : std::fmt::Debug,
    T : std::marker::Sized,
    T : std::marker::Copy,
    T : std::clone::Clone,
    T: Rand
{
    pub matrix: &'a Matrix<T>,
    pub index: usize,    
}


impl <'a, T> Iterator for IterMatrix<'a, T> where
    T : std::fmt::Debug,
    T : std::marker::Sized,
    T : std::marker::Copy,
    T : std::clone::Clone,
    T: Rand 
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


impl <'a, T> IntoIterator for &'a Matrix<T> where
    T : std::fmt::Debug,
    T : std::marker::Sized,
    T : std::marker::Copy,
    T : std::clone::Clone,
    T: Rand
{
    type Item = &'a [T];
    type IntoIter = IterMatrix<'a, T>;

    fn into_iter(self) -> Self::IntoIter {
        IterMatrix{ matrix: self, index: 0 } 
    }
}




#[test]
fn test_iter() {
    let matrix = Matrix::new(10, 5, 0);
    for x in &matrix {
        assert_eq!(x[0], 0);
    }
}
