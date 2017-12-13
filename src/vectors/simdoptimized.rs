use std::fmt;
use std::iter::IntoIterator;
use std::marker::{Copy, Sized};

use util;


/// Basic "matrix' we use for fast SIMD and parallel operations.
///
/// Note: Right now we use a Matrix mostly as a vector of vectors and is mostly
/// intended for read operations.
pub struct SimdOptimized<T>
    where
        T: Copy + Sized,
{
    /// Number of vectors this matrix has
    pub vectors: usize,

    /// Number of attributes this matrix has per subvector
    pub attributes: usize,

    /// Actual length of vectors
    pub vector_length: usize,

    /// We store all data in one giant array for performance reasons (caching)
    pub data: Vec<T>,
}





/// Basic iterator struct to go over matrix
pub struct IterManyVectors<'a, T: 'a>
    where
        T: Copy + Sized,
{
    /// Reference to the matrix we iterate over.
    pub matrix: &'a SimdOptimized<T>,

    /// Current index of vector iteration.
    pub index: usize,
}




impl<T> SimdOptimized<T>
    where
        T: Copy + Sized,
{
    /// Creates a new empty Matrix.
    pub fn with_dimension(vectors: usize, attributes: usize, default: T) -> SimdOptimized<T> {
        let preferred_length = util::prefered_simd_size(attributes);

        SimdOptimized::<T> {
            vectors,
            attributes,
            vector_length: preferred_length,
            data: vec![default; vectors * preferred_length],
        }
    }


    #[inline]
    pub fn get_vector(&self, index_vector: usize) -> &[T] {
        let start_index = self.offset(index_vector, 0);
        &self.data[start_index..start_index + self.vector_length]
    }

    #[inline]
    pub fn get_vector_mut(&mut self, index_vector: usize) -> &mut [T] {
        let start_index = self.offset(index_vector, 0);
        &mut self.data[start_index..start_index + self.vector_length]
    }


    #[inline]
    pub fn set_vector(&mut self, index_vector: usize, vector: &[T]) {
        let start_index = self.offset(index_vector, 0);
        for i in 0..self.attributes {
            self.data[start_index + i] = vector[i];
        }
    }

    #[inline]
    pub fn offset(&self, index_vector: usize, index_attribute: usize) -> usize {
        (index_vector * self.vector_length + index_attribute)
    }

    #[inline]
    pub fn set(&mut self, index_vector: usize, index_attribute: usize, value: T) {
        let index = self.offset(index_vector, index_attribute);
        self.data[index] = value;
    }

    #[inline]
    pub fn get(&self, index_vector: usize, index_attribute: usize) -> T {
        let index = self.offset(index_vector, index_attribute);
        self.data[index]
    }
}



impl<T> fmt::Debug for SimdOptimized<T>
    where
        T: Copy + Sized,
{
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "({}, {}, [data])", self.vectors, self.attributes)
    }
}



impl<'a, T> IntoIterator for &'a SimdOptimized<T>
    where
        T: Copy + Sized,
{
    type Item = &'a [T];
    type IntoIter = IterManyVectors<'a, T>;

    fn into_iter(self) -> Self::IntoIter {
        IterManyVectors {
            matrix: self,
            index: 0,
        }
    }
}



impl<'a, T> Iterator for IterManyVectors<'a, T>
    where
        T: Copy + Sized,
{
    type Item = &'a [T];

    fn next(&mut self) -> Option<Self::Item> {
        if self.index >= self.matrix.vectors {
            None
        } else {
            self.index += 1;
            Some(self.matrix.get_vector(self.index - 1))
        }
    }
}





#[cfg(test)]
mod tests {
    use vectors::simdoptimized::SimdOptimized;

    #[test]
    fn test_iter() {
        let matrix = SimdOptimized::with_dimension(10, 5, 0);
        for x in &matrix {
            assert_eq!(x[0], 0);
        }
    }
}


