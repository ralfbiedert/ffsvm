/// Main type we use in here
pub type Feature = f32;
pub type Probability = f32;
pub type Class = u32;


/// Basic Matrix we use for fast SIMD and parallel operations

#[derive(Debug)]
pub struct Matrix<T> {
    pub vectors: usize,
    pub attributes: usize,
    pub data: Vec<T>
}


impl<T> Matrix<T> {

    /// Creates a new Matrix
    pub fn new(vectors: usize, attributes: usize) -> Matrix<T> {
        Matrix::<T> {
            vectors,
            attributes,
            data: Vec::<T>::with_capacity(vectors * attributes)
        }
    }

}


/// Base data for our CSVM
#[derive(Debug)]
pub struct CSVM {
}

#[derive(Debug)]
pub struct ModelCSVM {
    pub support_vectors: Matrix<Feature>
}
