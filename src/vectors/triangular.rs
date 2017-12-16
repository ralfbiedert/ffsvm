use std::ops::{Index, IndexMut};


// Layout of our data in this structure.
// The values will be stored from "left to right", from "up" to "down".
//
//    j --->
//       0   1   2   3
//  i 0      x   x   x
//  | 1          x   x
//  v 2              x
//    3


/// Square triangular matrix.
pub struct Triangular<T>
    where
        T: Copy + Sized,
{

    /// Width and height of the matrix 
    pub dimension: usize,
    
    /// Actual data
    pub data: Vec<T>,
}




impl <T> Triangular<T> where T: Copy + Sized,
{
    /// Creates a triangular with the given dimension.
    pub fn with_dimension(dimension: usize, default: T) -> Triangular<T> {
        let len = (dimension * (dimension - 1)) / 2;
        Triangular {
            dimension,
            data: vec![default; len]
        }
    }

    /// Computes the offset for a given i,j position.
    #[inline]
    pub fn offset(&self, i: usize, j: usize) -> usize {
        debug_assert!(i < j);
        debug_assert!(i < self.dimension);
        debug_assert!(j < self.dimension);

        //    j --->
        //       0   1   2   3   4
        //  i 0      x   x   x   x
        //  | 1          x   x   x
        //  v 2              x   x
        //    3                  x 
        // 
        // 0,0 -> invalid
        // 0,1 -> 0
        // 0,3 -> 2
        // 2,3 -> 5

        // base i:2 ==  (n-1) + (n-2) + ... + (n - j)
        //          ==  i*n + (-1 + -2 + ... + -j)
        //          ==  i*n - 1/2i (i+1)

        // for j = 2 ... 1 2       == 3
        // for j = 3 ... 1 2 3     == 6
        // for j = 4 ... 1 2 3 4   == 10
        // for j = 5 ... 1 2 3 4 5 == 15
        // x = 1/2 j (j+1)
        
        
        if i == 0 { return j - 1; }
        let last_index = i * self.dimension - (i / 2) * (i + 1);

        last_index + (j - i - 1)
    }

}



impl <T> Index<usize> for Triangular<T> where T: Copy + Sized,
{
    type Output = T;

    fn index(&self, index: usize) -> &T {
        let offset = self.offset(index, 0);
        &self.data[offset]
    }
}



impl <T> IndexMut<usize> for Triangular<T> where T: Copy + Sized,
{
    fn index_mut(&mut self, index: usize) -> &mut T {
        let offset = self.offset(index, 0);
        &mut self.data[offset]
    }
}



#[cfg(test)]
mod tests {
    use vectors::triangular::Triangular;

    #[test]
    fn test_index() {
        let matrix = Triangular::with_dimension(4, 0);

        assert_eq!(matrix.offset(0,1), 0);
        assert_eq!(matrix.offset(0,3), 2);
        assert_eq!(matrix.offset(2,3), 5);
    }
}
