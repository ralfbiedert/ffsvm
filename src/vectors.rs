use std::{
    fmt,
    ops::{Index, IndexMut},
};

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
#[derive(Clone, Default)]
pub struct Triangular<T>
where
    T: Copy + Sized,
{
    /// Width and height of the matrix
    crate dimension: usize,

    /// Actual data, see comment above for how indices are stored.
    crate data: Vec<T>,
}

impl<T> Triangular<T>
where
    T: Copy + Sized,
{
    /// Creates a triangular with the given dimension.
    pub fn with_dimension(dimension: usize, default: T) -> Triangular<T> {
        let len = match dimension {
            0 => 0,
            _ => (dimension * (dimension - 1)) / 2,
        };

        Triangular {
            dimension,
            data: vec![default; len],
        }
    }

    /// Computes the offset for a given i,j position.
    #[inline]
    pub fn offset(&self, i: usize, j: usize) -> usize {
        debug_assert!(i < j);
        debug_assert!(i < self.dimension);
        debug_assert!(j < self.dimension);

        //    j --->
        //       0   1   2   3   4  // n = 5
        //  i 0      x   x   x   x
        //  | 1          x   x   x
        //  v 2              x   x
        //    3                  x
        //
        // 0,0 -> invalid
        // 0,1 -> 0
        // 0,3 -> 2
        // 2,3 -> 5

        // base i:2 ==  (n-1) + (n-2) + ... + (n - i)
        //          ==  i*n + (-1 + -2 + ... + -i)
        //          ==  i*n - 1/2i (i+1)

        // i*2
        // for i = 2 ... 1 2       == 3
        // for i = 3 ... 1 2 3     == 6
        // for i = 4 ... 1 2 3 4   == 10
        // for i = 5 ... 1 2 3 4 5 == 15
        // x = (i * i + i) / 2

        // i:1, j:4
        // (n - 1)

        // The vast majority of calculations we do will be i == 0. Fast pass.
        if i == 0 {
            return j - 1;
        }

        // We're doing (i*i+i)/2 instead of i/2*(i+1) to prevent math errors.
        // Pro tip: don't write a function like this in the middle of the night ...
        let last_index = (i * self.dimension) - (i * i + i) / 2;

        last_index + (j - i - 1)
    }
}

impl<'a, T> From<&'a Vec<T>> for Triangular<T>
where
    T: Copy + Sized,
{
    fn from(vec: &Vec<T>) -> Self {
        // len  1:   dim: 2
        // len  3:   dim: 3
        // len  6:   dim: 4
        // len 10:   dim: 5

        // dim = round_down(sqrt(2 * len)) + 1
        //
        let dimension = (((2 * vec.len()) as f32).sqrt() as usize) + 1;

        Triangular { dimension, data: vec.clone() }
    }
}

impl<T> Index<(usize, usize)> for Triangular<T>
where
    T: Copy + Sized,
{
    type Output = T;

    fn index(&self, index: (usize, usize)) -> &T {
        let offset = self.offset(index.0, index.1);
        &self.data[offset]
    }
}

impl<T> Index<usize> for Triangular<T>
where
    T: Copy + Sized,
{
    type Output = T;

    fn index(&self, index: usize) -> &T { &self.data[index] }
}

impl<T> IndexMut<(usize, usize)> for Triangular<T>
where
    T: Copy + Sized,
{
    fn index_mut(&mut self, index: (usize, usize)) -> &mut T {
        let offset = self.offset(index.0, index.1);
        &mut self.data[offset]
    }
}

impl<T> fmt::Debug for Triangular<T>
where
    T: Copy + Sized,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result { write!(f, "(Triangular {}, [data])", self.dimension) }
}

#[cfg(test)]
mod tests {
    use super::Triangular;

    #[test]
    fn test_offset() {
        let matrix = Triangular::with_dimension(4, 0);

        assert_eq!(matrix.offset(0, 1), 0);
        assert_eq!(matrix.offset(0, 3), 2);
        assert_eq!(matrix.offset(2, 3), 5);

        let matrix = Triangular::with_dimension(5, 0);

        assert_eq!(matrix.offset(0, 1), 0);
        assert_eq!(matrix.offset(1, 4), 6);
        assert_eq!(matrix.offset(2, 3), 7);
        assert_eq!(matrix.offset(3, 4), 9);
    }

    #[test]
    fn test_index() {
        let mut matrix = Triangular::with_dimension(5, 0);

        matrix[(2, 3)] = 667;

        assert_eq!(matrix.data[7], 667);
    }

}
