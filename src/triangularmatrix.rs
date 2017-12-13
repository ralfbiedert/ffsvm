use std::fmt;
use std::iter::IntoIterator;
use std::marker::{Copy, Sized};

use util;


/// Square triangular matrix.
/// 
/// A data structure for matrices that look like this:
///     /// THIS SHIT BREAKS COMPILATION ...

/*
/
/      j ---
/         0   1   2   3
/    i 0      x   x   x
/    | 1          x   x
/      2              x
/      3

*/
/// 
/// The values will be stored from "left to right", from 
/// "up" to "down".
///
pub struct TriangularMatrix<T>
where
    T: Copy + Sized,
{
    
    /// Width and height of the matrix 
    pub dimension: usize,
    
    /// Actual data
    pub data: Vec<T>,
}


impl <T> TriangularMatrix<T>
where
    T: Copy + Sized, 
{
    pub fn with_dimension(dimension: usize, default: T) -> TriangularMatrix<T> {
        let len = (dimension * (dimension - 1)) / 2;
        TriangularMatrix {
            dimension,
            data: vec![default; len]
        }
    }

    /// 
    /// Setting
    /// THIS SHIT BREAKS COMPILATION ...
//    / 
//    /          
//    /      j ---
//    /         0   1   2   3
//    /    i 0      x   x   x
//    /    | 1          x   x
//    /      2              x
//    /      3
//    / 
//    / 0,0 -> invalid
//    / 0,1 -> 0
//    / 0,3 -> 3
//    / 2,3 -> 3
    pub fn set(&self, i: usize, j: usize) {
        debug_assert!(i < j);
        debug_assert!(i < self.dimension);
        debug_assert!(j < self.dimension);
        
    }
    
}


