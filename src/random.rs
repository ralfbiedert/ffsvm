use std::marker::{Copy, Sized};
use std::iter::Iterator;
use rand::{ChaChaRng, Rand, Rng};


/// Randomizes a data structure
pub trait Randomize {
    /// Randomizes data in a structure (mostly its vectors) within the structure's parameters.
    fn randomize(self) -> Self;
}


pub trait Random {
    /// Creates a new random thing.
    fn new_random() -> Self;
}


/// Creates a vector of random
pub fn random_vec<T>(size: usize) -> Vec<T>
where
    T: Rand,
{
    let mut rng = ChaChaRng::new_unseeded();
    rng.gen_iter().take(size).collect()
}

