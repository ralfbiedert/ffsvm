use std::marker::{Sized,Copy};
use std::iter::Iterator;
use rand::{ChaChaRng, Rng, Rand};

use manyvectors::ManyVectors;
use data::Class;


/// Randomizes a data structure 
pub trait Randomize {

    /// Randomizes data in a structure (mostly its vectors) within the strcuture's parameters.
    fn randomize(self) -> Self;
}



impl <T> Randomize for ManyVectors<T> where T : Sized + Copy + Rand {
    
    fn randomize(mut self) -> Self {
        let mut rng = ChaChaRng::new_unseeded();

        for i in 0 .. self.vectors {
            let gen = rng.gen_iter::<T>();
            let vector = gen.take(self.attributes).collect::<Vec<T>>();
            
            self.set_vector(i, vector.as_slice());
        }
        
        self
    }
}




impl Randomize for Class {

    fn randomize(mut self) -> Self {
        self.coefficients = self.coefficients.randomize();
        self.support_vectors = self.support_vectors.randomize();
        self
    }
    
}


/// Creates a vector of random 
pub fn random_vec<T>(size: usize) -> Vec<T> where T: Rand
{
    let mut rng = ChaChaRng::new_unseeded();
    rng.gen_iter().take(size).collect()
}


