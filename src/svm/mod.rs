crate mod class;
crate mod csvm;
crate mod kernel;
crate mod predict;
crate mod problem;

use crate::vectors::Triangular;

#[derive(Clone, Debug, Default)]
crate struct Probabilities {
    crate a: Triangular<f64>,

    crate b: Triangular<f64>,
}
