extern crate ffsvm;

use ffsvm::*;

use self::parser::*;
use self::types::*;
use self::classifier::{predict_probability_csvm};

pub fn main() {
    let svm = ModelCSVM{};
    let features: [f32; 8] = [1.0, 2.0, 3.0, 4.0, 5.0, 7.0, 1.0, 0.0];
    let mut probabilities = [0.0f32; 8];


    let result = predict_probability_csvm(&svm, &features, &mut probabilities);
    println!("{:?}", result);

    parse_model_csvm();

}