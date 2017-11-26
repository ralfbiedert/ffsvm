extern crate ffsvm;

use ffsvm::parser::{parse_model_csvm};

pub fn main() {
//    let features: [f32; 8] = [1.0, 2.0, 3.0, 4.0, 5.0, 7.0, 1.0, 0.0];
//    let mut probabilities = [0.0f32; 8];
//
//
//    let result = predict_probability_csvm(&svm, &features, &mut probabilities);
//    println!("{:?}", result);
//
//
    let model_str: &str = include_str!("test.model");
    let model = parse_model_csvm(model_str);

    println!("{:?}", model);
}