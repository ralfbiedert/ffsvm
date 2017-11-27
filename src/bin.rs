extern crate ffsvm;

use ffsvm::parser::{parse_model};
use ffsvm::csvm::CSVM;
use ffsvm::matrix::Matrix;

pub fn main() {
    let model_str: &str = include_str!("test.model");
    let model = parse_model(model_str).unwrap();
    let problem = Matrix::new(1, 10, 0.0f32);

    let mut csvm = CSVM::new(&model).unwrap();
    
    csvm.predict_probability_csvm(&problem);
    
    println!("{:?}", csvm.num_classes);
}