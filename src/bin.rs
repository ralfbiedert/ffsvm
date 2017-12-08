extern crate ffsvm;

use ffsvm::parser::RawModel;
use ffsvm::csvm::{CSVM, Problem};


pub fn main() {

    let model_str: &str = include_str!("test.model");
    let model = RawModel::from_str(model_str).unwrap();
//    let mut problem = produce_problem(1, 10);
//
    let csvm = CSVM::from_raw_model(&model).unwrap();
    let mut problem1 = Problem::from_csvm(&csvm);
    let mut problem2 = Problem::from_csvm(&csvm);

    problem1.features = vec![0.3093766, 0.0, 0.0, 0.0, 0.0, 0.1764706, 0.0, 0.0, 1.0, 0.1137485];
    problem2.features = vec![0.3332312, 0.0, 0.0, 0.0, 0.09657142, 1.0, 0.0, 0.0, 1.0, 0.09917226];

//    csvm.predict_value_one(&mut problem1);
    csvm.predict_value_one(&mut problem2);
    
    println!("{:?} ", problem2);
    
//    println!("{:?} {:?}", problem1.label, problem2);
//
//    // 256 0:0.3093766 1:0 2:0 3:0 4:0 5:0.1764706 6:0 7:0 8:1 9:0.1137485
//    problem.set_vector(0, &[0.3093766, 0.0, 0.0, 0.0, 0.0, 0.1764706, 0.0, 0.0, 1.0, 0.1137485]);
//    csvm.predict_probability_csvm(&problem);
//
//    // -256 0:0.3332312 1:0 2:0 3:0 4:0.09657142 5:1 6:0 7:0 8:1 9:0.09917226
//    problem.set_vector(0, &[0.3332312, 0.0, 0.0, 0.0, 0.09657142, 1.0, 0.0, 0.0, 1.0, 0.09917226]);
//    csvm.predict_probability_csvm(&problem);


}


