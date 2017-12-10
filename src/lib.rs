// TODO:
// Run Clippy
// Go through Rust-idioms and adjust
// Consider creating chaining initialization ... new().randomX().randomY().
//
// Cleanup: STEP 1)


#![feature(toowned_clone_into)]
#![feature(test)]
#![feature(conservative_impl_trait)] // to "return impl FnMut"
#![feature(repr_simd)]

extern crate faster;
extern crate itertools;
#[macro_use]
extern crate nom;
extern crate rand;
extern crate rayon;
extern crate test;

pub mod manyvectors;
pub mod rbfcsvm;
pub mod parser;
pub mod data;
pub mod util;
pub mod randomization;
pub mod rbfkernel;


use data::Problem;
use rbfcsvm::RbfCSVM;
use parser::RawModel;



#[test]
fn test_simple_classification() {
    let model_str: &str = include_str!("lib.test.model");
    let model = RawModel::from_str(model_str).unwrap();
    let csvm = RbfCSVM::from_raw_model(&model).unwrap();

    let mut problem0 = Problem::from_svm(&csvm);
    let mut problem1 = Problem::from_svm(&csvm);

    problem0.features = vec![
        0.3093766,
        0.0,
        0.0,
        0.0,
        0.0,
        0.1764706,
        0.0,
        0.0,
        1.0,
        0.1137485,
        0f32,
        0f32,
        0f32,
        0f32,
        0f32,
        0f32,
    ];
    problem1.features = vec![
        0.3332312,
        0.0,
        0.0,
        0.0,
        0.09657142,
        1.0,
        0.0,
        0.0,
        1.0,
        0.09917226,
        0f32,
        0f32,
        0f32,
        0f32,
        0f32,
        0f32,
    ];

    csvm.predict_value_one(&mut problem0);
    csvm.predict_value_one(&mut problem1);

    assert_eq!(0, problem0.label);
    assert_eq!(1, problem1.label);
}
