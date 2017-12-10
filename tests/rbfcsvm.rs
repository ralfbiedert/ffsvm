#![feature(test)]

extern crate test;
extern crate ffsvm;

#[cfg(test)]
mod tests {
    use ffsvm::RbfCSVM;
    use ffsvm::Problem;
    use ffsvm::RawModel;
    
    #[test]
    fn rbfcsvm_test_small_model() {
        let model_str: &str = include_str!("test.small.model");
        let model = RawModel::from_str(model_str).unwrap();
        let csvm = RbfCSVM::from_raw_model(&model).unwrap();

        let mut problem0 = Problem::from_svm(&csvm);
        let mut problem1 = Problem::from_svm(&csvm);

        problem0.features = vec![ 0.3093766, 0.0, 0.0, 0.0, 0.0, 0.1764706, 0.0, 0.0, 1.0, 0.1137485, 0f32, 0f32, 0f32, 0f32, 0f32, 0f32 ];
        problem1.features = vec![ 0.3332312, 0.0, 0.0, 0.0, 0.09657142, 1.0, 0.0, 0.0, 1.0, 0.09917226, 0f32, 0f32, 0f32, 0f32, 0f32, 0f32 ];

        csvm.predict_value_one(&mut problem0);
        csvm.predict_value_one(&mut problem1);

        assert_eq!(0, problem0.label);
        assert_eq!(1, problem1.label);
    }
}
