#![feature(test)]
#![feature(try_from)]

extern crate test;
extern crate ffsvm;

#[cfg(test)]
mod tests {
    use std::convert::TryFrom;
    use ffsvm::{RbfCSVM, PredictProblem, Problem};
    use ffsvm::{ModelFile};

    
    // This doesn't work like that. Support vectors are not guaranteed 
    // to be classified into their respective class ...
    //
    //    fn rbfcsvm_test_model(model_str: &str) {
    //        let model = RawModel::from_str(model_str).unwrap();
    //        let csvm: RbfCSVM = RbfCSVM::from_raw_model(&model).unwrap();
    //        let num_preferred_attributes = util::prefered_simd_size(csvm.num_attributes);
    //
    //        for class in &csvm.classes {
    //            for support_vector in &class.support_vectors {
    //                let mut v = vec![0f32; num_preferred_attributes];
    //
    //                for (i, a) in support_vector.iter().enumerate() {
    //                    v[i] = *a;
    //                }
    //
    //                let mut problem = Problem { features: v, .. Problem::from_svm(&csvm) };
    //
    //                csvm.predict_value(&mut problem);
    //
    //                println!("{:?}: {:?} {:?}", problem.features, class.label, problem.label);
    //                assert_eq!(class.label, problem.label);
    //            }
    //        }
    //
    //
    //    }
    
    #[test]
    fn rbfcsvm_small() {
        let model_str: &str = include_str!("test.small.model");
        let model = ModelFile::try_from(model_str).unwrap();
        let csvm = RbfCSVM::try_from(&model).unwrap();

        let mut problem0 = Problem::from(&csvm);
        let mut problem1 = Problem::from(&csvm);

        problem0.features = vec![ 0.3093766, 0.0, 0.0, 0.0, 0.0, 0.1764706, 0.0, 0.0, 1.0, 0.1137485, 0f32, 0f32, 0f32, 0f32, 0f32, 0f32 ];
        problem1.features = vec![ 0.3332312, 0.0, 0.0, 0.0, 0.09657142, 1.0, 0.0, 0.0, 1.0, 0.09917226, 0f32, 0f32, 0f32, 0f32, 0f32, 0f32 ];

        csvm.predict_value(&mut problem0);
        csvm.predict_value(&mut problem1);

        assert_eq!(0, problem0.label);
        assert_eq!(1, problem1.label);
    }
}
