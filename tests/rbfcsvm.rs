#![feature(test)]
#![feature(try_from)]

extern crate test;
extern crate ffsvm;


#[cfg(test)]
mod tests {
    use std::convert::TryFrom;
    use ffsvm::{RbfCSVM, PredictProblem, Problem};
    use ffsvm::{ModelFile};

  
    #[test]
    fn rbfcsvm_small() {
        let model_str: &str = include_str!("test.small.model");
        let model = ModelFile::try_from(model_str).unwrap();
        let csvm = RbfCSVM::try_from(&model).unwrap();

        let mut problem0 = Problem::from(&csvm);
        let mut problem1 = Problem::from(&csvm);

        problem0.features = vec![ 0.3093766, 0.0, 0.0, 0.0, 0.0, 0.1764706, 0.0, 0.0, 1.0, 0.1137485 ];
        problem1.features = vec![ 0.3332312, 0.0, 0.0, 0.0, 0.09657142, 1.0, 0.0, 0.0, 1.0, 0.09917226 ];

        csvm.predict_value(&mut problem0);
        csvm.predict_value(&mut problem1);

        // Results as per `libsvm`
        assert_eq!(0, problem0.label);
        assert_eq!(1, problem1.label);
    }
    
    #[test]
    fn rbfcsvm_large_prob() {
        let model_str: &str = include_str!("test.large.model");
        let model = ModelFile::try_from(model_str).unwrap();
        let csvm = RbfCSVM::try_from(&model).unwrap();

        let mut problem0 = Problem::from(&csvm);
        let mut problem1 = Problem::from(&csvm);
        let mut problem2 = Problem::from(&csvm);

        //256 0:0.5106233 1:0.1584117 2:0.1689098 3:0.1664358 4:0.2327561 5:0 6:0 7:0 8:1 9:0.1989241
        //256 0:0.5018305 1:0.0945542 2:0.09242307 3:0.09439687 4:0.1398575 5:0 6:0 7:0 8:1 9:1
        //256 0:0.5020829 1:0 2:0 3:0 4:0.1393665 5:1 6:0 7:0 8:1 9:0


        problem0.features = vec![ 0.5106233, 0.1584117, 0.1689098, 0.1664358, 0.2327561, 0.0, 0.0, 0.0, 1.0, 0.1989241, 0f32, 0f32, 0f32, 0f32, 0f32, 0f32 ];
        problem1.features = vec![ 0.5018305, 0.0945542, 0.09242307, 0.09439687, 0.1398575, 0.0, 0.0, 0.0, 1.0, 1.0 , 0f32, 0f32, 0f32, 0f32, 0f32, 0f32];
        problem2.features = vec![ 0.5020829, 0.0, 0.0, 0.0, 0.1393665, 1.0, 0.0, 0.0, 1.0, 0.0 , 0f32, 0f32, 0f32, 0f32, 0f32, 0f32];

        csvm.predict_probability(&mut problem0);
        csvm.predict_probability(&mut problem1);
        csvm.predict_probability(&mut problem2);

        // LibSVM output:
        // 0 0.809408 0.190592
        // 0 0.700839 0.299161
        // 1 0.0904989 0.909501
        assert_eq!(0, problem0.label);
        assert_eq!(0, problem1.label);
        assert_eq!(1, problem2.label);
        
        const DECIMALS: i32 = 2;

        assert!(approx_equal(problem0.probabilities[0], 0.809408, DECIMALS) );
        assert!(approx_equal(problem0.probabilities[1], 0.190592, DECIMALS) );
        
        assert!(approx_equal(problem1.probabilities[0], 0.700839, DECIMALS) );
        assert!(approx_equal(problem1.probabilities[1], 0.299161, DECIMALS) );
        
        assert!(approx_equal(problem2.probabilities[0], 0.0904989, DECIMALS) );
        assert!(approx_equal(problem2.probabilities[1], 0.909501, DECIMALS) );
    }

    
    // https://stackoverflow.com/questions/41447678/comparison-of-two-floats-in-rust-to-arbitrary-level-of-precision
    fn approx_equal(a: f64, b: f64, decimal_places: i32) -> bool {
        let factor = 10.0f64.powi(decimal_places);
        let a = (a * factor).trunc();
        let b = (b * factor).trunc();
        a == b
    }
}
