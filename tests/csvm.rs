#![feature(test)]
#![feature(try_from)]

extern crate ffsvm;
extern crate test;

#[cfg(test)]
mod tests {
    use ffsvm::{ModelFile, Predict, Problem, SVMError, CSVM};
    use std::convert::TryFrom;

    #[test]
    fn rbfcsvm_multiclass() -> Result<(), SVMError> {
        let model_str: &str = include_str!("test.multiclass.model");
        let model = ModelFile::try_from(model_str)?;
        let csvm = CSVM::try_from(&model)?;

        let mut problem0 = Problem::from(&csvm);
        let mut problem1 = Problem::from(&csvm);

        problem0.features_mut().clone_from_slice(&[
            -0.55838, -0.157895, 0.581292, -0.221184, 0.135713, -0.874396, -0.563197, -1.0, -1.0,
        ]);

        problem1.features_mut().clone_from_slice(&[
            -0.371381, 0.100752, -1.0, -0.0467289, 0.0892856, -0.545894, -0.806691, 0.828571, -1.0,
        ]);

        csvm.predict_probability(&mut problem0).expect("Worked");
        csvm.predict_probability(&mut problem1).expect("Worked");

        assert_eq!(2, problem0.label());
        assert_eq!(7, problem1.label());

        // csvm.predict_value(&mut problem1);
        Ok(())
    }

    #[test]
    fn linearcsvm_small() -> Result<(), SVMError> {
        let model_str: &str = include_str!("x.model");
        let model = ModelFile::try_from(model_str)?;
        let csvm = CSVM::try_from(&model)?;

        let mut problem0 = Problem::from(&csvm);
        let mut problem1 = Problem::from(&csvm);

        problem0.features_mut().clone_from_slice(&[
            0.3093766, 0.0, 0.0, 0.0, 0.0, 0.1764706, 0.0, 0.0, 1.0, 0.1137485,
        ]);
        problem1.features_mut().clone_from_slice(&[
            0.3332312, 0.0, 0.0, 0.0, 0.09657142, 1.0, 0.0, 0.0, 1.0, 0.09917226,
        ]);

        csvm.predict_value(&mut problem0).expect("Worked");
        csvm.predict_value(&mut problem1).expect("Worked");

        // Results as per `libsvm`
        assert_eq!(0, problem0.label());
        assert_eq!(1, problem1.label());

        Ok(())
    }

    #[test]
    fn rbfcsvm_small() -> Result<(), SVMError> {
        let model_str: &str = include_str!("test.small.model");
        let model = ModelFile::try_from(model_str)?;
        let csvm = CSVM::try_from(&model)?;

        let mut problem0 = Problem::from(&csvm);
        let mut problem1 = Problem::from(&csvm);

        problem0.features_mut().clone_from_slice(&[
            0.3093766, 0.0, 0.0, 0.0, 0.0, 0.1764706, 0.0, 0.0, 1.0, 0.1137485,
        ]);
        problem1.features_mut().clone_from_slice(&[
            0.3332312, 0.0, 0.0, 0.0, 0.09657142, 1.0, 0.0, 0.0, 1.0, 0.09917226,
        ]);

        csvm.predict_value(&mut problem0).expect("Worked");
        csvm.predict_value(&mut problem1).expect("Worked");

        // Results as per `libsvm`
        assert_eq!(0, problem0.label());
        assert_eq!(1, problem1.label());

        Ok(())
    }

    #[test]
    fn rbfcsvm_large_prob() -> Result<(), SVMError> {
        let model_str: &str = include_str!("test.large.model");
        let model = ModelFile::try_from(model_str).unwrap();
        let csvm = CSVM::try_from(&model).unwrap();

        let mut problem0 = Problem::from(&csvm);
        let mut problem1 = Problem::from(&csvm);
        let mut problem2 = Problem::from(&csvm);

        assert_eq!(csvm.class_index_for_label(0), Some(0));
        assert_eq!(csvm.class_index_for_label(1), Some(1));
        assert_eq!(csvm.class_index_for_label(2), None);

        // 256 0:0.5106233 1:0.1584117 2:0.1689098 3:0.1664358 4:0.2327561 5:0 6:0 7:0 8:1 9:0.1989241
        // 256 0:0.5018305 1:0.0945542 2:0.09242307 3:0.09439687 4:0.1398575 5:0 6:0 7:0 8:1 9:1
        // 256 0:0.5020829 1:0 2:0 3:0 4:0.1393665 5:1 6:0 7:0 8:1 9:0

        problem0.features_mut().clone_from_slice(&[
            0.5106233, 0.1584117, 0.1689098, 0.1664358, 0.2327561, 0.0, 0.0, 0.0, 1.0, 0.1989241,
        ]);
        problem1.features_mut().clone_from_slice(&[
            0.5018305, 0.0945542, 0.09242307, 0.09439687, 0.1398575, 0.0, 0.0, 0.0, 1.0, 1.0,
        ]);
        problem2
            .features_mut()
            .clone_from_slice(&[0.5020829, 0.0, 0.0, 0.0, 0.1393665, 1.0, 0.0, 0.0, 1.0, 0.0]);

        csvm.predict_probability(&mut problem0).expect("Worked");
        csvm.predict_probability(&mut problem1).expect("Worked");
        csvm.predict_probability(&mut problem2).expect("Worked");

        // LibSVM output:
        // 0 0.809408 0.190592
        // 0 0.700839 0.299161
        // 1 0.0904989 0.909501
        assert_eq!(0, problem0.label());
        assert_eq!(0, problem1.label());
        assert_eq!(1, problem2.label());

        const DECIMALS: i32 = 2;

        assert!(approx_equal(
            problem0.probabilities()[0],
            0.809408,
            DECIMALS
        ));
        assert!(approx_equal(
            problem0.probabilities()[1],
            0.190592,
            DECIMALS
        ));

        assert!(approx_equal(
            problem1.probabilities()[0],
            0.700839,
            DECIMALS
        ));
        assert!(approx_equal(
            problem1.probabilities()[1],
            0.299161,
            DECIMALS
        ));

        assert!(approx_equal(
            problem2.probabilities()[0],
            0.0904989,
            DECIMALS
        ));
        assert!(approx_equal(
            problem2.probabilities()[1],
            0.909501,
            DECIMALS
        ));

        Ok(())
    }

    // https://stackoverflow.com/questions/41447678/comparison-of-two-floats-in-rust-to-arbitrary-level-of-precision
    fn approx_equal(a: f64, b: f64, decimal_places: i32) -> bool {
        let factor = 10.0f64.powi(decimal_places);
        let a = (a * factor).trunc();
        let b = (b * factor).trunc();

        (a - b).abs() < 0.001
    }
}
