#![feature(test)]
#![feature(try_from)]

use ffsvm::SVMResult;

fn similar(a: SVMResult, b: SVMResult) -> bool {
    match (a, b) {
        (SVMResult::Value(a), SVMResult::Value(b)) => (a - b).abs() < 0.001 * ((a + b) / 2.0),
        _ => false,
    }
}

macro_rules! test_model {
    ($name:ident, $file:expr, $prob:expr, $libsvm:expr, $libsvm_prob:expr) => {
        #[test]
        fn $name() -> Result<(), SVMError> {
            let model = include_str!(concat!("data/", $file));
            let svm = SVM::try_from(model)?;

            let mut problem_0 = Problem::from(&svm);
            problem_0.features_mut().clone_from_slice(&[
                0.00010000000092214275,
                0.00010000000054355651,
                0.00010000000063263872,
                0.00010000000020654017,
                0.00010000000077325587,
                0.00010000000089953001,
                0.00010000000064117786,
                0.00010000000020787097,
            ]);

            let mut problem_7 = Problem::from(&svm);
            problem_7.features_mut().clone_from_slice(&[
                1.2877848951077797,
                0.9860317088181307,
                1.4862471751386734,
                1.1280829602674647,
                0.8910305675176804,
                1.1643633497666765,
                0.9285991400016091,
                1.1407629818262937,
            ]);

            svm.predict_value(&mut problem_0)?;
            svm.predict_value(&mut problem_7)?;

            assert!(similar(problem_0.result(), SVMResult::Value($libsvm[0])));
            assert!(similar(problem_7.result(), SVMResult::Value($libsvm[1])));

            if $prob {
                svm.predict_probability(&mut problem_0)?;
                svm.predict_probability(&mut problem_7)?;

                assert!(similar(problem_0.result(), SVMResult::Value($libsvm_prob[0])));
                assert!(similar(problem_7.result(), SVMResult::Value($libsvm_prob[1])));
            }

            Ok(())
        }
    };
}

#[cfg(test)]
mod svm_regression {
    use super::similar;
    use ffsvm::{Predict, Problem, SVMError, SVMResult, SVM};
    use std::convert::TryFrom;

    // E-SVR

    test_model!(
        m_e_svr_linear_prob,
        "m_e_svr_linear_prob.libsvm",
        true,
        [0.369232, 7.1004],
        [0.369232, 7.1004]
    );
}
