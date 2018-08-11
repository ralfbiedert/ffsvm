#![feature(test)]
#![feature(try_from)]

extern crate ffsvm;
extern crate test;

macro_rules! test_model {
    ($name:ident, $file:expr, $prob:expr, $libsvm:expr, $libsvm_prob:expr) => {
        #[test]
        fn $name() -> Result<(), SVMError> {
            let model = include_str!($file);
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

            assert_eq!(problem_0.label(), $libsvm[0], "predict_value(problem_0)");
            assert_eq!(problem_7.label(), $libsvm[1], "predict_value(problem_7)");

            if $prob {
                svm.predict_probability(&mut problem_0)?;
                svm.predict_probability(&mut problem_7)?;

                assert_eq!(
                    problem_0.label(),
                    $libsvm_prob[0],
                    "predict_probability(problem_0)"
                );
                assert_eq!(
                    problem_7.label(),
                    $libsvm_prob[1],
                    "predict_probability(problem_7)"
                );
            }

            Ok(())
        }
    };
}

#[cfg(test)]
mod tests {
    use ffsvm::{Predict, Problem, SVMError, SVM};
    use std::convert::TryFrom;

    test_model!(m_csvm_linear_prob, "m_csvm_linear_prob.libsvm", true, [0, 7], [0, 7]);
    test_model!(m_csvm_poly_prob, "m_csvm_poly_prob.libsvm", true, [0, 7], [5, 7]); // apparently `libSVM` gets this wrong
    test_model!(m_csvm_rbf_prob, "m_csvm_rbf_prob.libsvm", true, [0, 7], [2, 7]); // apparently `libSVM` gets this wrong
    test_model!(m_csvm_sigmoid_prob, "m_csvm_sigmoid_prob.libsvm", true, [0, 5], [0, 7]); // apparently `libSVM` gets this wrong

    test_model!(m_csvm_linear, "m_csvm_linear.libsvm", false, [0, 7], []);
    test_model!(m_csvm_poly, "m_csvm_poly.libsvm", false, [0, 7], []);
    test_model!(m_csvm_rbf, "m_csvm_rbf.libsvm", false, [0, 7], []);
    test_model!(m_csvm_sigmoid, "m_csvm_sigmoid.libsvm", false, [0, 5], []);

}
