#![feature(test)]
#![feature(try_from)]

use ffsvm::Outcome;

fn similar(a: Outcome, b: Outcome) -> bool {
    match (a, b) {
        (Outcome::Value(a), Outcome::Value(b)) => (a - b).abs() < 0.001 * ((a + b) / 2.0),
        _ => false,
    }
}

macro_rules! test_model {
    ($name:ident, $file:expr, $prob:expr, $libsvm:expr, $libsvm_prob:expr) => {
        #[test]
        fn $name() -> Result<(), Error> {
            let model = include_str!(concat!("data_dense/", $file));
            let svm = DenseSVM::try_from(model)?;

            let mut problem_0 = Problem::from(&svm);
            let features_0 = problem_0.features().as_slice_mut();
            features_0.clone_from_slice(&[
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
            let features_7 = problem_7.features().as_slice_mut();
            features_7.clone_from_slice(&[
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

            assert!(similar(problem_0.result(), Outcome::Value($libsvm[0])));
            assert!(similar(problem_7.result(), Outcome::Value($libsvm[1])));

            if $prob {
                svm.predict_probability(&mut problem_0)?;
                svm.predict_probability(&mut problem_7)?;

                assert!(similar(problem_0.result(), Outcome::Value($libsvm_prob[0])));
                assert!(similar(problem_7.result(), Outcome::Value($libsvm_prob[1])));
            }

            Ok(())
        }
    };
}

#[cfg(test)]
mod svm_dense_regression {
    use super::similar;
    use ffsvm::{DenseSVM, Predict, Problem, Error, Outcome};
    use std::convert::TryFrom;

    // E-SVR

    test_model!(m_e_svr_linear_prob, "m_e_svr_linear_prob.libsvm", true, [0.369232, 7.1004], [0.369232, 7.1004]);
    test_model!(m_e_svr_poly_prob, "m_e_svr_poly_prob.libsvm", true, [2.71936, 6.89966], [2.71936, 6.89966]);
    test_model!(m_e_svr_rbf_prob, "m_e_svr_rbf_prob.libsvm", true, [0.581717, 6.39637], [0.581717, 6.39637]);
    test_model!(m_e_svr_sigmoid_prob, "m_e_svr_sigmoid_prob.libsvm", true, [0.0267491, 5.26548], [0.0267491, 5.26548]);

    test_model!(m_e_svr_linear, "m_e_svr_linear.libsvm", false, [0.369232, 7.1004], []);
    test_model!(m_e_svr_poly, "m_e_svr_poly.libsvm", false, [2.71936, 6.89966], []);
    test_model!(m_e_svr_rbf, "m_e_svr_rbf.libsvm", false, [0.581717, 6.39637], []);
    test_model!(m_e_svr_sigmoid, "m_e_svr_sigmoid.libsvm", false, [0.0267491, 5.26548], []);

    // Nu-SVR

    test_model!(m_nu_svr_linear_prob, "m_nu_svr_linear_prob.libsvm", true, [0.471485, 7.05909], [0.471485, 7.05909]);
    test_model!(m_nu_svr_poly_prob, "m_nu_svr_poly_prob.libsvm", true, [2.18783, 6.55455], [2.18783, 6.55455]);
    test_model!(m_nu_svr_rbf_prob, "m_nu_svr_rbf_prob.libsvm", true, [0.653419, 6.49803], [0.653419, 6.49803]);
    test_model!(m_nu_svr_sigmoid_prob, "m_nu_svr_sigmoid_prob.libsvm", true, [0.396866, 5.52985], [0.396866, 5.52985]);

    test_model!(m_nu_svr_linear, "m_nu_svr_linear.libsvm", false, [0.471485, 7.05909], []);
    test_model!(m_nu_svr_poly, "m_nu_svr_poly.libsvm", false, [2.18783, 6.55455], []);
    test_model!(m_nu_svr_rbf, "m_nu_svr_rbf.libsvm", false, [0.653419, 6.49803], []);
    test_model!(m_nu_svr_sigmoid, "m_nu_svr_sigmoid.libsvm", false, [0.396866, 5.52985], []);

}
