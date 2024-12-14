use ffsvm::Label;

fn similar(a: Label, b: Label) -> bool {
    match (a, b) {
        (Label::Value(a), Label::Value(b)) => (a - b).abs() < 0.001 * ((a + b) / 2.0),
        _ => false,
    }
}

macro_rules! test_model {
    ($name:ident, $file:expr, $prob:expr, $libsvm:expr, $libsvm_prob:expr) => {
        #[test]
        fn $name() -> Result<(), Error> {
            let model = include_str!(concat!("data_dense/", $file));
            let svm = DenseSVM::try_from(model)?;

            let mut problem_0 = FeatureVector::from(&svm);
            let features_0 = problem_0.features();
            features_0.clone_from_slice(&[0.000_1, 0.000_1, 0.000_1, 0.000_1, 0.000_1, 0.000_1, 0.000_1, 0.000_1]);

            let mut problem_7 = FeatureVector::from(&svm);
            let features_7 = problem_7.features();
            features_7.clone_from_slice(&[1.287_784_9, 0.986_031_7, 1.486_247_2, 1.128_083, 0.891_030_55, 1.164_363_4, 0.928_599_1, 1.140_762_9]);

            svm.predict_value(&mut problem_0)?;
            svm.predict_value(&mut problem_7)?;

            assert!(similar(problem_0.label(), Label::Value($libsvm[0])));
            assert!(similar(problem_7.label(), Label::Value($libsvm[1])));

            if $prob {
                svm.predict_probability(&mut problem_0)?;
                svm.predict_probability(&mut problem_7)?;

                assert!(similar(problem_0.label(), Label::Value($libsvm_prob[0])));
                assert!(similar(problem_7.label(), Label::Value($libsvm_prob[1])));
            }

            Ok(())
        }
    };
}

#[cfg(test)]
mod svm_dense_regression {
    use super::similar;
    use ffsvm::{DenseSVM, Error, FeatureVector, Label, Predict};
    use std::convert::TryFrom;

    // E-SVR
    test_model!(m_e_svr_linear_prob, "m_e_svr_linear_prob.libsvm", true, [0.369_232, 7.1004], [0.369_232, 7.1004]);
    test_model!(m_e_svr_poly_prob, "m_e_svr_poly_prob.libsvm", true, [2.71936, 6.89966], [2.71936, 6.89966]);
    test_model!(m_e_svr_rbf_prob, "m_e_svr_rbf_prob.libsvm", true, [0.581_717, 6.39637], [0.581_717, 6.39637]);
    test_model!(m_e_svr_sigmoid_prob, "m_e_svr_sigmoid_prob.libsvm", true, [0.026_749_1, 5.26548], [0.026_749_1, 5.26548]);

    test_model!(m_e_svr_linear, "m_e_svr_linear.libsvm", false, [0.369_232, 7.1004], [0.0, 0.0]);
    test_model!(m_e_svr_poly, "m_e_svr_poly.libsvm", false, [2.71936, 6.89966], [0.0, 0.0]);
    test_model!(m_e_svr_rbf, "m_e_svr_rbf.libsvm", false, [0.581_717, 6.39637], [0.0, 0.0]);
    test_model!(m_e_svr_sigmoid, "m_e_svr_sigmoid.libsvm", false, [0.026_749_1, 5.26548], [0.0, 0.0]);

    // Nu-SVR
    test_model!(m_nu_svr_linear_prob, "m_nu_svr_linear_prob.libsvm", true, [0.471_485, 7.05909], [0.471_485, 7.05909]);
    test_model!(m_nu_svr_poly_prob, "m_nu_svr_poly_prob.libsvm", true, [2.18783, 6.55455], [2.18783, 6.55455]);
    test_model!(m_nu_svr_rbf_prob, "m_nu_svr_rbf_prob.libsvm", true, [0.653_419, 6.49803], [0.653_419, 6.49803]);
    test_model!(m_nu_svr_sigmoid_prob, "m_nu_svr_sigmoid_prob.libsvm", true, [0.396_866, 5.52985], [0.396_866, 5.52985]);

    test_model!(m_nu_svr_linear, "m_nu_svr_linear.libsvm", false, [0.471_485, 7.05909], [0.0, 0.0]);
    test_model!(m_nu_svr_poly, "m_nu_svr_poly.libsvm", false, [2.18783, 6.55455], [0.0, 0.0]);
    test_model!(m_nu_svr_rbf, "m_nu_svr_rbf.libsvm", false, [0.653_419, 6.49803], [0.0, 0.0]);
    test_model!(m_nu_svr_sigmoid, "m_nu_svr_sigmoid.libsvm", false, [0.396_866, 5.52985], [0.0, 0.0]);
}
