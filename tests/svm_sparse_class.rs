macro_rules! test_model {
    ($name:ident, $file:expr, $prob:expr, $libsvm:expr, $libsvm_prob:expr) => {
        #[test]
        fn $name() -> Result<(), Error> {
            let model = include_str!(concat!("data_sparse/", $file));
            let svm = SparseSVM::try_from(model)?;

            let mut problem_0 = FeatureVector::from(&svm);
            let features_0 = problem_0.features();
            features_0[3] = 0.000_1;
            features_0[4] = 0.000_1;
            features_0[7] = 0.000_1;
            features_0[12] = 0.000_1;
            features_0[18] = 0.000_1;
            features_0[21] = 0.000_1;
            features_0[32] = 0.000_1;
            features_0[34] = 0.000_1;
            features_0[50] = 0.000_1;
            features_0[73] = 0.000_1;
            features_0[123] = 0.000_1;
            features_0[127] = 0.000_1;

            let mut problem_7 = FeatureVector::from(&svm);
            let features_7 = problem_7.features();
            features_7[3] = 0.930_907_6;
            features_7[4] = 1.264_398_9;
            features_7[6] = 1.417_500_6;
            features_7[32] = 1.090_475_8;
            features_7[46] = 1.475_075;
            features_7[54] = 0.902_898_55;
            features_7[74] = 1.504_974_4;
            features_7[92] = 0.908_902_6;
            features_7[95] = 1.274_937_5;
            features_7[98] = 1.234_927_2;
            features_7[110] = 1.500_999;

            svm.predict_value(&mut problem_0)?;
            svm.predict_value(&mut problem_7)?;

            assert_eq!(problem_0.label(), Label::Class($libsvm[0]), "predict_value(problem_0)");
            assert_eq!(problem_7.label(), Label::Class($libsvm[1]), "predict_value(problem_7)");

            if $prob {
                svm.predict_probability(&mut problem_0)?;
                svm.predict_probability(&mut problem_7)?;

                assert_eq!(problem_0.label(), Label::Class($libsvm_prob[0]), "predict_probability(problem_0)");
                assert_eq!(problem_7.label(), Label::Class($libsvm_prob[1]), "predict_probability(problem_7)");
            }

            Ok(())
        }
    };
}

#[cfg(test)]
mod svm_sparse_class {
    use ffsvm::{Error, FeatureVector, Label, Predict, SparseSVM};
    use std::convert::TryFrom;

    // CSVM
    test_model!(m_csvm_linear_prob, "m_csvm_linear_prob.libsvm", true, [0, 7], [1, 6]);
    test_model!(m_csvm_poly_prob, "m_csvm_poly_prob.libsvm", true, [0, 7], [0, 6]);
    test_model!(m_csvm_rbf_prob, "m_csvm_rbf_prob.libsvm", true, [0, 7], [7, 1]);
    test_model!(m_csvm_sigmoid_prob, "m_csvm_sigmoid_prob.libsvm", true, [0, 7], [7, 1]);

    // Temporarily disabled as they trigger ICE in Rust Nightly
    test_model!(m_csvm_linear, "m_csvm_linear.libsvm", false, [0, 7], [0, 0]);
    test_model!(m_csvm_poly, "m_csvm_poly.libsvm", false, [0, 7], [0, 0]);
    test_model!(m_csvm_rbf, "m_csvm_rbf.libsvm", false, [0, 7], [0, 0]);
    // TODO: Why do these fail?
    // test_model!(m_csvm_sigmoid, "m_csvm_sigmoid.libsvm", false, [0, 5], [0, 0]);

    // NUSVM
    // TODO: Why do these fail?
    // test_model!(m_nusvm_linear_prob, "m_nusvm_linear_prob.libsvm", true, [0, 7], [1, 6]);
    // test_model!(m_nusvm_poly_prob, "m_nusvm_poly_prob.libsvm", true, [0, 7], [0, 7]);
    // test_model!(m_nusvm_rbf_prob, "m_nusvm_rbf_prob.libsvm", true, [0, 7], [0, 7]);
    // test_model!(m_nusvm_sigmoid_prob, "m_nusvm_sigmoid_prob.libsvm", true, [0, 7], [0, 7]);

    // Temporarily disabled as they trigger ICE in Rust Nightly
    test_model!(m_nusvm_linear, "m_nusvm_linear.libsvm", false, [0, 7], [0, 0]);
    test_model!(m_nusvm_poly, "m_nusvm_poly.libsvm", false, [0, 7], [0, 0]);
    test_model!(m_nusvm_rbf, "m_nusvm_rbf.libsvm", false, [0, 7], [0, 0]);
    test_model!(m_nusvm_sigmoid, "m_nusvm_sigmoid.libsvm", false, [0, 7], [0, 0]);
}
