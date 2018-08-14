#![feature(test)]
#![feature(try_from)]

extern crate ffsvm;
extern crate test;

macro_rules! test_model {
    ($name:ident, $file:expr, $prob:expr, $libsvm:expr, $libsvm_prob:expr) => {
        #[test]
        fn $name() -> Result<(), SVMError> {
            let model = include_str!(concat!("data_sparse/", $file));
            let svm = SparseSVM::try_from(model)?;

            let mut problem_0 = Problem::from(&svm);
            let features_0 = problem_0.features();
            features_0[3] = 0.0001000000007146063;
            features_0[4] = 0.00010000000018581445;
            features_0[7] = 0.00010000000043775396;
            features_0[12] = 0.00010000000060915153;
            features_0[18] = 0.00010000000016903845;
            features_0[21] = 0.00010000000089347425;
            features_0[32] = 0.00010000000034352026;
            features_0[34] = 0.00010000000018032126;
            features_0[50] = 0.00010000000020026886;
            features_0[73] = 0.00010000000000769077;
            features_0[123] = 0.0001000000003393198;
            features_0[127] = 0.0001000000002766062;

            let mut problem_7 = Problem::from(&svm);
            let features_7 = problem_7.features();
            features_7[3] = 0.9309075801528132;
            features_7[4] = 1.26439892382077;
            features_7[6] = 1.4175005579408642;
            features_7[32] = 1.0904757546581592;
            features_7[46] = 1.4750749887406807;
            features_7[54] = 0.9028985536152319;
            features_7[74] = 1.504974343097001;
            features_7[92] = 0.9089026148123164;
            features_7[95] = 1.2749374770851736;
            features_7[98] = 1.234927191914445;
            features_7[110] = 1.5009990007593412;

            svm.predict_value(&mut problem_0)?;
            svm.predict_value(&mut problem_7)?;

            assert_eq!(problem_0.result(), SVMResult::Label($libsvm[0]), "predict_value(problem_0)");
            assert_eq!(problem_7.result(), SVMResult::Label($libsvm[1]), "predict_value(problem_7)");

            if $prob {
                svm.predict_probability(&mut problem_0)?;
                svm.predict_probability(&mut problem_7)?;

                assert_eq!(problem_0.result(), SVMResult::Label($libsvm_prob[0]), "predict_probability(problem_0)");
                assert_eq!(problem_7.result(), SVMResult::Label($libsvm_prob[1]), "predict_probability(problem_7)");
            }

            Ok(())
        }
    };
}

#[cfg(test)]
mod svm_sparse_class {
    use ffsvm::{Predict, Problem, SVMError, SVMResult, SparseSVM};
    use std::convert::TryFrom;

    // CSVM

    test_model!(m_csvm_linear_prob, "m_csvm_linear_prob.libsvm", true, [0, 7], [1, 6]);
    test_model!(m_csvm_poly_prob, "m_csvm_poly_prob.libsvm", true, [0, 7], [0, 6]);
    test_model!(m_csvm_rbf_prob, "m_csvm_rbf_prob.libsvm", true, [7, 7], [1, 0]);
    test_model!(m_csvm_sigmoid_prob, "m_csvm_sigmoid_prob.libsvm", true, [0, 7], [7, 1]);

    // Temporarily disabled as they trigger ICE in Rust Nightly
    // test_model!(m_csvm_linear, "m_csvm_linear.libsvm", false, [0, 7], []);
    // test_model!(m_csvm_poly, "m_csvm_poly.libsvm", false, [0, 7], []);
    // test_model!(m_csvm_rbf, "m_csvm_rbf.libsvm", false, [0, 7], []);
    // test_model!(m_csvm_sigmoid, "m_csvm_sigmoid.libsvm", false, [0, 5], []);

    // NUSVM

    // test_model!(m_nusvm_linear_prob, "m_nusvm_linear_prob.libsvm", true, [0, 7], [1, 6]);
    // test_model!(m_nusvm_poly_prob, "m_nusvm_poly_prob.libsvm", true, [0, 7], [0, 7]);
    // test_model!(m_nusvm_rbf_prob, "m_nusvm_rbf_prob.libsvm", true, [0, 7], [0, 7]);
    // test_model!(m_nusvm_sigmoid_prob, "m_nusvm_sigmoid_prob.libsvm", true, [0, 7], [0, 7]);

    // Temporarily disabled as they trigger ICE in Rust Nightly
    // test_model!(m_nusvm_linear, "m_nusvm_linear.libsvm", false, [0, 7], []);
    // test_model!(m_nusvm_poly, "m_nusvm_poly.libsvm", false, [0, 7], []);
    // test_model!(m_nusvm_rbf, "m_nusvm_rbf.libsvm", false, [0, 7], []);
    // test_model!(m_nusvm_sigmoid, "m_nusvm_sigmoid.libsvm", false, [0, 7], []);
}
