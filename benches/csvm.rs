#![feature(test)]

extern crate ffsvm;
extern crate test;

mod benchmarks {

    use ffsvm::{Kernel, Linear, Predict, Problem, Random, RandomSVM, Randomize, Rbf, CSVM};
    use test::Bencher;

    /// Produces a test case run for benchmarking
    #[allow(dead_code)]
    fn produce_testcase<K>(
        num_classes: usize,
        num_sv_per_class: usize,
        num_attributes: usize,
    ) -> impl FnMut()
    where
        K: Kernel + Random + 'static,
    {
        let mut svm = CSVM::random::<K>(num_classes, num_sv_per_class, num_attributes);
        let mut problem = Problem::from(&svm).randomize();

        move || {
            (&mut svm)
                .predict_value(&mut problem)
                .expect("This should work")
        }
    }

    #[bench]
    fn csvm_predict_rbf_sv128_attr16_problems1(b: &mut Bencher) {
        b.iter(produce_testcase::<Rbf>(2, 64, 16));
    }

    #[bench]
    fn csvm_predict_rbf_sv1024_attr16_problems1(b: &mut Bencher) {
        b.iter(produce_testcase::<Rbf>(2, 512, 16));
    }

    #[bench]
    fn csvm_predict_rbf_sv1024_attr1024_problems1(b: &mut Bencher) {
        b.iter(produce_testcase::<Rbf>(2, 512, 1024));
    }

    #[bench]
    fn csvm_predict_linear_sv128_attr16_problems1(b: &mut Bencher) {
        b.iter(produce_testcase::<Linear>(2, 64, 16));
    }

    #[bench]
    fn csvm_predict_linear_sv1024_attr16_problems1(b: &mut Bencher) {
        b.iter(produce_testcase::<Linear>(2, 512, 16));
    }

    #[bench]
    fn csvm_predict_linear_sv1024_attr1024_problems1(b: &mut Bencher) {
        b.iter(produce_testcase::<Linear>(2, 512, 1024));
    }

}
