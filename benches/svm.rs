#![feature(test)]

extern crate ffsvm;
extern crate test;

mod svm {

    use ffsvm::{DenseSVM, KernelDense, Linear, Poly, Predict, Problem, Random, RandomSVM, Randomize, Rbf, SVMType, Sigmoid};
    use test::Bencher;

    /// Produces a test case run for benchmarking
    #[allow(dead_code)]
    fn produce_testcase<K>(svm_type: SVMType, num_classes: usize, num_sv_per_class: usize, num_attributes: usize) -> impl FnMut()
    where
        K: KernelDense + Random + 'static,
    {
        let mut svm = DenseSVM::random::<K>(svm_type, num_classes, num_sv_per_class, num_attributes);
        let mut problem = Problem::from(&svm).randomize();

        move || (&mut svm).predict_value(&mut problem).expect("This should work")
    }

    // RBF

    #[bench]
    fn predict_rbf_sv128_attr16_problems1(b: &mut Bencher) { b.iter(produce_testcase::<Rbf>(SVMType::CSvc, 2, 64, 16)); }

    #[bench]
    fn predict_rbf_sv1024_attr16_problems1(b: &mut Bencher) { b.iter(produce_testcase::<Rbf>(SVMType::CSvc, 2, 512, 16)); }

    #[bench]
    fn predict_rbf_sv1024_attr1024_problems1(b: &mut Bencher) { b.iter(produce_testcase::<Rbf>(SVMType::CSvc, 2, 512, 1024)); }

    // Linear

    #[bench]
    fn predict_linear_sv128_attr16_problems1(b: &mut Bencher) { b.iter(produce_testcase::<Linear>(SVMType::CSvc, 2, 64, 16)); }

    #[bench]
    fn predict_linear_sv1024_attr16_problems1(b: &mut Bencher) { b.iter(produce_testcase::<Linear>(SVMType::CSvc, 2, 512, 16)); }

    #[bench]
    fn predict_linear_sv1024_attr1024_problems1(b: &mut Bencher) { b.iter(produce_testcase::<Linear>(SVMType::CSvc, 2, 512, 1024)); }

    // Poly

    #[bench]
    fn predict_poly_sv128_attr16_problems1(b: &mut Bencher) { b.iter(produce_testcase::<Poly>(SVMType::CSvc, 2, 64, 16)); }

    #[bench]
    fn predict_poly_sv1024_attr16_problems1(b: &mut Bencher) { b.iter(produce_testcase::<Poly>(SVMType::CSvc, 2, 512, 16)); }

    #[bench]
    fn predict_poly_sv1024_attr1024_problems1(b: &mut Bencher) { b.iter(produce_testcase::<Poly>(SVMType::CSvc, 2, 512, 1024)); }

    // Sigmoid

    #[bench]
    fn predict_sigmoid_sv128_attr16_problems1(b: &mut Bencher) { b.iter(produce_testcase::<Sigmoid>(SVMType::CSvc, 2, 64, 16)); }

    #[bench]
    fn predict_sigmoid_sv1024_attr16_problems1(b: &mut Bencher) { b.iter(produce_testcase::<Sigmoid>(SVMType::CSvc, 2, 512, 16)); }

    #[bench]
    fn predict_sigmoid_sv1024_attr1024_problems1(b: &mut Bencher) { b.iter(produce_testcase::<Sigmoid>(SVMType::CSvc, 2, 512, 1024)); }

}
