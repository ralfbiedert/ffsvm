#![feature(test)]

extern crate ffsvm;
extern crate test;

mod benchmarks {

    use ffsvm::{PredictProblem, Problem, Randomize, RbfCSVM};
    use test::Bencher;

    /// Produces a test case run for benchmarking
    #[allow(dead_code)]
    fn produce_testcase(
        num_classes: usize,
        num_sv_per_class: usize,
        num_attributes: usize,
        num_problems: usize,
    ) -> impl FnMut() {
        let mut svm = RbfCSVM::random(num_classes, num_sv_per_class, num_attributes);
        let mut problems = (0 .. num_problems)
            .map(|_| Problem::from(&svm).randomize())
            .collect::<Vec<Problem>>();

        move || (&mut svm).predict_values(&mut problems)
    }

    #[bench]
    fn csvm_predict_aaa_t(b: &mut Bencher) { b.iter(produce_testcase(2, 1500, 8, 30)); }

    #[bench]
    fn csvm_predict_sv128_attr16_problems1(b: &mut Bencher) {
        b.iter(produce_testcase(2, 64, 16, 1));
    }

    #[bench]
    fn csvm_predict_sv1024_attr16_problems1(b: &mut Bencher) {
        b.iter(produce_testcase(2, 512, 16, 1));
    }

    #[bench]
    fn csvm_predict_sv128_attr16_problems1024(b: &mut Bencher) {
        b.iter(produce_testcase(2, 64, 16, 1024));
    }

    #[bench]
    fn csvm_predict_sv1024_attr16_problems128(b: &mut Bencher) {
        b.iter(produce_testcase(2, 512, 16, 128));
    }

    #[bench]
    fn csvm_predict_sv1024_attr1024_problems1(b: &mut Bencher) {
        b.iter(produce_testcase(2, 512, 1024, 1));
    }
}
