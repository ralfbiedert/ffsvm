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
    ) -> impl FnMut() {
        let mut svm = RbfCSVM::random(num_classes, num_sv_per_class, num_attributes);
        let mut problem = Problem::from(&svm).randomize();

        move || (&mut svm).predict_value(&mut problem).expect("This should work")
    }


    #[bench]
    fn csvm_predict_sv128_attr16_problems1(b: &mut Bencher) {
        b.iter(produce_testcase(2, 64, 16));
    }

    #[bench]
    fn csvm_predict_sv1024_attr16_problems1(b: &mut Bencher) {
        b.iter(produce_testcase(2, 512, 16));
    }

    #[bench]
    fn csvm_predict_sv1024_attr1024_problems1(b: &mut Bencher) {
        b.iter(produce_testcase(2, 512, 1024));
    }
}
