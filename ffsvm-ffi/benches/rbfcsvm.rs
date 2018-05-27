#![feature(test)]

extern crate ffsvm;
extern crate ffsvm_ffi;
extern crate test;

mod benchmarks {

    use ffsvm::{Problem, Randomize, RbfCSVM};
    use ffsvm_ffi::{predict_values};
    use test::Bencher;

    /// Produces a test case run for benchmarking
    #[allow(dead_code)]
    fn produce_testcase(
        num_classes: usize,
        num_sv_per_class: usize,
        num_attributes: usize,
        num_problems: usize,
    ) -> impl FnMut() {
        let svm = RbfCSVM::random(num_classes, num_sv_per_class, num_attributes);
        let mut problems = (0 .. num_problems)
            .map(|_| Problem::from(&svm).randomize())
            .collect::<Vec<Problem>>();

        move || predict_values(&svm, &mut problems)
    }

    #[bench]
    fn csvm_predict_sv128_attr16_problems1024(b: &mut Bencher) {
        b.iter(produce_testcase(2, 64, 16, 1024));
    }

    #[bench]
    fn csvm_predict_sv1024_attr16_problems128(b: &mut Bencher) {
        b.iter(produce_testcase(2, 512, 16, 128));
    }
}
