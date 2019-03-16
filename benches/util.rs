use ffsvm::{Attribute, Header, ModelFile, SupportVector};
use rand::Rng;

pub fn random_dense<'b>(svm_type: &'b str, kernel_type: &'b str, total_sv: u32, attr: u32) -> ModelFile<'b> {
    let mut rng = rand::thread_rng();

    ModelFile {
        header: Header {
            svm_type,
            kernel_type,
            total_sv,
            gamma: Some(rng.gen::<f32>()),
            coef0: Some(rng.gen::<f32>()),
            degree: Some(rng.gen_range(1, 10)),
            nr_class: 2,
            rho: vec![rng.gen::<f64>()],
            label: vec![0, 1],
            prob_a: Some(vec![rng.gen::<f64>(), rng.gen::<f64>()]),
            prob_b: Some(vec![rng.gen::<f64>(), rng.gen::<f64>()]),
            nr_sv: vec![total_sv / 2, total_sv / 2],
        },
        vectors: (0 .. total_sv)
            .map(|_| SupportVector {
                coefs: vec![rng.gen::<f32>()],
                features: (0 .. attr)
                    .map(|i| Attribute {
                        index: i,
                        value: rng.gen::<f32>(),
                    })
                    .collect(),
            })
            .collect(),
    }
}
