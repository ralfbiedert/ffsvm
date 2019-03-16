use crate::{
    svm::{SVMType},
    vectors::Triangular,
};

macro_rules! prepare_svm {
    ($raw_model:expr, $k:ty, $m32:ty, $svm:tt) => {
        // To quickly check what broke again during parsing ...
        // println!("{:?}", raw_model);
        {
            let header = &$raw_model.header;
            let vectors = &$raw_model.vectors;

            // Get basic info
            let num_attributes = vectors[0].features.len();
            let num_total_sv = header.total_sv as usize;

            let svm_type = match $raw_model.header.svm_type {
                "c_svc" => SVMType::CSvc,
                "nu_svc" => SVMType::NuSvc,
                "epsilon_svr" => SVMType::ESvr,
                "nu_svr" => SVMType::NuSvr,
                _ => unimplemented!(),
            };

            let kernel: Box<$k> = match $raw_model.header.kernel_type {
                "rbf" => Box::new(Rbf::try_from($raw_model)?),
                "linear" => Box::new(Linear::from($raw_model)),
                "polynomial" => Box::new(Poly::try_from($raw_model)?),
                "sigmoid" => Box::new(Sigmoid::try_from($raw_model)?),
                _ => unimplemented!(),
            };

            let num_classes = match svm_type {
                SVMType::CSvc | SVMType::NuSvc => header.nr_class as usize,
                // For SVRs we set number of classes to 1, since that resonates better
                // with our internal handling
                SVMType::ESvr | SVMType::NuSvr => 1,
            };

            let nr_sv = match svm_type {
                SVMType::CSvc | SVMType::NuSvc => header.nr_sv.clone(),
                // For SVRs we set number of classes to 1, since that resonates better
                // with our internal handling
                SVMType::ESvr | SVMType::NuSvr => vec![num_total_sv as u32],
            };

            // Construct vector of classes
            let classes = match svm_type {
                // TODO: CLEAN THIS UP ... We can probably unify the logic
                SVMType::CSvc | SVMType::NuSvc => (0..num_classes)
                    .map(|c| {
                        let label = header.label[c];
                        let num_sv = nr_sv[c] as usize;
                        Class::<$m32>::with_parameters(num_classes, num_sv, num_attributes, label)
                    })
                    .collect::<Vec<Class<$m32>>>(),
                SVMType::ESvr | SVMType::NuSvr => vec![Class::<$m32>::with_parameters(
                    2,
                    num_total_sv,
                    num_attributes,
                    0,
                )],
            };

            let probabilities = match (&$raw_model.header.prob_a, &$raw_model.header.prob_b) {
                // Regular case for classification with probabilities
                (&Some(ref a), &Some(ref b)) => Some(Probabilities {
                    a: Triangular::from(a),
                    b: Triangular::from(b),
                }),
                // For SVRs only one probability array is given
                (&Some(ref a), None) => Some(Probabilities {
                    a: Triangular::from(a),
                    b: Triangular::with_dimension(0, 0.0),
                }),
                // Regular case for classification w/o probabilities
                (_, _) => None,
            };

            // Allocate model
            (
                $svm {
                    num_total_sv,
                    num_attributes,
                    probabilities,
                    kernel,
                    svm_type,
                    rho: Triangular::from(&header.rho),
                    classes,
                },
                nr_sv,
            )
        }
    };
}

// We do late include here to capture our macros above ...
pub mod dense;
pub mod sparse;
