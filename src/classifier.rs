
use types::{Probability, Feature, ModelCSVM};

pub fn predict_probability_csvm(_csvm: &ModelCSVM, _feature_vector: &[Feature], _probabilities: &mut [Probability]) {

    //    faster SIMD goes here ...
    //    let x = (&feature_vector[..]);
    //    let mut mp = x.simd_iter().map(|vector| { f32s::splat(10.0) + vector.abs() });
    //    let c = mp.scalar_collect();

}
