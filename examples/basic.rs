use ffsvm::*;
use std::convert::TryFrom;

fn main() -> Result<(), Error> {
    let svm = DenseSVM::try_from(SAMPLE_MODEL)?;

    let mut fv = FeatureVector::from(&svm);
    let features = fv.features();

    features[0] = 0.558_382;
    features[1] = -0.157_895;
    features[2] = 0.581_292;
    features[3] = -0.221_184;

    svm.predict_value(&mut fv)?;

    assert_eq!(fv.label(), Label::Class(42));

    Ok(())
}
