use simd_aligned::{f32s, f64s, RowOptimized, SimdMatrix, SimdVector};
use std::{convert::TryFrom, marker::PhantomData};

use crate::{
    errors::Error,
    parser::ModelFile,
    svm::{
        class::Class,
        core::SVMCore,
        kernel::{KernelDense, Linear, Poly, Rbf, Sigmoid},
        predict::Predict,
        problem::{Problem, Solution},
        DenseSVM, Probabilities, SVMType,
    },
    util::{find_max_index, set_all, sigmoid_predict},
    vectors::Triangular,
};

impl DenseSVM {
    impl_common_svm!(SimdVector<f32s>);
}

impl Predict<SimdVector<f32s>, SimdVector<f64s>> for DenseSVM {
    impl_common_predict!(SimdVector<f32s>);
}

impl<'a, 'b> TryFrom<&'a str> for DenseSVM {
    type Error = Error;

    fn try_from(input: &'a str) -> Result<DenseSVM, Error> {
        let raw_model = ModelFile::try_from(input)?;
        Self::try_from(&raw_model)
    }
}

impl<'a, 'b> TryFrom<&'a ModelFile<'b>> for DenseSVM {
    type Error = Error;

    fn try_from(raw_model: &'a ModelFile<'_>) -> Result<DenseSVM, Error> {
        let (mut svm, nr_sv) = prepare_svm!(raw_model, dyn KernelDense, SimdMatrix<f32s, RowOptimized>);

        let vectors = &raw_model.vectors;

        // Things down here are a bit ugly as the file format is a bit ugly ...
        // Now read all vectors and decode stored information
        let mut start_offset = 0;

        // In the raw file, support vectors are grouped by class
        for (i, num_sv_per_class) in nr_sv.iter().enumerate() {
            let stop_offset = start_offset + *num_sv_per_class as usize;

            // Set support vector and coefficients
            for (i_vector, vector) in vectors[start_offset .. stop_offset].iter().enumerate() {
                let mut last_attribute = None;

                // Set support vectors
                for (i_attribute, attribute) in vector.features.iter().enumerate() {
                    if let Some(last) = last_attribute {
                        // In case we have seen an attribute already, this one must be strictly
                        // the successor attribute
                        if attribute.index != last + 1 {
                            return Result::Err(Error::AttributesUnordered {
                                index: attribute.index,
                                value: attribute.value,
                                last_index: last,
                            });
                        }
                    };

                    let mut support_vectors = svm.classes[i].support_vectors.flat_mut();
                    support_vectors[(i_vector, i_attribute)] = attribute.value;

                    last_attribute = Some(attribute.index);
                }

                // Set coefficients
                for (i_coefficient, coefficient) in vector.coefs.iter().enumerate() {
                    let mut coefficients = svm.classes[i].coefficients.flat_mut();
                    coefficients[(i_coefficient, i_vector)] = f64::from(*coefficient);
                }
            }

            // Update last offset.
            start_offset = stop_offset;
        }

        // Return what we have
        Result::Ok(svm)
    }
}

#[cfg(test)]
mod tests {
    use crate::*;
    use std::convert::TryFrom;

    #[test]
    fn class_operations() -> Result<(), Error> {
        let svm = DenseSVM::try_from(SAMPLE_MODEL)?;

        assert_eq!(None, svm.class_index_for_label(0));
        assert_eq!(Some(1), svm.class_index_for_label(42));

        Ok(())
    }

}
