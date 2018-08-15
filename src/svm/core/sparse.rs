use crate::sparse::{SparseMatrix, SparseVector};

use std::{convert::TryFrom, marker::PhantomData};

use crate::{
    errors::Error,
    parser::ModelFile,
    svm::{
        class::Class,
        core::SVMCore,
        kernel::{KernelSparse, Linear, Poly, Rbf, Sigmoid},
        predict::Predict,
        problem::{Problem, Solution},
        Probabilities, SVMType, SparseSVM,
    },
    util::{find_max_index, set_all, sigmoid_predict},
    vectors::Triangular,
};

impl SparseSVM {
    impl_common_svm!(SparseVector<f32>);
}

impl Predict<SparseVector<f32>, SparseVector<f64>> for SparseSVM {
    impl_common_predict!(SparseVector<f32>);
}

impl<'a, 'b> TryFrom<&'a str> for SparseSVM {
    type Error = Error;

    fn try_from(input: &'a str) -> Result<SparseSVM, Error> {
        let raw_model = ModelFile::try_from(input)?;
        Self::try_from(&raw_model)
    }
}

impl<'a, 'b> TryFrom<&'a ModelFile<'b>> for SparseSVM {
    type Error = Error;

    fn try_from(raw_model: &'a ModelFile<'_>) -> Result<SparseSVM, Error> {
        let (mut svm, nr_sv) = prepare_svm!(raw_model, dyn KernelSparse, SparseMatrix<f32>);

        let vectors = &raw_model.vectors;

        // Things down here are a bit ugly as the file format is a bit ugly ...
        // Now read all vectors and decode stored information
        let mut start_offset = 0;

        // In the raw file, support vectors are grouped by class
        for (i, num_sv_per_class) in nr_sv.iter().enumerate() {
            let stop_offset = start_offset + *num_sv_per_class as usize;

            // Set support vector and coefficients
            for (i_vector, vector) in vectors[start_offset .. stop_offset].iter().enumerate() {
                // Set support vectors
                for attribute in &vector.features {
                    let support_vectors = &mut svm.classes[i].support_vectors;
                    support_vectors[(i_vector, attribute.index as usize)] = attribute.value;
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
