use std::{convert::TryFrom, marker::Sync};

use faster::{f64s, IntoSIMDRefIterator, IntoSIMDZip, SIMDIterator, SIMDZippedIterator, Sum};

use super::SVMError;
use kernel::Kernel;
use parser::ModelFile;
use random::{Random, Randomize};
use svm::{
    problem::Problem, Class, PredictProblem, Probabilities,
    SVMError::{MaxIterationsExceededPredictingProbabilities, ModelDoesNotSupportProbabilities},
    SVM,
};
use util::{find_max_index, set_all, sigmoid_predict};
use vectors::Triangular;

impl<Knl> SVM<Knl>
where
    Knl: Kernel + Random,
{
    /// Creates a new random CSVM
    pub fn random(num_classes: usize, num_sv_per_class: usize, num_attributes: usize) -> Self {
        let num_total_sv = num_classes * num_sv_per_class;
        let classes = (0 .. num_classes)
            .map(|class| {
                Class::with_parameters(num_classes, num_sv_per_class, num_attributes, class as u32)
                    .randomize()
            })
            .collect::<Vec<Class>>();

        SVM {
            num_total_sv,
            num_attributes,
            rho: Triangular::with_dimension(num_classes, Default::default()),
            kernel: Knl::new_random(),
            probabilities: None,
            classes,
        }
    }

    /// Computes the kernel values for this problem
    fn compute_kernel_values(&self, problem: &mut Problem) {
        // Get current problem and decision values array
        let problem_features = &problem.features[..];

        // Compute kernel values per class
        for (i, class) in self.classes.iter().enumerate() {
            let kvalues = &mut problem.kernel_values[i];

            self.kernel
                .compute(&class.support_vectors, problem_features, kvalues);
        }
    }

    // This is pretty much copy-paste of `multiclass_probability` from libSVM which we need
    // to be compatibly for predicting probability for multiclass SVMs. The method is in turn
    // based on Method 2 from the paper "Probability Estimates for Multi-class
    // Classification by Pairwise Coupling", Journal of Machine Learning Research 5 (2004) 975-1005,
    // by Ting-Fan Wu, Chih-Jen Lin and Ruby C. Weng.
    fn compute_multiclass_probabilities(&self, problem: &mut Problem) -> Result<(), SVMError> {
        let num_classes = self.classes.len();
        let max_iter = 100.max(num_classes);
        let q = &mut problem.q;
        let qp = &mut problem.qp;
        let eps = 0.005 / num_classes as f64; // Magic number .005 comes from libSVM.

        // We first build up matrix Q as defined in (14) in the paper above. Q should have
        // the property of being a transition matrix for a Markov Chain.
        for t in 0 .. num_classes {
            problem.probabilities[t] = 1.0 / num_classes as f64;

            q[(t, t)] = 0.0;

            for j in 0 .. t {
                q[(t, t)] += problem.pairwise[(j, t)] * problem.pairwise[(j, t)];
                q[(t, j)] = q[(j, t)];
            }

            for j in t + 1 .. num_classes {
                q[(t, t)] += problem.pairwise[(j, t)] * problem.pairwise[(j, t)];
                q[(t, j)] = -problem.pairwise[(j, t)] * problem.pairwise[(t, j)];
            }
        }

        // We now try to satisfy (21), (23) and (24) in the paper above.
        for i in 0 ..= max_iter {
            let mut pqp = 0.0;

            for t in 0 .. num_classes {
                qp[t] = 0.0;

                for j in 0 .. num_classes {
                    qp[t] += q[(t, j)] * problem.probabilities[j];
                }

                pqp += problem.probabilities[t] * qp[t];
            }

            // Check if we fulfilled our abort criteria, which seems to be related
            // to (21).
            let mut max_error = 0.0;

            for item in qp.iter() {
                let error = (*item - pqp).abs();

                if error > max_error {
                    max_error = error;
                }
            }

            if max_error < eps {
                break;
            }

            // In case we are on the last iteration round past the threshold
            // we know something went wrong. Signal we exceeded the threshold.
            if i == max_iter {
                return Err(MaxIterationsExceededPredictingProbabilities);
            }

            // This seems to be the main function performing (23) and (24).
            for t in 0 .. num_classes {
                let diff = (-qp[t] + pqp) / q[(t, t)];

                problem.probabilities[t] += diff;
                pqp = (pqp + diff * (diff * q[(t, t)] + 2.0 * qp[t])) / (1.0 + diff) / (1.0 + diff);

                for j in 0 .. num_classes {
                    qp[j] = (qp[j] + diff * q[(t, j)]) / (1.0 + diff);
                    problem.probabilities[j] /= 1.0 + diff;
                }
            }
        }

        Ok(())
    }

    /// Based on kernel values, computes the decision values for this problem.
    fn compute_decision_values(&self, problem: &mut Problem) {
        // Reset all votes
        set_all(&mut problem.vote, 0);

        // Since classification is symmetric, if we have N classes, we only need to go through
        // (N * N - 1) - 1 cases. For example for 4 classes we do:
        //
        //          j --->
        //          0    1   2   3
        //    i 0        x   x   x
        //    | 1            x   x
        //    v 2                x
        //      3
        //
        // For each valid combination (e.g., i:1, j:2), we then need to compute
        // the decision values, which consists of two parts:
        //
        // a) The coefficients of class(1) related to class(2) and
        // b) The coefficients of class(2) related to class(1).
        //
        // Both a) and b) are multiplied with the computed kernel values and summed,
        // and eventually used to compute on which side we are.

        for i in 0 .. self.classes.len() {
            for j in (i + 1) .. self.classes.len() {
                let sv_coef0 = &self.classes[i].coefficients[j - 1];
                let sv_coef1 = &self.classes[j].coefficients[i];

                // For `faster` we have to limit the length of our kvalues slice to the length of
                // our (shorter) sv_coef slice.
                let kvalues0 = &problem.kernel_values[i][0 .. sv_coef0.len()];
                let kvalues1 = &problem.kernel_values[j][0 .. sv_coef1.len()];

                let sum0: f64 = (
                    sv_coef0.simd_iter(f64s(0.0f64)),
                    kvalues0.simd_iter(f64s(0.0f64)),
                ).zip()
                    .simd_map(|(a, b)| a * b)
                    .simd_reduce(f64s::splat(0.0), |a, v| a + v)
                    .sum();

                let sum1: f64 = (
                    sv_coef1.simd_iter(f64s(0.0f64)),
                    kvalues1.simd_iter(f64s(0.0f64)),
                ).zip()
                    .simd_map(|(a, b)| a * b)
                    .simd_reduce(f64s::splat(0.0), |a, v| a + v)
                    .sum();

                let sum = sum0 + sum1 - self.rho[(i, j)];
                let index_to_vote = if sum > 0.0 { i } else { j };

                problem.decision_values[(i, j)] = sum;
                problem.vote[index_to_vote] += 1;
            }
        }
    }
}

impl<'a, 'b, Knl> TryFrom<&'a ModelFile<'b>> for SVM<Knl>
where
    Knl: Kernel + From<&'a ModelFile<'b>>,
{
    type Error = SVMError;

    /// Creates a SVM from the given raw model.
    fn try_from(raw_model: &'a ModelFile<'b>) -> Result<SVM<Knl>, SVMError> {
        let header = &raw_model.header;
        let vectors = &raw_model.vectors;

        // Get basic info
        let num_attributes = vectors[0].features.len();
        let num_classes = header.nr_class as usize;
        let num_total_sv = header.total_sv as usize;

        // Construct vector of classes
        let classes = (0 .. num_classes)
            .map(|class| {
                let label = header.label[class];
                let num_sv = header.nr_sv[class] as usize;
                Class::with_parameters(num_classes, num_sv, num_attributes, label)
            })
            .collect::<Vec<Class>>();

        let probabilities = match (&raw_model.header.prob_a, &raw_model.header.prob_b) {
            (&Some(ref a), &Some(ref b)) => Some(Probabilities {
                a: Triangular::from(a),
                b: Triangular::from(b),
            }),

            (_, _) => None,
        };

        // Allocate model
        let mut svm = SVM {
            num_total_sv,
            num_attributes,
            probabilities,
            kernel: Knl::from(raw_model),
            rho: Triangular::from(&header.rho),
            classes,
        };

        // Things down here are a bit ugly as the file format is a bit ugly ...

        // Now read all vectors and decode stored information
        let mut start_offset = 0;

        // In the raw file, support vectors are grouped by class
        for i in 0 .. num_classes {
            let num_sv_per_class = &header.nr_sv[i];
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
                            return Result::Err(SVMError::SvmAttributesUnordered {
                                index: attribute.index,
                                value: attribute.value,
                                last_index: last,
                            });
                        }
                    }

                    svm.classes[i].support_vectors[(i_vector, i_attribute)] = attribute.value;

                    last_attribute = Some(attribute.index);
                }

                // Set coefficients
                for (i_coefficient, coefficient) in vector.coefs.iter().enumerate() {
                    svm.classes[i].coefficients[(i_coefficient, i_vector)] =
                        f64::from(*coefficient);
                }
            }

            // Update last offset.
            start_offset = stop_offset;
        }

        // Return what we have
        Result::Ok(svm)
    }
}

impl<Knl> PredictProblem for SVM<Knl>
where
    Knl: Kernel + Random + Sync,
{
    fn predict_probability(&self, problem: &mut Problem) -> Result<(), SVMError> {
        const MIN_PROB: f64 = 1e-7;

        // Ensure we have probabilities set. If not, somebody used us the wrong way
        if self.probabilities.is_none() {
            return Err(ModelDoesNotSupportProbabilities);
        }

        let num_classes = self.classes.len();
        let probabilities = self.probabilities.as_ref().unwrap();

        // First we need to predict the problem for our decision values
        self.predict_value(problem)?;

        // Now compute probability values
        for i in 0 .. num_classes {
            for j in i + 1 .. num_classes {
                let decision_value = problem.decision_values[(i, j)];
                let a = probabilities.a[(i, j)];
                let b = probabilities.b[(i, j)];

                let sigmoid = sigmoid_predict(decision_value, a, b)
                    .max(MIN_PROB)
                    .min(1f64 - MIN_PROB);

                problem.pairwise[(i, j)] = sigmoid;
                problem.pairwise[(j, i)] = 1f64 - sigmoid;
            }
        }

        if num_classes == 2 {
            problem.probabilities[0] = problem.pairwise[(0, 1)];
            problem.probabilities[1] = problem.pairwise[(1, 0)];
        } else {
            self.compute_multiclass_probabilities(problem)?;
        }

        let max_index = find_max_index(problem.probabilities.as_slice());
        problem.label = self.classes[max_index].label;

        Ok(())
    }

    // Predict the value for one problem.
    fn predict_value(&self, problem: &mut Problem) -> Result<(), SVMError> {
        // Compute kernel, decision values and eventually the label
        self.compute_kernel_values(problem);
        self.compute_decision_values(problem);

        // Compute highest vote
        let highest_vote = find_max_index(&problem.vote);
        problem.label = self.classes[highest_vote].label;

        Ok(())
    }
}
