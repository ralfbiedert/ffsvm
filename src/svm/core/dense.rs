use simd_aligned::{f32s, f64s, RowOptimized, SimdMatrix, SimdVector};
use std::{convert::TryFrom, marker::PhantomData};

use crate::{
    errors::Error,
    parser::ModelFile,
    svm::{
        class::Class,
        kernel::{KernelDense, Linear, Poly, Rbf, Sigmoid},
        predict::Predict,
        problem::{Problem, Solution},
        Probabilities, SVMType,
    },
    util::{find_max_index, set_all, sigmoid_predict},
    vectors::Triangular,
};



/// Generic support vector machine core, used by both SVM types.
///
/// The SVM holds a kernel, class information and all other numerical data read from
/// the model.
///
/// # Creating a SVM
///
/// Models can be constructed like this:
///
/// ```
/// #![feature(try_from)]
///
/// use ffsvm::*;
/// use std::convert::TryFrom;
///
/// let svm = SparseSVM::try_from("...");
/// ```
///
pub struct DenseSVM {
    /// Total number of support vectors
    pub(crate) num_total_sv: usize,
    
    /// Number of attributes per support vector
    pub(crate) num_attributes: usize,
    
    pub(crate) rho: Triangular<f64>,
    
    pub(crate) probabilities: Option<Probabilities>,
    
    pub(crate) svm_type: SVMType,
    
    /// SVM specific data needed for classification
    pub(crate) kernel: Box<dyn KernelDense>,
    
    /// All classes
    pub(crate) classes: Vec<Class<SimdMatrix<f32s, RowOptimized>>>,
}


impl DenseSVM {
    
    /// Finds the class index for a given label.
    ///
    /// # Description
    ///
    /// This method takes a `label` as defined in the libSVM training model
    /// and returns the internal `index` where this label resides. The index
    /// equals [Problem::probabilities] index where that label's
    /// probability can be found.
    ///
    /// # Returns
    ///
    /// If the label was found its index returned in the [Option]. Otherwise `None`
    /// is returned.
    ///
    pub fn class_index_for_label(&self, label: i32) -> Option<usize> {
        for (i, class) in self.classes.iter().enumerate() {
            if class.label != label {
                continue;
            }
            
            return Some(i);
        }
        
        None
    }
    
    /// Returns the class label for a given index.
    ///
    /// # Description
    ///
    /// The inverse of [SVMCore::class_index_for_label], this function returns the class label
    /// associated with a certain internal index. The index equals the [Problem]'s
    /// `.probabilities` index where a label's probability can be found.
    ///
    /// # Returns
    ///
    /// If the index was found it is returned in the [Option]. Otherwise `None`
    /// is returned.
    pub fn class_label_for_index(&self, index: usize) -> Option<i32> {
        if index >= self.classes.len() {
            None
        } else {
            Some(self.classes[index].label)
        }
    }
    
    /// Computes the kernel values for this problem
    pub(crate) fn compute_kernel_values(&self, problem: &mut Problem<SimdVector<f32s>>) {
        // Get current problem and decision values array
        let features = &problem.features;
        let kernel_values = &mut problem.kernel_values;
        
        // Compute kernel values per class
        for (i, class) in self.classes.iter().enumerate() {
            let kvalues = kernel_values.row_as_flat_mut(i);
            
            self.kernel.compute(&class.support_vectors, features.as_raw(), kvalues);
        }
    }
    
    
    // This is pretty much copy-paste of `multiclass_probability` from libSVM which we need
    // to be compatibly for predicting probability for multiclass SVMs. The method is in turn
    // based on Method 2 from the paper "Probability Estimates for Multi-class
    // Classification by Pairwise Coupling", Journal of Machine Learning Research 5 (2004) 975-1005,
    // by Ting-Fan Wu, Chih-Jen Lin and Ruby C. Weng.
    pub(crate) fn compute_multiclass_probabilities(&self, problem: &mut Problem<SimdVector<f32s>>) -> Result<(), Error> {
        let num_classes = self.classes.len();
        let max_iter = 100.max(num_classes);
        let mut q = problem.q.flat_mut();
        let qp = &mut problem.qp;
        let eps = 0.005 / num_classes as f64; // Magic number .005 comes from libSVM.
        let pairwise = problem.pairwise.flat();
        let probabilities = problem.probabilities.flat_mut();
        
        // We first build up matrix Q as defined in (14) in the paper above. Q should have
        // the property of being a transition matrix for a Markov Chain.
        for t in 0 .. num_classes {
            probabilities[t] = 1.0 / num_classes as f64;
            
            q[(t, t)] = 0.0;
            
            for j in 0 .. t {
                q[(t, t)] += pairwise[(j, t)] * pairwise[(j, t)];
                q[(t, j)] = q[(j, t)];
            }
            
            for j in t + 1 .. num_classes {
                q[(t, t)] += pairwise[(j, t)] * pairwise[(j, t)];
                q[(t, j)] = -pairwise[(j, t)] * pairwise[(t, j)];
            }
        }
        
        // We now try to satisfy (21), (23) and (24) in the paper above.
        for i in 0 ..= max_iter {
            let mut pqp = 0.0;
            
            for t in 0 .. num_classes {
                qp[t] = 0.0;
                
                for j in 0 .. num_classes {
                    qp[t] += q[(t, j)] * probabilities[j];
                }
                
                pqp += probabilities[t] * qp[t];
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
                return Err(Error::IterationsExceeded);
            }
            
            // This seems to be the main function performing (23) and (24).
            for t in 0 .. num_classes {
                let diff = (-qp[t] + pqp) / q[(t, t)];
                
                probabilities[t] += diff;
                pqp = (pqp + diff * (diff * q[(t, t)] + 2.0 * qp[t])) / (1.0 + diff) / (1.0 + diff);
                
                for j in 0 .. num_classes {
                    qp[j] = (qp[j] + diff * q[(t, j)]) / (1.0 + diff);
                    probabilities[j] /= 1.0 + diff;
                }
            }
        }
        
        Ok(())
    }
    
    /// Based on kernel values, computes the decision values for this problem.
    pub(crate) fn compute_classification_values(&self, problem: &mut Problem<SimdVector<f32s>>) {
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
                let sv_coef0 = self.classes[i].coefficients.row(j - 1);
                let sv_coef1 = self.classes[j].coefficients.row(i);
                
                let kvalues0 = problem.kernel_values.row(i);
                let kvalues1 = problem.kernel_values.row(j);
                
                let sum0 = sv_coef0.iter().zip(kvalues0).map(|(a, b)| (*a * *b).sum()).sum::<f64>();
                let sum1 = sv_coef1.iter().zip(kvalues1).map(|(a, b)| (*a * *b).sum()).sum::<f64>();
                
                let sum = sum0 + sum1 - self.rho[(i, j)];
                let index_to_vote = if sum > 0.0 { i } else { j };
                
                problem.decision_values[(i, j)] = sum;
                problem.vote[index_to_vote] += 1;
            }
        }
    }
    
    /// Based on kernel values, computes the decision values for this problem.
    pub(crate) fn compute_regression_values(&self, problem: &mut Problem<SimdVector<f32s>>) {
        let class = &self.classes[0];
        let coef = class.coefficients.row(0);
        let kvalues = problem.kernel_values.row(0);
        
        let mut sum = coef.iter().zip(kvalues).map(|(a, b)| (*a * *b).sum()).sum::<f64>();
        
        sum -= self.rho[0];
        
        problem.result = Solution::Value(sum as f32);
    }
    
    
    /// Returns number of attributes, reflecting the libSVM model.
    pub fn attributes(&self) -> usize {
        self.num_attributes
    }
    
    /// Returns number of classes, reflecting the libSVM model.
    pub fn classes(&self) -> usize {
        self.classes.len()
    }
}

impl Predict<SimdVector<f32s>, SimdVector<f64s>> for DenseSVM {
    fn predict_probability(&self, problem: &mut Problem<SimdVector<f32s>>) -> Result<(), Error> {
        match self.svm_type {
            SVMType::CSvc | SVMType::NuSvc => {
                const MIN_PROB: f64 = 1e-7;
                
                // Ensure we have probabilities set. If not, somebody used us the wrong way
                if self.probabilities.is_none() {
                    return Err(Error::NoProbabilities);
                }
                
                let num_classes = self.classes.len();
                let probabilities = self.probabilities.as_ref().unwrap();
                
                // First we need to predict the problem for our decision values
                self.predict_value(problem)?;
                
                let mut pairwise = problem.pairwise.flat_mut();
                
                // Now compute probability values
                for i in 0 .. num_classes {
                    for j in i + 1 .. num_classes {
                        let decision_value = problem.decision_values[(i, j)];
                        let a = probabilities.a[(i, j)];
                        let b = probabilities.b[(i, j)];
                        
                        let sigmoid = sigmoid_predict(decision_value, a, b).max(MIN_PROB).min(1f64 - MIN_PROB);
                        
                        pairwise[(i, j)] = sigmoid;
                        pairwise[(j, i)] = 1f64 - sigmoid;
                    }
                }
                
                let problem_probabilities = problem.probabilities.flat_mut();
                
                if num_classes == 2 {
                    problem_probabilities[0] = pairwise[(0, 1)];
                    problem_probabilities[1] = pairwise[(1, 0)];
                } else {
                    self.compute_multiclass_probabilities(problem)?;
                }
                
                let max_index = find_max_index(problem.probabilities.flat());
                problem.result = Solution::Label(self.classes[max_index].label);
                
                Ok(())
            }
            // This fallback behavior is mandated by `libSVM`.
            SVMType::ESvr | SVMType::NuSvr => self.predict_value(problem),
        }
    }
    
    
    // Predict the value for one problem.
    fn predict_value(&self, problem: &mut Problem<SimdVector<f32s>>) -> Result<(), Error> {
        match self.svm_type {
            SVMType::CSvc | SVMType::NuSvc => {
                // Compute kernel, decision values and eventually the label
                self.compute_kernel_values(problem);
                self.compute_classification_values(problem);
                
                // Compute highest vote
                let highest_vote = find_max_index(&problem.vote);
                problem.result = Solution::Label(self.classes[highest_vote].label);
                
                Ok(())
            }
            SVMType::ESvr | SVMType::NuSvr => {
                self.compute_kernel_values(problem);
                self.compute_regression_values(problem);
                Ok(())
            }
        }
    }
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
        let (mut svm, nr_sv) = prepare_svm!(raw_model, dyn KernelDense, SimdMatrix<f32s, RowOptimized>, DenseSVM);

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
