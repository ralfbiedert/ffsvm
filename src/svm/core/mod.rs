macro_rules! prepare_svm {
    ($raw_model:expr, $k:ty, $m32:ty, $svm:tt) => {
        // To quickly check what broke again during parsing ...
        // println!("{:?}", raw_model);
        {
            let header = &$raw_model.header();
            let vectors = &$raw_model.vectors();

            // Get basic info
            let num_attributes = vectors[0].features.len();
            let num_total_sv = header.total_sv as usize;

            let svm_type = match $raw_model.header().svm_type {
                "c_svc" => SVMType::CSvc,
                "nu_svc" => SVMType::NuSvc,
                "epsilon_svr" => SVMType::ESvr,
                "nu_svr" => SVMType::NuSvr,
                _ => unimplemented!(),
            };

            let kernel: Box<$k> = match $raw_model.header().kernel_type {
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
                SVMType::CSvc | SVMType::NuSvc => (0 .. num_classes)
                    .map(|c| {
                        let label = header.label[c];
                        let num_sv = nr_sv[c] as usize;
                        Class::<$m32>::with_parameters(num_classes, num_sv, num_attributes, label)
                    })
                    .collect::<Vec<Class<$m32>>>(),
                SVMType::ESvr | SVMType::NuSvr => vec![Class::<$m32>::with_parameters(2, num_total_sv, num_attributes, 0)],
            };

            let probabilities = match (&$raw_model.header().prob_a, &$raw_model.header().prob_b) {
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

macro_rules! compute_multiclass_probabilities_impl {
    ($self:tt, $problem:tt) => {{
        let num_classes = $self.classes.len();
        let max_iter = 100.max(num_classes);
        let mut q = $problem.q.flat_mut();
        let qp = &mut $problem.qp;
        let eps = 0.005 / num_classes as f64; // Magic number .005 comes from libSVM.
        let pairwise = $problem.pairwise.flat();
        let probabilities = $problem.probabilities.flat_mut();

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
    }};
}

macro_rules! compute_classification_values_impl {
    ($self:tt, $problem:tt) => {{
        use simd_aligned::traits::Simd;
        set_all(&mut $problem.vote, 0);

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

        for i in 0 .. $self.classes.len() {
            for j in (i + 1) .. $self.classes.len() {
                let sv_coef0 = $self.classes[i].coefficients.row(j - 1);
                let sv_coef1 = $self.classes[j].coefficients.row(i);

                let kvalues0 = $problem.kernel_values.row(i);
                let kvalues1 = $problem.kernel_values.row(j);

                let sum0 = sv_coef0.iter().zip(kvalues0).map(|(a, b)| (*a * *b).sum()).sum::<f64>();
                let sum1 = sv_coef1.iter().zip(kvalues1).map(|(a, b)| (*a * *b).sum()).sum::<f64>();

                let sum = sum0 + sum1 - $self.rho[(i, j)];
                let index_to_vote = if sum > 0.0 { i } else { j };

                $problem.decision_values[(i, j)] = sum;
                $problem.vote[index_to_vote] += 1;
            }
        }
    }};
}

macro_rules! predict_probability_impl {
    ($self:tt, $problem:tt) => {{
        match $self.svm_type {
            SVMType::CSvc | SVMType::NuSvc => {
                const MIN_PROB: f64 = 1e-7;

                // Ensure we have probabilities set. If not, somebody used us the wrong way
                if $self.probabilities.is_none() {
                    return Err(Error::NoProbabilities);
                }

                let num_classes = $self.classes.len();
                let probabilities = $self.probabilities.as_ref().unwrap();

                // First we need to predict the problem for our decision values
                $self.predict_value($problem)?;

                let mut pairwise = $problem.pairwise.flat_mut();

                // Now compute probability values
                for i in 0 .. num_classes {
                    for j in i + 1 .. num_classes {
                        let decision_value = $problem.decision_values[(i, j)];
                        let a = probabilities.a[(i, j)];
                        let b = probabilities.b[(i, j)];

                        let sigmoid = sigmoid_predict(decision_value, a, b).max(MIN_PROB).min(1_f64 - MIN_PROB);

                        pairwise[(i, j)] = sigmoid;
                        pairwise[(j, i)] = 1_f64 - sigmoid;
                    }
                }

                let problem_probabilities = $problem.probabilities.flat_mut();

                if num_classes == 2 {
                    problem_probabilities[0] = pairwise[(0, 1)];
                    problem_probabilities[1] = pairwise[(1, 0)];
                } else {
                    $self.compute_multiclass_probabilities($problem)?;
                }

                let max_index = find_max_index($problem.probabilities.flat());
                $problem.result = Label::Class($self.classes[max_index].label);

                Ok(())
            }
            // This fallback behavior is mandated by `libSVM`.
            SVMType::ESvr | SVMType::NuSvr => $self.predict_value($problem),
        }
    }};
}

// We do late include here to capture our macros above ...
pub mod dense;
pub mod sparse;
