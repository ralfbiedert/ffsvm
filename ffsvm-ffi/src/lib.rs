
pub mod ffi;

/// Predicts all values for a set of problems.
fn predict_values(&self, problems: &mut [Problem]) {
    // Compute all problems ...
    problems
        .par_iter_mut()
        .for_each(|problem| self.predict_value(problem));
}

/// Predicts all probabilities for a set of problems.
fn predict_probabilities(&self, problems: &mut [Problem]) {
    // Compute all problems ...
    problems
        .par_iter_mut()
        .for_each(|problem| self.predict_probability(problem));
}