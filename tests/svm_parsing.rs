#[cfg(test)]
mod svm_performance {
    use ffsvm::{DenseSVM, Error};
    use std::convert::TryFrom;

    #[test]
    fn load_large() -> Result<(), Error> {
        let model = include_str!("data_misc/model_large.libsvm");
        let _svm = DenseSVM::try_from(model)?;

        Ok(())
    }

    #[test]
    fn load_label_negative() -> Result<(), Error> {
        let model = include_str!("data_misc/model_label_negative.libsvm");
        let _svm = DenseSVM::try_from(model)?;

        Ok(())
    }
}
