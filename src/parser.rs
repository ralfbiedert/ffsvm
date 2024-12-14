use crate::errors::Error;
use std::{convert::TryFrom, str};

/// Parsing result of a model file used to instantiate a [`DenseSVM`](`crate::DenseSVM`) or [`SparseSVM`](`crate::SparseSVM`).
///
/// # Obtaining Models
/// A model file is produced by [libSVM](https://github.com/cjlin1/libsvm). For details
/// how to produce a model see the top-level [FFSVM](index.html#creating-a-libsvm-model)
/// documentation.
///
/// # Loading Models
///
/// Models are generally produced by parsing a [`&str`] using the [`ModelFile::try_from`] function:
///
/// ```rust
/// use ffsvm::ModelFile;
/// # use ffsvm::SAMPLE_MODEL;
///
/// let model_result = ModelFile::try_from(SAMPLE_MODEL);
/// ```
///
/// Should anything be wrong with the model format, an [`Error`] will be returned. Once you have
/// your model, you can use it to create an SVM, for example by invoking `DenseSVM::try_from(model)`.
///
/// # Model Format
///
/// For FFSVM to load a model, it needs to look approximately like below. Note that you cannot
/// reasonably create this model by hand, it needs to come from [libSVM](https://github.com/cjlin1/libsvm).
///
/// ```text
/// svm_type c_svc
/// kernel_type rbf
/// gamma 1
/// nr_class 2
/// total_sv 3012
/// rho -2.90877
/// label 0 1
/// probA -1.55583
/// probB 0.0976659
/// nr_sv 1513 1499
/// SV
/// 256 0:0.5106233 1:0.1584117 2:0.1689098 3:0.1664358 4:0.2327561 5:0 6:0 7:0 8:1 9:0.1989241
/// 256 0:0.5018305 1:0.0945542 2:0.09242307 3:0.09439687 4:0.1398575 5:0 6:0 7:0 8:1 9:1
/// 256 0:0.5020829 1:0 2:0 3:0 4:0.1393665 5:1 6:0 7:0 8:1 9:0
/// 256 0:0.4933203 1:0.1098869 2:0.1048947 3:0.1069601 4:0.2152338 5:0 6:0 7:0 8:1 9:1
/// ```
///
/// Apart from "one-class SVM" (`-s 2` in libSVM) and "precomputed kernel" (`-t 4`) all
/// generated libSVM models should be supported.
///
/// However, note that for the [`DenseSVM`](`crate::DenseSVM`) to work, all support vectors
/// (past the `SV` line) must have **strictly** increasing attribute identifiers starting at `0`,
/// without skipping an attribute. In other words, your attributes have to be named `0:`, `1:`,
/// `2:`, ... `n:` and not, say, `0:`, `1:`, `4:`, ... `n:`.
#[derive(Clone, Debug, Default)]
pub struct ModelFile<'a> {
    header: Header<'a>,
    vectors: Vec<SupportVector>,
}

impl<'a> ModelFile<'a> {
    #[doc(hidden)]
    #[must_use]
    pub const fn new(header: Header<'a>, vectors: Vec<SupportVector>) -> Self { Self { header, vectors } }

    #[doc(hidden)]
    #[must_use]
    pub const fn header(&self) -> &Header { &self.header }

    #[doc(hidden)]
    #[must_use]
    pub fn vectors(&self) -> &[SupportVector] { self.vectors.as_slice() }
}

#[doc(hidden)]
#[derive(Clone, Debug, Default)]
pub struct Header<'a> {
    pub svm_type: &'a str,
    pub kernel_type: &'a str,
    pub gamma: Option<f32>,
    pub coef0: Option<f32>,
    pub degree: Option<u32>,
    pub nr_class: u32,
    pub total_sv: u32,
    pub rho: Vec<f64>,
    pub label: Vec<i32>,
    pub prob_a: Option<Vec<f64>>,
    pub prob_b: Option<Vec<f64>>,
    pub nr_sv: Vec<u32>,
}

#[doc(hidden)]
#[derive(Copy, Clone, Debug, Default)]
pub struct Attribute {
    pub value: f32,
    pub index: u32,
}

#[doc(hidden)]
#[derive(Clone, Debug, Default)]
pub struct SupportVector {
    pub coefs: Vec<f32>,
    pub features: Vec<Attribute>,
}

impl<'a> TryFrom<&'a str> for ModelFile<'a> {
    type Error = Error;

    /// Parses a string into an SVM model
    #[allow(clippy::similar_names)]
    fn try_from(input: &str) -> Result<ModelFile<'_>, Error> {
        let mut svm_type = Option::None;
        let mut kernel_type = Option::None;
        let mut gamma = Option::None;
        let mut coef0 = Option::None;
        let mut degree = Option::None;
        let mut nr_class = Option::None;
        let mut total_sv = Option::None;
        let mut rho = Vec::new();
        let mut label = Vec::new();
        let mut prob_a = Option::None;
        let mut prob_b = Option::None;
        let mut nr_sv = Vec::new();

        let mut vectors = Vec::new();

        for line in input.lines() {
            let tokens = line.split_whitespace().collect::<Vec<_>>();

            match tokens.first() {
                // Single value headers
                //
                // svm_type c_svc
                // kernel_type rbf
                // gamma 0.5
                // nr_class 6
                // total_sv 153
                // rho 2.37333 -0.579888 0.535784 0.0701838 0.609329 -0.932983 -0.427481 -1.15801 -0.108324 0.486988 -0.0642337 0.52711 -0.292071 0.214309 0.880031
                // label 1 2 3 5 6 7
                // probA -1.26241 -2.09056 -3.04781 -2.49489 -2.79378 -2.55612 -1.80921 -1.90492 -2.6911 -2.67778 -2.15836 -2.53895 -2.21813 -2.03491 -1.91923
                // probB 0.135634 0.570051 -0.114691 -0.397667 0.0687938 0.839527 -0.310816 -0.787629 0.0335196 0.15079 -0.389211 0.288416 0.186429 0.46585 0.547398
                // nr_sv 50 56 17 11 7 12
                // SV
                Some(x) if *x == "svm_type" => {
                    svm_type = Some(tokens[1]);
                }
                Some(x) if *x == "kernel_type" => {
                    kernel_type = Some(tokens[1]);
                }
                Some(x) if *x == "gamma" => {
                    gamma = tokens[1].parse::<f32>().ok();
                }
                Some(x) if *x == "coef0" => {
                    coef0 = tokens[1].parse::<f32>().ok();
                }
                Some(x) if *x == "degree" => {
                    degree = tokens[1].parse::<u32>().ok();
                }
                Some(x) if *x == "nr_class" => {
                    nr_class = tokens[1].parse::<u32>().ok();
                }
                Some(x) if *x == "total_sv" => {
                    total_sv = tokens[1].parse::<u32>().ok();
                }
                // Multi value headers
                Some(x) if *x == "rho" => rho = tokens.iter().skip(1).filter_map(|x| x.parse::<f64>().ok()).collect(),
                Some(x) if *x == "label" => label = tokens.iter().skip(1).filter_map(|x| x.parse::<i32>().ok()).collect(),
                Some(x) if *x == "nr_sv" => nr_sv = tokens.iter().skip(1).filter_map(|x| x.parse::<u32>().ok()).collect(),
                Some(x) if *x == "probA" => prob_a = Some(tokens.iter().skip(1).filter_map(|x| x.parse::<f64>().ok()).collect()),
                Some(x) if *x == "probB" => prob_b = Some(tokens.iter().skip(1).filter_map(|x| x.parse::<f64>().ok()).collect()),
                // Header separator
                Some(x) if *x == "SV" => {}
                // These are all regular lines without a clear header (after SV) ...
                //
                // 0.0625 0:0.6619648 1:0.8464851 2:0.4801146 3:0 4:0 5:0.02131653 6:0 7:0 8:0 9:0 10:0 11:0 12:0 13:0 14:0 15:0.5579834 16:0.1106567 17:0 18:0 19:0 20:0
                // 0.0625 0:0.5861949 1:0.5556895 2:0.619291 3:0 4:0 5:0 6:0 7:0 8:0 9:0 10:0 11:0.5977631 12:0 13:0 14:0 15:0.6203156 16:0 17:0 18:0 19:0.1964417 20:0
                // 0.0625 0:0.44675 1:0.4914977 2:0.4227562 3:0.2904663 4:0.2904663 5:0.268158 6:0 7:0 8:0 9:0 10:0 11:0.6202393 12:0.0224762 13:0 14:0 15:0.6427917 16:0.0224762 17:0 18:0 19:0.1739655 20:0
                Some(_) => {
                    let mut sv = SupportVector {
                        coefs: Vec::new(),
                        features: Vec::new(),
                    };

                    let (features, coefs): (Vec<&str>, Vec<&str>) = tokens.iter().partition(|x| x.contains(':'));

                    sv.coefs = coefs.iter().filter_map(|x| x.parse::<f32>().ok()).collect();
                    sv.features = features
                        .iter()
                        .filter_map(|x| {
                            let split = x.split(':').collect::<Vec<&str>>();

                            Some(Attribute {
                                index: split.first()?.parse::<u32>().ok()?,
                                value: split.get(1)?.parse::<f32>().ok()?,
                            })
                        })
                        .collect();

                    vectors.push(sv);
                }

                // Empty end of file
                None => break,
            }
        }

        Ok(ModelFile {
            header: Header {
                svm_type: svm_type.ok_or(Error::MissingRequiredAttribute)?,
                kernel_type: kernel_type.ok_or(Error::MissingRequiredAttribute)?,
                gamma,
                coef0,
                degree,
                nr_class: nr_class.ok_or(Error::MissingRequiredAttribute)?,
                total_sv: total_sv.ok_or(Error::MissingRequiredAttribute)?,
                rho,
                label,
                prob_a,
                prob_b,
                nr_sv,
            },
            vectors,
        })
    }
}
