
/// Parsing result of a model file used to instantiate a [SVM].
///
/// # Obtaining a model
/// A model file is produced by [libSVM](https://github.com/cjlin1/libsvm). For details
/// how to produce a model see the top-level [FFSVM](index.html#creating-a-libsvm-model)
/// documentation.
///
/// # Loading a model
///
/// Model are generally produced by parsing a `&str` using the `ModelFile::try_from` function:
///
/// ```ignore
/// let model = ModelFile::try_from(model_str)!
/// ```
///
/// Should anything be wrong with the model format, a [ModelError] will be returned. Once you have
/// your model, you can use it to create a [SVM] (in particular an [RbfSVM]).
///
/// # Model format
///
/// For FFSVM to load a model, it needs to approximately look like below. Note that you cannot
/// reasonably create this model by hand, it needs to come from [libSVM](https://github.com/cjlin1/libsvm).
///
/// ```ignore
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
/// In particular:
///
/// * `svm_type` must be `c_svc`.
/// * `kernel_type` must be `rbf` or `linear`
/// * All support vectors (past the `SV` line) must have **strictly** increasing attribute
/// identifiers, without skipping an attribute.
///
#[doc(hidden)]
#[derive(Clone, Debug, Default)]
pub struct ModelFile<'a> {
    pub header: Header<'a>,
    pub vectors: Vec<SupportVector>,
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
    pub label: Vec<u32>,
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

