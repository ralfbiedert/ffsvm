use std::{
    convert::TryFrom,
    str::{self, FromStr},
};

use pest::Parser;
use pest_derive::Parser;

use crate::errors::SVMError;

/// (Start here) Parsing result of a model file used to instantiate a [SVM].
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
#[derive(Clone, Debug, Default)]
pub struct ModelFile<'a> {
    crate header: Header<'a>,
    crate vectors: Vec<SupportVector>,
}

#[derive(Clone, Debug, Default)]
pub struct Header<'a> {
    crate svm_type: &'a str,
    crate kernel_type: &'a str,
    crate gamma: Option<f32>,
    crate coef0: Option<f32>,
    crate degree: Option<u32>,
    crate nr_class: u32,
    crate total_sv: u32,
    crate rho: Vec<f64>,
    crate label: Vec<u32>,
    crate prob_a: Option<Vec<f64>>,
    crate prob_b: Option<Vec<f64>>,
    crate nr_sv: Vec<u32>,
}

#[derive(Copy, Clone, Debug, Default)]
crate struct Attribute {
    crate index: u32,
    crate value: f32,
}

#[derive(Clone, Debug, Default)]
crate struct SupportVector {
    pub coefs: Vec<f32>,
    pub features: Vec<Attribute>,
}

// Hack to make `pest` re-generate parser every time file changes.
#[cfg(debug_assertions)]
const _GRAMMAR: &str = include_str!("model.pest");

#[derive(Parser)]
#[grammar = "parser/model.pest"]
struct LibSVMModel;

// We keep this here just in case I have to touch the parser ever again ...
#[allow(non_snake_case)]
fn JUST_FUCKING_DEBUG_IT<T>(t: T) -> T
where
    T: std::fmt::Debug,
{
    // println!("{:?}", t);
    t
}

macro_rules! next {
    ($p:expr,str) => {
        $p.next()?.as_str()
    };
    ($p:expr, $t:ty) => {
        JUST_FUCKING_DEBUG_IT($p.next()?.as_str()).parse::<$t>()?
    };
}

macro_rules! convert {
    ($p:expr, $t:ty) => {
        JUST_FUCKING_DEBUG_IT($p.as_str()).parse::<$t>()?
    };
}

impl<'a> TryFrom<&'a str> for ModelFile<'a> {
    type Error = SVMError;

    /// Parses a string into a SVM model
    fn try_from(input: &str) -> Result<ModelFile<'_>, SVMError> {
        let parsed = LibSVMModel::parse(Rule::file, input)?.next()?;

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

        for line in parsed.into_inner() {
            match line.as_rule() {
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
                Rule::line_multiple => {
                    let mut line_pairs = line.into_inner();
                    match next!(line_pairs, str) {
                        "svm_type" => svm_type = Some(next!(line_pairs, str)),
                        "kernel_type" => kernel_type = Some(next!(line_pairs, str)),
                        "gamma" => gamma = Some(next!(line_pairs, f32)),
                        "coef0" => coef0 = Some(next!(line_pairs, f32)),
                        "degree" => degree = Some(next!(line_pairs, u32)),
                        "nr_class" => nr_class = Some(next!(line_pairs, u32)),
                        "total_sv" => total_sv = Some(next!(line_pairs, u32)),
                        "rho" => while let Some(x) = line_pairs.next() {
                            rho.push(convert!(x, f64))
                        },
                        "label" => while let Some(x) = line_pairs.next() {
                            label.push(convert!(x, u32))
                        },
                        "nr_sv" => while let Some(x) = line_pairs.next() {
                            nr_sv.push(convert!(x, u32))
                        },
                        "probA" => {
                            let mut v = Vec::<f64>::new();
                            while let Some(x) = line_pairs.next() {
                                v.push(convert!(x, f64))
                            }
                            prob_a = Option::Some(v);
                        }
                        "probB" => {
                            let mut v = Vec::<f64>::new();
                            while let Some(x) = line_pairs.next() {
                                v.push(convert!(x, f64))
                            }
                            prob_b = Option::Some(v);
                        }
                        "SV" => (),
                        unknown => panic!("Unknown header `{}`!", unknown),
                    };
                }

                // 0.0625 0:0.6619648 1:0.8464851 2:0.4801146 3:0 4:0 5:0.02131653 6:0 7:0 8:0 9:0 10:0 11:0 12:0 13:0 14:0 15:0.5579834 16:0.1106567 17:0 18:0 19:0 20:0
                // 0.0625 0:0.5861949 1:0.5556895 2:0.619291 3:0 4:0 5:0 6:0 7:0 8:0 9:0 10:0 11:0.5977631 12:0 13:0 14:0 15:0.6203156 16:0 17:0 18:0 19:0.1964417 20:0
                // 0.0625 0:0.44675 1:0.4914977 2:0.4227562 3:0.2904663 4:0.2904663 5:0.268158 6:0 7:0 8:0 9:0 10:0 11:0.6202393 12:0.0224762 13:0 14:0 15:0.6427917 16:0.0224762 17:0 18:0 19:0.1739655 20:0
                Rule::line_sv => {
                    let line_pairs = line.into_inner();

                    let mut sv = SupportVector {
                        coefs: Vec::new(),
                        features: Vec::new(),
                    };

                    for element in line_pairs {
                        match element.as_rule() {
                            Rule::sv => {
                                let mut sv_pairs = element.into_inner();
                                let index = next!(sv_pairs, u32);
                                let value = next!(sv_pairs, f32);

                                sv.features.push(Attribute { index, value })
                            }
                            Rule::number => sv.coefs.push(convert!(element, f32)),
                            _ => unreachable!(),
                        }
                    }

                    vectors.push(sv);
                }
                _ => unreachable!(),
            };
        }

        Ok(ModelFile {
            header: Header {
                svm_type: svm_type?,
                kernel_type: kernel_type?,
                gamma,
                coef0,
                degree,
                nr_class: nr_class?,
                total_sv: total_sv?,
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
