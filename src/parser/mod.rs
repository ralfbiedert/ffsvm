mod raw;

use std::{convert::TryFrom, str};

pub use self::raw::*;

use pest::Parser;
use pest_derive::Parser;

use crate::errors::SVMError;

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
