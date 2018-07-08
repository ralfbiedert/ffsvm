use nom::{
    is_alphanumeric, line_ending, types::CompleteStr, named, do_parse, 
    tag, call, opt, many0, map_res, error_position, alt, preceded, tuple, 
    tuple_parser, named_args, take_while_s, take_while, count, ws, sep, wrap_sep
};
use std::{
    convert::TryFrom, str::{self, FromStr},
};

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
/// your model, you can use it to create a [SVM] (in particular an [RbfCSVM]).
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
/// * `kernel_type` must be `rbf`
/// * All support vectors (past the `SV` line) must have **strictly** increasing attribute
/// identifiers, without skipping an attribute.  
///
#[derive(Clone, Debug, Default)]
pub struct ModelFile<'a> {
    pub(crate) header: Header<'a>,
    pub(crate) vectors: Vec<SupportVector>,
}

#[derive(Clone, Debug, Default)]
pub struct Header<'a> {
    pub(crate) svm_type: &'a str,
    pub(crate) kernel_type: &'a str,
    pub(crate) gamma: f32,
    pub(crate) nr_class: u32,
    pub(crate) total_sv: u32,
    pub(crate) rho: Vec<f64>,
    pub(crate) label: Vec<u32>,
    pub(crate) prob_a: Option<Vec<f64>>,
    pub(crate) prob_b: Option<Vec<f64>>,
    pub(crate) nr_sv: Vec<u32>,
}

#[derive(Copy, Clone, Debug, Default)]
pub struct Attribute {
    pub(crate) index: u32,
    pub(crate) value: f32,
}

#[derive(Clone, Debug, Default)]
pub struct SupportVector {
    pub coefs: Vec<f32>,
    pub features: Vec<Attribute>,
}

/// Possible error types when loading a [ModelFile].
#[derive(Debug)]
pub enum ModelError {
    /// This signals there was a general parsing error. For models generated with `svm-train`
    /// this should not happen.
    ParsingError,
}

impl<'a> TryFrom<&'a str> for ModelFile<'a> {
    type Error = ModelError;

    /// Parses a string into a SVM model
    fn try_from(model: &str) -> Result<ModelFile, ModelError> {
        // Parse string to struct
        // I fucking regret using `nom` for this ...
        let complete_string = CompleteStr(model);
        let res = svm_file(complete_string);

        match res {
            Ok(m) => Result::Ok(m.1),
            Err(_) => Result::Err(ModelError::ParsingError),
        }
    }
}

/// Accepts an alphanumeric identifier or '_'.
fn svm_non_whitespace(chr: char) -> bool {
    is_alphanumeric(chr as u8) || chr == '_' || chr == '.' || chr == '-'
}

named!(svm_string <CompleteStr, &str>, 
    do_parse! ( x: take_while_s!(svm_non_whitespace) >> (x.0)) 
);

named!(svm_line_string <CompleteStr, (&str)>,
    do_parse!( svm_string >> tag!(" ") >> value: svm_string >> line_ending >> (value) )
);

named!(svm_line_f32 <CompleteStr, (f32)>,
    do_parse!( svm_string >> tag!(" ") >> value: map_res!(svm_string, FromStr::from_str) >> line_ending >> (value) )
);

named!(svm_line_u32 <CompleteStr, (u32)>,
    do_parse!( svm_string >> tag!(" ") >> value: map_res!(svm_string, FromStr::from_str) >> line_ending >> (value) )
);

named!(svm_line_vec_f64 <CompleteStr, (Vec<f64>)>,
    do_parse!( svm_string >> values: many0!(preceded!(tag!(" "), map_res!(svm_string, FromStr::from_str))) >> line_ending >> (values) )
);

named!(svm_line_prob_vec_f64 <CompleteStr, (Vec<f64>)>,
    do_parse!( alt!(tag!("probA") | tag!("probB")) >> values: many0!(preceded!(tag!(" "), map_res!(svm_string, FromStr::from_str))) >> line_ending >> (values) )
);

named!(svm_line_vec_u32 <CompleteStr, (Vec<u32>)>,
    do_parse!( svm_string >> values: many0!(preceded!(tag!(" "), map_res!(svm_string, FromStr::from_str))) >> line_ending >> (values) )
);

named!(svm_attribute <CompleteStr, (Attribute)>,
    do_parse!(
        many0!(tag!(" ")) >>
        index: map_res!(svm_string, FromStr::from_str) >> 
        tag!(":") >>
        value: map_res!(svm_string, FromStr::from_str)  >>
        many0!(tag!(" ")) >>
        (Attribute{
            index,
            value
        })
    )
);

named!(pub svm_header <CompleteStr, Header>,
    do_parse!(
        svm_type: svm_line_string >>
        kernel_type: svm_line_string >>
        gamma: svm_line_f32 >>
        nr_class: svm_line_u32 >>
        total_sv: svm_line_u32 >>
        rho: svm_line_vec_f64 >>
        label: svm_line_vec_u32 >>
        prob_a: opt!(svm_line_prob_vec_f64) >>
        prob_b: opt!(svm_line_prob_vec_f64) >>
        nr_sv: svm_line_vec_u32 >>
        (
            Header {
                svm_type,
                kernel_type,
                gamma,
                nr_class,
                total_sv,
                rho,
                label,
                prob_a,
                prob_b,                
                nr_sv,
            }
        )
    )
);

named_args!(svm_coef(n: u32) <CompleteStr, Vec<f32>>,
    do_parse!(
        rval: count!(map_res!(ws!(svm_string), FromStr::from_str), n as usize) >>
        (rval)
    )
);

named_args!(svm_line_sv(num_coef: u32) <CompleteStr, (SupportVector)>,
    do_parse!(
        coefs: call!(svm_coef, num_coef) >>
        features: many0!(svm_attribute) >>
        many0!(tag!(" ")) >>
        line_ending >>
        (SupportVector {
            coefs,
            features
        })
    )
);

named_args!(svm_svs(num_coef: u32) <CompleteStr, (Vec<SupportVector>)>,
    do_parse!(
        vectors: many0!(call!(svm_line_sv, num_coef)) >>
        (vectors)
    )
);

named!(svm_file <CompleteStr, ModelFile>,
    do_parse!(
        header: svm_header >>
        tag!("SV") >> line_ending >>
        vectors: call!(svm_svs, header.nr_class - 1) >>
        (
            ModelFile {
                header,
                vectors,
            }
        )
    )
);
