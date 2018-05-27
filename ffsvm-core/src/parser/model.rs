use nom::{is_alphanumeric, line_ending, types::CompleteStr};
use std::{
    convert::TryFrom, str::{self, FromStr},
};

/// Parsing result of a model file (start here).
/// 
#[derive(Clone, Debug, Default)]
pub struct ModelFile<'a> {
    pub (crate) header: Header<'a>,
    pub (crate) vectors: Vec<SupportVector>,
}

#[derive(Clone, Debug, Default)]
pub struct Header<'a> {
    pub (crate) svm_type: &'a str,
    pub (crate) kernel_type: &'a str,
    pub (crate) gamma: f32,
    pub (crate) nr_class: u32,
    pub (crate) total_sv: u32,
    pub (crate) rho: Vec<f64>,
    pub (crate) label: Vec<u32>,
    pub (crate) prob_a: Option<Vec<f64>>,
    pub (crate) prob_b: Option<Vec<f64>>,
    pub (crate) nr_sv: Vec<u32>,
}

#[derive(Copy, Clone, Debug, Default)]
pub struct Attribute {
    pub (crate) index: u32,
    pub (crate) value: f32,
}

#[derive(Clone, Debug, Default)]
pub struct SupportVector {
    pub coefs: Vec<f32>,
    pub features: Vec<Attribute>,
}

#[derive(Debug)]
pub enum ModelError {
    ParsingError    
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
