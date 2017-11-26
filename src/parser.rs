use nom::{IResult, line_ending, is_alphanumeric };
use std::str;
use std::str::FromStr;

use types::*;

#[derive(Debug)]
struct Header<'a> {
    svm_type: &'a str,
    kernel_type: &'a str,
    gamma: f32,
    nr_class: u32,
    total_sv: u32,
    rho: f32,
    label: Vec<&'a str>,
    nr_sv: Vec<u32>,
}

#[derive(Debug)]
struct Attribute {
    index: u32,
    value: Feature
}

#[derive(Debug)]
struct SupportVector {
    label: Feature,
    features: Vec<Attribute>
}

#[derive(Debug)]
struct ModelFile<'a> {
    header: Header<'a>,
    vectors: Vec<SupportVector>
}



/// Accepts an alphanumeric identifier or '_'
fn svm_non_whitespace(chr: char) -> bool {
    is_alphanumeric(chr as u8) || chr == '_' || chr == '.' || chr == '-'
}

named!(svm_string <&str, &str>, take_while_s!(svm_non_whitespace));


named!(svm_line_string <&str, (&str)>,
    do_parse!( svm_string >> tag!(" ") >> value: svm_string >> line_ending >> (value) )
);

named!(svm_line_f32 <&str, (f32)>,
    do_parse!( svm_string >> tag!(" ") >> value: map_res!(svm_string, FromStr::from_str) >> line_ending >> (value) )
);

named!(svm_line_u32 <&str, (u32)>,
    do_parse!( svm_string >> tag!(" ") >> value: map_res!(svm_string, FromStr::from_str) >> line_ending >> (value) )
);

named!(svm_line_vec_u32 <&str, (Vec<u32>)>,
    do_parse!( svm_string >> values: many0!(preceded!(tag!(" "), map_res!(svm_string, FromStr::from_str))) >> line_ending >> (values) )
);

named!(svm_line_vec_str <&str, (Vec<&str>)>,
    do_parse!( svm_string >> values: many0!(preceded!(tag!(" "), svm_string)) >> line_ending >> (values) )
);

named!(svm_attribute <&str, (Attribute)>,
    do_parse!(
        index: map_res!(svm_string, FromStr::from_str) >>
        tag!(":") >>
        value: map_res!(svm_string, FromStr::from_str)  >>
        (Attribute{
            index: index,
            value: value
        })
    )
);

named!(svm_header <&str, Header>,
    do_parse!(
        svm_type: svm_line_string >>
        kernel_type: svm_line_string >>
        gamma: svm_line_f32 >>
        nr_class: svm_line_u32 >>
        total_sv: svm_line_u32 >>
        rho: svm_line_f32 >>
        label: svm_line_vec_str >>
        nr_sv: svm_line_vec_u32 >>
        (
            Header {
                svm_type: svm_type,
                kernel_type: kernel_type,
                gamma: gamma,
                nr_class: nr_class,
                total_sv: total_sv,
                rho: rho,
                label: label,
                nr_sv: nr_sv,
            }
        )
    )
);

named!(svm_line_sv <&str, (SupportVector)>,
    do_parse!(
        label: map_res!(svm_string, FromStr::from_str) >>
        values: many0!(preceded!(tag!(" "), svm_attribute)) >>
        many0!(tag!(" ")) >>
        line_ending >>
        (SupportVector {
            label: label,
            features: values
        })
    )
);

named!(svm_svs <&str, (Vec<SupportVector>)>,
    do_parse!(
        vectors: many0!(svm_line_sv) >>
        (vectors)
    )
);


named!(svm_file <&str, ModelFile>,
    do_parse!(
        header: svm_header >>
        svm_string >> line_ending >>
        support_vectors: svm_svs >>
        (
            ModelFile {
                header: header,
                vectors: support_vectors,
            }
        )
    )
);


/// Parses a string into a SVM model
pub fn parse_model_csvm() -> Option<ModelCSVM> {

    let model: &str = include_str!("test.model");
    let res = svm_file(model);


    match res {
        IResult::Done(a, b) => {
            //            println!("{:?} {:?}", b.header.total_sv, b.vectors[0]);
                        println!("{:?} {:?}", b, 1);
        },

        IResult::Error(e) => {
            println!("Err {:?}", e)
        },

        IResult::Incomplete(_) => {
            println!("Inc")

        }
    }

    Option::Some(ModelCSVM{})
}