// TODO:
// * One call classify multiple problems
// * Use SIMD
// * Use parallelism

#[macro_use]
extern crate nom;
extern crate faster;

use nom::{IResult, line_ending, space, multispace, alphanumeric, is_alphanumeric };
use std::str;

/// Main type we use in here
type Feature = f32;
type Probability = f32;
type Class = u32;


/// Base data for our CSVM
pub struct CSVM {
}


pub struct ModelCSVM {
}



pub fn predict_probability_csvm(csvm: &ModelCSVM, feature_vector: &[Feature], probabilities: &mut [Probability]) {

//    faster SIMD goes here ...
//    let x = (&feature_vector[..]);
//    let mut mp = x.simd_iter().map(|vector| { f32s::splat(10.0) + vector.abs() });
//    let c = mp.scalar_collect();

}


/// Accepts an alphanumeric identifier or '_'
fn svm_identifier_character(chr: u8) -> bool { is_alphanumeric(chr) || chr == '_' as u8 || chr == '.' as u8 || chr == '-' as u8}


/// Parses an identifier or value
named!(svm_identifier, take_while!(svm_identifier_character));

/// Parses a single line
named!(svm_header_line_2 <&[u8], (&str, &str)>,
    do_parse!(
        key: svm_identifier >>
        tag!(" ") >>
        value: svm_identifier >>
        line_ending >>
        (str::from_utf8(key).unwrap(), str::from_utf8(value).unwrap())
    )
);

struct FFF<'a> {
    svm: &'a str
}

/// Parses a single line
named!(svm_header_line_3 <&[u8], (&str, &str, &str)>,
    do_parse!(
        key: svm_identifier >>
        tag!(" ") >>
        value1: svm_identifier >>
        tag!(" ") >>
        value2: svm_identifier >>
        line_ending >>
        (str::from_utf8(key).unwrap(), str::from_utf8(value1).unwrap(), str::from_utf8(value2).unwrap())
    )
);


/// Parses the header
named!(svm_header <&[u8], FFF>,
    do_parse!(
        x: svm_header_line_2 >>
        (
            FFF {
                svm: x.1
            }
        )
    )
);


//        key: some_token!
//        >> opt!(space)
//        >> value: map_res!(some_token)
//        >> (key, value)

//named!(string_single <&[u8], (&str, &str)>,
//    some_token
//);

/// Parses a string into a SVM model
pub fn parse_model_csvm() -> Option<ModelCSVM> {
    const model:&str = include_str!("test.model");

    let res = svm_header(model.as_bytes());

    match res {
        IResult::Done(a, b) => {
            println!("{:?} {:?}", b.svm, 2);
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