use nom::{is_alphanumeric, line_ending};
use std::str;
use std::str::FromStr;


#[derive(Debug)]
pub struct Header<'a> {
    pub svm_type: &'a str,
    pub kernel_type: &'a str,
    pub gamma: f32,
    pub nr_class: u32,
    pub total_sv: u32,
    pub rho: Vec<f64>,
    pub label: Vec<u32>,
    pub nr_sv: Vec<u32>,
}

#[derive(Debug)]
pub struct Attribute {
    pub index: u32,
    pub value: f32,
}

#[derive(Debug)]
pub struct SupportVector {
    pub coefs: Vec<f32>,
    pub features: Vec<Attribute>,
}

#[derive(Debug)]
pub struct LibSvmModel<'a> {
    pub header: Header<'a>,
    pub vectors: Vec<SupportVector>,
}



impl<'a> LibSvmModel<'a> {
    /// Parses a string into a SVM model
    pub fn from_str(model: &str) -> Result<LibSvmModel, &'static str> {
        // Parse string to struct
        let res = svm_file(model);

        match res {
            Ok(m) => Result::Ok(m.1),
            Err(_) => Result::Err("Error parsing file."),
        }
    }
}



/// Accepts an alphanumeric identifier or '_'.
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

named!(svm_line_vec_f64 <&str, (Vec<f64>)>,
    do_parse!( svm_string >> values: many0!(preceded!(tag!(" "), map_res!(svm_string, FromStr::from_str))) >> line_ending >> (values) )
);

named!(svm_line_vec_u32 <&str, (Vec<u32>)>,
    do_parse!( svm_string >> values: many0!(preceded!(tag!(" "), map_res!(svm_string, FromStr::from_str))) >> line_ending >> (values) )
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
        rho: svm_line_vec_f64 >>
        label: svm_line_vec_u32 >>
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

named_args!(svm_coef(n: u32) <&str,Vec<f32>>,
    do_parse!(
        opt!(tag!(" ")) >>
        rval: count!(map_res!(svm_string, FromStr::from_str), n as usize) >>
        (rval)
    )
);


named_args!(svm_line_sv(num_coef: u32) <&str, (SupportVector)>,
    do_parse!(
        // label: map_res!(svm_string, FromStr::from_str) >>
        coefs: call!(svm_coef, num_coef) >>
        values: many0!(preceded!(tag!(" "), svm_attribute)) >>
        many0!(tag!(" ")) >>
        line_ending >>
        (SupportVector {
            coefs: coefs,
            features: values
        })
    )
);

named_args!(svm_svs(num_coef: u32) <&str, (Vec<SupportVector>)>,
    do_parse!(
        vectors: many0!(call!(svm_line_sv, num_coef)) >>
        (vectors)
    )
);


named!(svm_file <&str, LibSvmModel>,
    do_parse!(
        header: svm_header >>
        svm_string >> line_ending >>
        support_vectors: call!(svm_svs, header.nr_class - 1) >>
        (
            LibSvmModel {
                header: header,
                vectors: support_vectors,
            }
        )
    )
);
