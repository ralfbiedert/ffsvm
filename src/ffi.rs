use libc::{c_char, uint32_t};
use std::ffi::CStr;
use std::ptr::{null_mut};
use std::convert::TryFrom;

use svm::{RbfCSVM, Problem};
use parser::ModelFile;


/// Possible error conditions we can return.
enum Errors {
    Ok = 0,
    NoValidUTF8 = -1,
    ModelParseError = -2,
    SVMCreationError = -3,
}


/// The main context we expose over FFI, containing everything
/// we need. 
pub struct Context {
    max_problems: usize,
    model: Option<Box<RbfCSVM>>,
    problems: Vec<Problem>,
}

#[no_mangle]
pub extern fn ffsvm_test(value: i32) -> i32 {
    println!("Function ffsvm_test({}); called. If you can read this, it works.", value);
    value * value
}


#[no_mangle]
pub extern fn ffsvm_context_create(context_ptr: *mut *mut Context) -> i32 {
    
    let context = Context {
        max_problems: 128,
        model: None,
        problems: Vec::new()
    };
    
    let boxed = Box::from(context);
    let context_raw = Box::into_raw(boxed);
    
    unsafe {
        *context_ptr = context_raw;
    }
    
    Errors::Ok as i32
}

#[no_mangle]
pub extern fn ffsvm_parse_model(context: *mut Context, model_c_ptr: *const c_char, max_problems: usize) -> i32 {
    assert!(!context.is_null());
    assert!(!model_c_ptr.is_null());

    let context = unsafe { 
        &mut *context 
    };
    
    // Convert pointer to our strucutre  
    let c_str = unsafe {
        CStr::from_ptr(model_c_ptr)
    };

    // Create UTF8 string
    let model_str = match c_str.to_str() {
        Err(_) => { return Errors::NoValidUTF8 as i32; }
        Ok(s) => { s }
    };
    
    // Parse model
    let model = match ModelFile::try_from(model_str) {
        Err(_) => { return Errors::ModelParseError as i32; }
        Ok(m) => { m }
    };
    
    // Convert into SVM
    let svm = match RbfCSVM::try_from(&model) {
        Err(_) => { return Errors::SVMCreationError as i32; }
        Ok(m) => { m }
    };

    context.max_problems = max_problems;
    context.problems = (0 .. max_problems).map(|_| Problem::from(&svm)).collect();
    context.model = Some(Box::from(svm));

    Errors::Ok as i32
}


#[no_mangle]
pub extern fn ffsvm_predict_values(context: *mut Context, num_problems: usize, data: *mut f32, return_values: *mut u32) -> i32 {
    assert!(!context.is_null());

    Errors::Ok as i32
}


#[no_mangle]
pub extern fn ffsvm_context_destroy(context_ptr: *mut *mut Context) -> i32 {

    let context = unsafe {
        Box::from_raw(*context_ptr)
    };
    
    Errors::Ok as i32
}


