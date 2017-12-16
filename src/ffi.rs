use libc::{c_char, uint32_t};
use std::ffi::CStr;
use std::slice;
use std::ptr::{null_mut};
use std::convert::TryFrom;

use svm::{RbfCSVM, Problem};
use parser::ModelFile;


/// Possible error conditions we can return.
enum Errors {
    Ok = 0,
    NullPointerPassed = -1,
    NoValidUTF8 = -10,
    ModelParseError = -20,
    SVMCreationError = -30,
    SVMNoModel = -31,
    ProblemPoolTooSmall = -40,
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
    if context_ptr.is_null() { return Errors::NullPointerPassed as i32; }
    
    let context = Context {
        max_problems: 1,
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
pub extern fn ffsvm_load_model(context_ptr: *mut Context, model_c_ptr: *const c_char) -> i32 {
    if context_ptr.is_null() { return Errors::NullPointerPassed as i32; }
    if model_c_ptr.is_null() { return Errors::NullPointerPassed as i32; }

    let context = unsafe { 
        &mut *context_ptr 
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

    context.problems = (0 .. context.max_problems).map(|_| Problem::from(&svm)).collect();
    context.model = Some(Box::from(svm));

    Errors::Ok as i32
}




#[no_mangle]
pub extern fn ffsvm_set_max_problems(context_ptr: *mut Context, max_problems: usize) -> i32 {
    if context_ptr.is_null() { return Errors::NullPointerPassed as i32; }


    let context = unsafe {
        &mut *context_ptr
    };
    
    match &context.model {
        &None => { return Errors::SVMNoModel as i32; }
        &Some(ref model) => {
            let svm = model.as_ref();
            context.max_problems = max_problems;
            context.problems = (0 .. max_problems).map(|_| Problem::from(svm)).collect();
        }
    }

    Errors::Ok as i32
}



#[no_mangle]
pub extern fn ffsvm_predict_values(context_ptr: *mut Context, features_ptr: *mut f32, features_len: usize, labels_ptr: *mut u32, labels_len: u32) -> i32 {
    if context_ptr.is_null() { return Errors::NullPointerPassed as i32; }
    if features_ptr.is_null() { return Errors::NullPointerPassed as i32; }
    if labels_ptr.is_null() { return Errors::NullPointerPassed as i32; }

    
    let context = unsafe {
        &mut *context_ptr
    };
    
    if num_problems > context.max_problems {
        return Errors::ProblemPoolTooSmall as i32;
    }
    
    let features = unsafe {
        labels_ptr
    }

    match &context.model {
        &None => { return Errors::SVMNoModel as i32; }
        &Some(ref model) => {
            let svm = model.as_ref();
            
            let ptr_size = num_problems * svm.num_attributes; 
        }
    }



    Errors::Ok as i32
}


#[no_mangle]
pub extern fn ffsvm_context_destroy(context_ptr: *mut *mut Context) -> i32 {
    if context_ptr.is_null() { return Errors::NullPointerPassed as i32; }

    let context = unsafe {
        Box::from_raw(*context_ptr)
    };
    
    Errors::Ok as i32
}


