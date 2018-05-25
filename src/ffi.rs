use libc::c_char;
use std::{convert::TryFrom, ffi::CStr, ptr::null_mut, slice};

use parser::ModelFile;
use svm::{PredictProblem, Problem, RbfCSVM};

/// Possible error conditions we can return.
enum Errors {
    Ok = 0,
    NullPointerPassed = -1,
    NoValidUTF8 = -2,
    ModelParseError = -20,
    SVMCreationError = -30,
    SVMNoModel = -31,
    SVMModelAlreadyLoaded = -32,
    ProblemPoolTooSmall = -40,
    ProblemLengthNotMultipleOfAttributes = -41,
    LabelLengthDoesNotEqualProblems = -42,
    ProbabilitiesDoesNotEqualProblemsXAttributes = -43,
    LabelLengthDoesNotMatchClassesLength = -50,
}

/// The main context we expose over FFI, containing everything
/// we need.
pub struct Context {
    max_problems: usize,
    model: Option<Box<RbfCSVM>>,
    problems: Vec<Problem>,
}

/// Tests if FFI works.
#[no_mangle]
pub extern "C" fn ffsvm_test(value: i32) -> i32 {
    println!(
        "Function ffsvm_test({}); called. If you can read this, it works.",
        value
    );
    value * value
}

/// Creates a context we need for operations.
///
/// Although FFSVM uses threading internally, this context may only be used
/// from a single thread by yourself!
#[no_mangle]
pub unsafe extern "C" fn ffsvm_context_create(context_ptr: *mut *mut Context) -> i32 {
    if context_ptr.is_null() {
        return Errors::NullPointerPassed as i32;
    }

    let context = Context {
        max_problems: 1,
        model: None,
        problems: Vec::new(),
    };

    let boxed = Box::from(context);
    let context_raw = Box::into_raw(boxed);

    *context_ptr = context_raw;

    Errors::Ok as i32
}

/// Loads a model into the given context.
///
/// The model must be passed as a `0x0` terminated C-string.   
#[no_mangle]
pub unsafe extern "C" fn ffsvm_model_load(
    context_ptr: *mut Context,
    model_c_ptr: *const c_char,
) -> i32 {
    if context_ptr.is_null() {
        return Errors::NullPointerPassed as i32;
    }
    if model_c_ptr.is_null() {
        return Errors::NullPointerPassed as i32;
    }

    let context = &mut *context_ptr;
    let c_str = CStr::from_ptr(model_c_ptr);

    // Create UTF8 string
    let model_str = match c_str.to_str() {
        Err(_) => {
            return Errors::NoValidUTF8 as i32;
        }
        Ok(s) => s,
    };

    // Parse model
    let model = match ModelFile::try_from(model_str) {
        Err(_) => {
            return Errors::ModelParseError as i32;
        }
        Ok(m) => m,
    };

    // Convert into SVM
    let svm = match RbfCSVM::try_from(&model) {
        Err(_) => {
            return Errors::SVMCreationError as i32;
        }
        Ok(m) => m,
    };

    context.problems = (0 .. context.max_problems)
        .map(|_| Problem::from(&svm))
        .collect();
    context.model = Some(Box::from(svm));

    Errors::Ok as i32
}

/// Sets the maximum number of problems we process at the same time.
///
/// Must be called [b]before[/b] the model is loaded.  
///
/// If you set this very low, you may not classify many problems at the same time and might
/// have to make multiple classification calls, increasing overhead.
///
/// If you set this too high, this lib will allocate unused resources. Each [i]problem slot[/i]
/// roughly consumes `n*(m+1) + m*m` floats, where `n` is the number of total support vectors of
/// the loaded model and `m` is the number of classes.         
#[no_mangle]
pub unsafe extern "C" fn ffsvm_set_max_problems(
    context_ptr: *mut Context,
    max_problems: u32,
) -> i32 {
    if context_ptr.is_null() {
        return Errors::NullPointerPassed as i32;
    }

    let context = &mut *context_ptr;

    match context.model {
        None => {
            context.max_problems = max_problems as usize;
        }
        Some(_) => {
            return Errors::SVMModelAlreadyLoaded as i32;
        }
    }

    Errors::Ok as i32
}

/// Given a number of problems (features), predict their classes with the current model.   
#[no_mangle]
pub unsafe extern "C" fn ffsvm_predict_values(
    context_ptr: *mut Context,
    features_ptr: *mut f32,
    features_len: u32,
    labels_ptr: *mut u32,
    labels_len: u32,
) -> i32 {
    if context_ptr.is_null() {
        return Errors::NullPointerPassed as i32;
    }
    if features_ptr.is_null() {
        return Errors::NullPointerPassed as i32;
    }
    if labels_ptr.is_null() {
        return Errors::NullPointerPassed as i32;
    }

    let context = &mut *context_ptr;
    let features = slice::from_raw_parts(features_ptr, features_len as usize);
    let labels = slice::from_raw_parts_mut(labels_ptr, labels_len as usize);

    let svm = match context.model {
        None => {
            return Errors::SVMNoModel as i32;
        }
        Some(ref model) => model.as_ref(),
    };

    // Make sure the pointers have the right length
    let num_problems = match features.len() % svm.num_attributes {
        0 => features.len() / svm.num_attributes,
        _ => {
            return Errors::ProblemLengthNotMultipleOfAttributes as i32;
        }
    };

    if num_problems > context.max_problems {
        return Errors::ProblemPoolTooSmall as i32;
    }

    if num_problems != labels_len as usize {
        return Errors::LabelLengthDoesNotEqualProblems as i32;
    }

    let problems = &mut context.problems;
    let num_attributes = svm.num_attributes;

    // Copy features to respective problems
    for i in 0 .. num_problems {
        let this_problem = &features[i * num_attributes .. (i + 1) * num_attributes];
        let src = &this_problem[.. num_attributes];

        // Internal problem length can be longer than given one due to SIMD alignment.
        problems[i].features[.. num_attributes].clone_from_slice(src);
    }

    // Predict values for given slice of actually used real problems.
    svm.predict_values(&mut problems[0 .. num_problems]);

    // And store the results
    for i in 0 .. num_problems {
        labels[i] = problems[i].label
    }

    Errors::Ok as i32
}

/// Given a number of problems (features), predict their classes with the current model.   
#[no_mangle]
pub unsafe extern "C" fn ffsvm_predict_probabilities(
    context_ptr: *mut Context,
    features_ptr: *mut f32,
    features_len: u32,
    probabilities_ptr: *mut f32,
    probabilities_len: u32,
) -> i32 {
    if context_ptr.is_null() {
        return Errors::NullPointerPassed as i32;
    }
    if features_ptr.is_null() {
        return Errors::NullPointerPassed as i32;
    }
    if probabilities_ptr.is_null() {
        return Errors::NullPointerPassed as i32;
    }

    let context = &mut *context_ptr;
    let features = slice::from_raw_parts(features_ptr, features_len as usize);
    let probabilities = slice::from_raw_parts_mut(probabilities_ptr, probabilities_len as usize);

    let svm = match context.model {
        None => return Errors::SVMNoModel as i32,
        Some(ref model) => model.as_ref(),
    };

    // Make sure the pointers have the right length
    let num_problems = match features.len() % svm.num_attributes {
        0 => features.len() / svm.num_attributes,
        _ => return Errors::ProblemLengthNotMultipleOfAttributes as i32,
    };

    if num_problems > context.max_problems {
        return Errors::ProblemPoolTooSmall as i32;
    }

    if probabilities_len != (num_problems * svm.classes.len()) as u32 {
        return Errors::ProbabilitiesDoesNotEqualProblemsXAttributes as i32;
    }

    let problems = &mut context.problems;
    let num_attributes = svm.num_attributes;

    // Copy features to respective problems
    for i in 0 .. num_problems {
        let this_problem = &features[i * num_attributes .. (i + 1) * num_attributes];
        let src = &this_problem[.. num_attributes];

        // Internal problem length can be longer than given one due to SIMD alignment.
        problems[i].features[.. num_attributes].clone_from_slice(src);
    }

    // Predict values for given slice of actually used real problems.
    svm.predict_probabilities(&mut problems[0 .. num_problems]);

    let mut ptr = 0;

    // And store the results
    for i in 0 .. num_problems {
        for j in 0 .. svm.classes.len() {
            probabilities[ptr] = problems[i].probabilities[j] as f32;
            ptr += 1;
        }
    }

    Errors::Ok as i32
}

/// Given a number of problems (features), predict their classes with the current model.   
#[no_mangle]
pub unsafe extern "C" fn ffsvm_model_get_labels(
    context_ptr: *mut Context,
    labels_ptr: *mut u32,
    labels_len: u32,
) -> i32 {
    if context_ptr.is_null() {
        return Errors::NullPointerPassed as i32;
    }
    if labels_ptr.is_null() {
        return Errors::NullPointerPassed as i32;
    }

    let context = &mut *context_ptr;
    let labels = slice::from_raw_parts_mut(labels_ptr, labels_len as usize);

    let svm = match context.model {
        None => {
            return Errors::SVMNoModel as i32;
        }
        Some(ref model) => model.as_ref(),
    };

    if labels_len != svm.classes.len() as u32 {
        return Errors::LabelLengthDoesNotMatchClassesLength as i32;
    }

    for i in 0 .. svm.classes.len() {
        labels[i] = svm.classes[i].label;
    }

    Errors::Ok as i32
}

/// Destroy the given context.
#[no_mangle]
pub unsafe extern "C" fn ffsvm_context_destroy(context_ptr: *mut *mut Context) -> i32 {
    if context_ptr.is_null() {
        return Errors::NullPointerPassed as i32;
    }

    // This claims ownership of the context box, and once the scope {} ends,
    // destroys it.
    {
        Box::from_raw(*context_ptr);
    }

    // We put this in a separate block to be sure the Box above is actually
    // destroyed when it leaves the unsafe { } scope above. Might be supersition.
    *context_ptr = null_mut();

    Errors::Ok as i32
}
