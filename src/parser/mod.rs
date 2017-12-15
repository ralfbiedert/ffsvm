use std::marker::Sized;
mod model;

pub use self::model::ModelFile;



pub trait FromModelFile where Self : Sized {
    
    /// Produces something from a model file. 
    fn from_model(&ModelFile) -> Result<Self, &'static str>;
    
}
