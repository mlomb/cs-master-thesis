use ort::{Session, SessionBuilder};
use std::fs;

pub struct Model {
    session: Session,
}

/*
impl Model {
    pub fn load(path: &str) -> Self {
        let session = SessionBuilder::new(&environment)?
            .with_intra_threads(1)?
            .with_model_from_file("models/best/onnx_model.onnx")?;
    }
}
*/

pub struct Pepe {
    root: String,
}

impl Pepe {
    pub fn new(root: &str) -> Self {
        Self {
            root: root.to_string(),
        }
    }

    pub fn latest(&self) {
        if let Ok(entries) = fs::read_dir(self.root.as_str()) {
            for entry in entries {
                if let Ok(entry) = entry {
                    // Here, `entry` is a `DirEntry`.
                    println!("{:?}", fs::canonicalize(entry.path()));
                }
            }
        }
    }
}
