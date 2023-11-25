use shared_memory::{Shmem, ShmemConf, ShmemError};

/// Create or open a shared memory mapping
pub fn open_shmem(id: &str, size: usize) -> Result<Shmem, ShmemError> {
    // required for Windows
    let size = size.next_multiple_of(4096);

    match ShmemConf::new().os_id(id).size(size).create() {
        Err(ShmemError::MappingIdExists) => {
            // if the rare case the mapping already exists, open it
            let mut shmem = ShmemConf::new().os_id(id).open()?;

            // give ownership to Rust
            shmem.set_owner(true);

            // insufficient size, drop and try again
            if shmem.len() != size {
                // force drop (does this really work on Windows?)
                drop(shmem);

                return open_shmem(id, size);
            }

            Ok(shmem)
        }
        res => res,
    }
}
