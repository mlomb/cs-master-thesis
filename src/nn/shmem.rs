use shared_memory::{Shmem, ShmemConf, ShmemError};

/// Create or open a shared memory mapping
pub fn open_shmem(id: &str, size: usize) -> Result<Shmem, ShmemError> {
    match ShmemConf::new().os_id(id).size(size).open() {
        Err(ShmemError::MapOpenFailed(_)) => ShmemConf::new().os_id(id).size(size).create(),
        res => res,
    }
}
