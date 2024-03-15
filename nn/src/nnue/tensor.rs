use std::fmt::Debug;
use std::io::{Cursor, Read};
use std::{
    alloc::{alloc, dealloc, Layout},
    fmt::Formatter,
};

/// Memory is aligned to 32 bits
pub struct Tensor<T> {
    layout: Layout,
    data: *mut T,
}

impl<T> Tensor<T> {
    pub fn from_cursor(cursor: &mut Cursor<Vec<u8>>, len: usize) -> std::io::Result<Self> {
        let mut data = vec![0u8; len * std::mem::size_of::<T>()];
        cursor.read_exact(data.as_mut_slice())?;

        Ok(Self::from_slice(unsafe {
            std::slice::from_raw_parts(data.as_ptr() as *const T, len)
        }))
    }

    pub fn from_slice(from: &[T]) -> Self {
        let layout = Layout::from_size_align(from.len() * std::mem::size_of::<T>(), 32).unwrap();
        let data = unsafe { alloc(layout) } as *mut T;
        unsafe {
            std::ptr::copy_nonoverlapping(from.as_ptr(), data, from.len());
        }
        Self { layout, data }
    }

    pub fn zeros(size: usize) -> Self {
        let layout = Layout::from_size_align(size * std::mem::size_of::<T>(), 32).unwrap();
        let data = unsafe { alloc(layout) } as *mut T;
        unsafe {
            std::ptr::write_bytes(data, 0, size);
        }
        Self { layout, data }
    }

    pub fn as_ptr(&self) -> *const T {
        self.data as *const T
    }

    pub fn as_mut_ptr(&self) -> *mut T {
        self.data
    }

    pub fn as_slice(&self) -> &[T] {
        unsafe { std::slice::from_raw_parts(self.data, self.len()) }
    }

    pub fn as_mut_slice(&self) -> &mut [T] {
        unsafe { std::slice::from_raw_parts_mut(self.data, self.len()) }
    }

    pub fn len(&self) -> usize {
        self.layout.size() / std::mem::size_of::<T>()
    }
}

impl<T> Drop for Tensor<T> {
    fn drop(&mut self) {
        unsafe {
            dealloc(self.data as *mut u8, self.layout);
        }
    }
}

impl<T: Debug> Debug for Tensor<T> {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "Tensor({:?})", self.as_slice())
    }
}
