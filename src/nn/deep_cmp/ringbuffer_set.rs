use rand::prelude::*;
use ringbuffer::{AllocRingBuffer, RingBuffer};
use std::{collections::HashSet, hash::Hash};

pub struct RingBufferSet<T> {
    ring: AllocRingBuffer<T>,
    set: HashSet<T>,
}

impl<T> RingBufferSet<T>
where
    T: Eq + Hash + Clone,
{
    pub fn new(capacity: usize) -> Self {
        Self {
            ring: AllocRingBuffer::new(capacity),
            set: HashSet::new(),
        }
    }

    pub fn insert(&mut self, value: T) {
        if self.set.insert(value.clone()) {
            if self.ring.is_full() {
                // remove the oldest element from set
                let removed = self.ring.front().unwrap();
                self.set.remove(&removed);
                // the next push will overwrite the removed element
            }
            self.ring.push(value);
        }
    }

    pub fn len(&self) -> usize {
        self.set.len()
    }

    #[allow(dead_code)]
    pub fn to_vec(&self) -> Vec<&T> {
        self.ring.iter().collect()
    }

    #[allow(dead_code)]
    pub fn sample<R: Rng>(&self, rng: &mut R, amount: usize) -> Vec<&T> {
        let mut vec = self.ring.iter().choose_multiple(rng, amount);
        // choose_multiple picks random elements
        // but does not guarantee shuffled order
        vec.shuffle(rng);
        vec
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ring_works() {
        let mut rs = RingBufferSet::new(3);

        rs.insert(1);
        rs.insert(2);
        rs.insert(3);
        rs.insert(4); // should replace 1

        assert_eq!(rs.to_vec(), vec![&2, &3, &4]);
        assert_eq!(rs.ring.len(), 3);
        assert_eq!(rs.set.len(), 3);
    }

    #[test]
    fn set_works() {
        let mut rs = RingBufferSet::new(3);

        rs.insert(1);
        rs.insert(2);
        rs.insert(3);
        rs.insert(4); // should replace 1
        rs.insert(3); // should not be inserted
        rs.insert(5); // should replace 2

        assert_eq!(rs.to_vec(), vec![&3, &4, &5]);
        assert_eq!(rs.ring.len(), 3);
        assert_eq!(rs.set.len(), 3);
    }

    #[test]
    fn sample() {
        let mut rs = RingBufferSet::new(3);

        rs.insert(1);
        rs.insert(2);
        rs.insert(3);

        let sample = rs.sample(&mut rand::thread_rng(), 3);

        assert_eq!(sample.len(), 3);
        assert!(sample.contains(&&1));
        assert!(sample.contains(&&2));
        assert!(sample.contains(&&3));
    }
}
