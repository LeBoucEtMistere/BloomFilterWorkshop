use std::{
    collections::hash_map::{DefaultHasher, RandomState},
    hash::{BuildHasher, Hash, Hasher},
    marker::PhantomData,
};

use bitvec::prelude::*;

/// A generic implementation of bloom filters
///
/// This structure is generic over the type of data, and allow users to enforce a theoretical rate of false positives.
/// The number of hash functions is derived from the expected false positive rate and the size of the filter.
///
/// Example usage:
/// ```
/// use bloom_filter::BloomFilter;
///
/// let mut bloom = BloomFilter::new(100, 0.01);
/// bloom.insert("item");
/// assert!(bloom.contains("item"));
/// ```
pub struct BloomFilter<T: ?Sized> {
    bitmap: BitVec,
    optimal_m: u64,
    optimal_k: u32,
    hashers: [DefaultHasher; 2],
    _marker: PhantomData<T>,
}

const LN2_SQUARED: f64 = core::f64::consts::LN_2 * core::f64::consts::LN_2;

impl<T: ?Sized + Hash> BloomFilter<T> {
    /// Create a new BloomFilter based on its size and the expected false positive rate.
    pub fn new(items_count: usize, fp_rate: f64) -> Self {
        // compute the optimal number of bits to use as filter size
        let optimal_m = Self::bitmap_size(items_count, fp_rate);
        // compute the optimal number of hash function to use
        let optimal_k = Self::optimal_k(fp_rate);
        // create two hashers initialized with a random state to derive all the k hashers from
        let hashers = [
            RandomState::new().build_hasher(),
            RandomState::new().build_hasher(),
        ];

        BloomFilter {
            bitmap: bitvec![0; optimal_m],
            optimal_m: optimal_m as u64,
            optimal_k,
            hashers,
            _marker: PhantomData,
        }
    }

    fn bitmap_size(items_count: usize, fp_rate: f64) -> usize {
        ((-1.0f64 * items_count as f64 * fp_rate.ln()) / LN2_SQUARED).ceil() as usize
    }

    fn optimal_k(fp_rate: f64) -> u32 {
        ((-1.0f64 * fp_rate.ln()) / core::f64::consts::LN_2).ceil() as u32
    }

    /// Insert an element into the Bloom Filter.
    pub fn insert(&mut self, item: &T) {
        // obtain h1 and h2, the two images of item by our two kernel hashing functions
        let (h1, h2) = self.hash_kernel(item);

        // for each of our actual k hash functions, derive the index in the bitvec we need to set to 1
        for k_i in 0..self.optimal_k {
            let index = self.get_index(h1, h2, k_i as u64);
            // this won't panic with out of bounds since index is enforced to be smaller than self.optimal_m, the size of the bitvec
            self.bitmap.set(index, true);
        }
    }

    /// Checks if an element is contained in the bloom filter.
    /// If this returns true, either the element is indeed in the filter or it isn't according to the false positive rate the user selected when building the filter
    /// If this returns false, the element is not in the set.
    pub fn contains(&mut self, item: &T) -> bool {
        let (h1, h2) = self.hash_kernel(item);

        for k_i in 0..self.optimal_k {
            let index = self.get_index(h1, h2, k_i as u64);

            #[allow(clippy::single_match)]
            match self.bitmap.get(index) {
                Some(boolean) => {
                    if !boolean {
                        return false;
                    }
                }
                None => {}
            }
        }

        true
    }

    fn hash_kernel(&self, item: &T) -> (u64, u64) {
        // we don't want the hashing to influence the state of the hasher for further computations
        // so we clone our two hashers before hashing.
        let hasher1 = &mut self.hashers[0].clone();
        let hasher2 = &mut self.hashers[1].clone();

        item.hash(hasher1);
        item.hash(hasher2);

        let hash1 = hasher1.finish();
        let hash2 = hasher2.finish();

        (hash1, hash2)
    }

    fn get_index(&self, h1: u64, h2: u64, k_i: u64) -> usize {
        // compute H_k(x) = h1(x) + k_i * h2(x) and use it to index into the m elements of the bitvec
        (h1.wrapping_add((k_i).wrapping_mul(h2)) % self.optimal_m) as usize
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn insert() {
        let mut bloom = BloomFilter::new(100, 0.01);
        bloom.insert("item");
        assert!(bloom.contains("item"));
    }

    #[test]
    fn check_and_insert() {
        let mut bloom = BloomFilter::new(100, 0.01);
        assert!(!bloom.contains("item_1"));
        assert!(!bloom.contains("item_2"));
        bloom.insert("item_1");
        assert!(bloom.contains("item_1"));
    }
}
