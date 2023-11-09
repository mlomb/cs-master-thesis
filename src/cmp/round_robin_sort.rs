use std::cmp::Ordering;

use super::loose_ord::LooseOrd;

/// Unstable sort for vectors of `LooseOrd` types.
///
/// https://en.wikipedia.org/wiki/Round-robin_tournament
pub fn rr_sort<T>(vec: &Vec<T>) -> Vec<&T>
where
    T: LooseOrd,
{
    let n = vec.len();
    let mut matrix = vec![Ordering::Equal; n * n];

    // all-play-all
    for i in 0..n {
        for j in 0..n {
            if i != j {
                matrix[i * n + j] = vec[i].loose_cmp(&vec[j]);
            }
        }
    }

    vec.iter().collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    // test whether sort
    #[test]
    fn ord_sorts_correct() {
        assert_eq!(rr_sort(&vec![3, 2, 1]), vec![&1, &2, &3]);
    }

    fn build_loose_ord_type(&matrix: Vec<Vec<Ordering>>) -> LooseType {
        struct LooseType;

        impl LooseOrd for LooseType {
            fn loose_cmp(&self, other: &Self) -> Ordering {
                Ordering::Equal
            }
        }

        LooseType
    }

    fn loose_ord_sorts_correct() {
        struct LooseType;

        let matrix = vec![
            vec![Ordering::Equal, Ordering::Less, Ordering::Less],
            vec![Ordering::Greater, Ordering::Equal, Ordering::Less],
            vec![Ordering::Greater, Ordering::Greater, Ordering::Equal],
        ];

        impl LooseOrd for LooseType {
            fn loose_cmp(&self, other: &Self) -> Ordering {
                Ordering::Equal
            }
        }
    }
}
