use super::loose_ord::LooseOrd;
use std::cmp::Ordering;

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
                let ord = vec[i].loose_cmp(&vec[j]);
                matrix[i * n + j] = ord;
            }
        }
    }

    fn sort_scoring(indexes: &mut [usize], n: usize, matrix: &Vec<Ordering>) {
        if indexes.len() == 1 {
            return;
        }

        let mut scores = vec![0; n];

        for i in 0..indexes.len() {
            for j in 0..indexes.len() {
                if i != j {
                    let true_i = indexes[i];
                    let true_j = indexes[j];

                    // lookup, dont compute again
                    let ord = matrix[true_i * n + true_j];
                    scores[true_i] += match ord {
                        Ordering::Less => 2,
                        Ordering::Equal => 1,
                        Ordering::Greater => 0,
                    };
                    scores[true_j] += match ord {
                        Ordering::Less => 0,
                        Ordering::Equal => 1,
                        Ordering::Greater => 2,
                    };
                }
            }
        }

        indexes.sort_unstable_by_key(|index| -scores[*index]);

        let indexes_len = indexes.len();
        let mut iter = indexes.group_by_mut(|a, b| scores[*a] == scores[*b]);

        while let Some(group) = iter.next() {
            if group.len() == indexes_len {
                // this means there is only one group (group 0)
                // so we did not untie any scores, so we are done

                // indexes.shuffle() ?
                return;
            }

            sort_scoring(group, n, matrix);
        }
    }

    let mut result: Vec<usize> = (0..n).collect();
    sort_scoring(&mut result.as_mut_slice(), n, &matrix);

    result.iter().map(|i| &vec[*i]).collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    // test whether sort
    #[test]
    fn ord_sorts_correct() {
        assert_eq!(rr_sort(&vec![2, 1]), vec![&1, &2]);
        assert_eq!(rr_sort(&vec![3, 2, 1]), vec![&1, &2, &3]);
        assert_eq!(rr_sort(&vec![2, 1, 2]), vec![&1, &2, &2]);
    }

    #[derive(Debug)]
    struct LooseType {
        value: u32,
        vs_right: HashMap<u32, Ordering>,
    }

    impl LooseOrd for LooseType {
        fn loose_cmp(&self, other: &Self) -> Ordering {
            self.vs_right.get(&other.value).unwrap().clone()
        }
    }

    impl PartialEq for LooseType {
        fn eq(&self, other: &Self) -> bool {
            self.value == other.value
        }
    }

    #[test]
    fn symmetric_loose_ord_sorts_correctly() {
        let one = LooseType {
            value: 1,
            vs_right: HashMap::from([
                (2, Ordering::Less),
                (3, Ordering::Less),
                (4, Ordering::Greater),
            ]),
        };
        let two = LooseType {
            value: 2,
            vs_right: HashMap::from([
                (1, Ordering::Greater),
                (3, Ordering::Less),
                (4, Ordering::Greater),
            ]),
        };
        let three = LooseType {
            value: 3,
            vs_right: HashMap::from([
                (1, Ordering::Greater),
                (2, Ordering::Greater),
                (4, Ordering::Less),
            ]),
        };
        let four = LooseType {
            value: 4,
            vs_right: HashMap::from([
                (1, Ordering::Less),
                (2, Ordering::Less),
                (3, Ordering::Greater),
            ]),
        };

        assert_eq!(
            rr_sort(&vec![two, four, one, three])
                .iter()
                .map(|x| x.value)
                .collect::<Vec<u32>>(),
            vec![4, 1, 2, 3]
        );
    }
}
