use std::cmp::Ordering;

/// Unstable sort for vectors of comparable elements.
/// It does not assume transitivity or symmetry.
///
/// https://en.wikipedia.org/wiki/Round-robin_tournament
pub fn rr_sort<'a, T>(vec: &'a Vec<T>, cmp: &dyn Fn(&T, &T) -> Ordering) -> Vec<&'a T> {
    let n = vec.len();
    let mut matrix = vec![Ordering::Equal; n * n];

    // all-play-all
    for i in 0..n {
        for j in 0..n {
            if i != j {
                matrix[i * n + j] = cmp(&vec[i], &vec[j]);
            }
        }
    }

    fn sort_scoring(indexes: &mut [usize], n: usize, matrix: &Vec<Ordering>) {
        if indexes.len() == 1 {
            return;
        }

        let mut scores = vec![0; n]; // square :(

        for i in indexes.iter() {
            for j in indexes.iter() {
                if i != j {
                    // lookup, dont compute again
                    let ord = matrix[*i * n + *j];
                    scores[*i] += match ord {
                        Ordering::Less => 2,
                        Ordering::Equal => 1,
                        Ordering::Greater => 0,
                    };
                    scores[*j] += match ord {
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
        let cmp = &mut |a: &u32, b: &u32| a.cmp(b);

        assert_eq!(rr_sort(&vec![2, 1], cmp), vec![&1, &2]);
        assert_eq!(rr_sort(&vec![3, 2, 1], cmp), vec![&1, &2, &3]);
        assert_eq!(rr_sort(&vec![2, 1, 2], cmp), vec![&1, &2, &2]);
    }

    #[test]
    fn symmetric_loose_ord_sorts_correctly() {
        let mut map = HashMap::new();
        map.insert((1, 2), Ordering::Less);
        map.insert((1, 3), Ordering::Less);
        map.insert((1, 4), Ordering::Greater);

        map.insert((2, 1), Ordering::Greater);
        map.insert((2, 3), Ordering::Less);
        map.insert((2, 4), Ordering::Greater);

        map.insert((3, 1), Ordering::Greater);
        map.insert((3, 2), Ordering::Greater);
        map.insert((3, 4), Ordering::Less);

        map.insert((4, 1), Ordering::Less);
        map.insert((4, 2), Ordering::Less);
        map.insert((4, 3), Ordering::Greater);

        let cmp = |a: &i32, b: &i32| map.get(&(*a, *b)).unwrap().clone();

        assert_eq!(rr_sort(&vec![2, 4, 1, 3], &cmp), vec![&4, &1, &2, &3]);
    }
}
