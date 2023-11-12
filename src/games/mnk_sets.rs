/// Returns a list of all winning sets (as bitsets) for a game of size mxn with k in a row to win
pub(super) fn mnk_winning_sets(m: usize, n: usize, k: usize) -> Vec<u64> {
    assert!(m * n <= 64);
    assert!(k <= m && k <= n);

    let mut sets = Vec::new();

    // horizontal
    for i in 0..m {
        for j in 0..n - k + 1 {
            let mut set = 0;
            for l in 0..k {
                set |= 1 << (i * n + j + l);
            }

            sets.push(set);
        }
    }

    // vertical
    for i in 0..m - k + 1 {
        for j in 0..n {
            let mut set = 0;
            for l in 0..k {
                set |= 1 << ((i + l) * n + j);
            }

            sets.push(set);
        }
    }

    // diagonal
    for i in 0..m - k + 1 {
        for j in 0..n - k + 1 {
            let mut set = 0;
            for l in 0..k {
                set |= 1 << ((i + l) * n + j + l);
            }

            sets.push(set);
        }
    }

    // anti-diagonal
    for i in 0..m - k + 1 {
        for j in k - 1..n {
            let mut set = 0;
            for l in 0..k {
                set |= 1 << ((i + l) * n + j - l);
            }

            sets.push(set);
        }
    }

    sets
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn winning_set_tictactoe() {
        let ttt = mnk_winning_sets(3, 3, 3);
        assert_eq!(ttt.len(), 8);
        assert!(ttt.contains(&0b111_000_000));
        assert!(ttt.contains(&0b000_111_000));
        assert!(ttt.contains(&0b000_000_111));
        assert!(ttt.contains(&0b100_100_100));
        assert!(ttt.contains(&0b010_010_010));
        assert!(ttt.contains(&0b001_001_001));
        assert!(ttt.contains(&0b100_010_001));
        assert!(ttt.contains(&0b001_010_100));
    }

    #[test]
    fn winning_set_connect4() {
        let c4 = mnk_winning_sets(6, 7, 4);
        assert_eq!(c4.len(), 69); // yeah, no kidding
    }
}
