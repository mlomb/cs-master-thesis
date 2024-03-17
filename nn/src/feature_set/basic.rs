use crate::feature_set::FeatureSet;
use shakmaty::{Board, Color, Move, Role};

/// The basic feature set
/// Tuple: <piece_square, piece_role>
pub struct Basic;

impl Basic {
    pub fn new() -> Self {
        Basic
    }
}

impl FeatureSet for Basic {
    fn num_features(&self) -> usize {
        64 * 6 * 2 // 768
    }

    fn requires_refresh(&self, _move: &Move) -> bool {
        // this feature set does not require refresh, its very simple
        false
    }

    fn active_features(&self, board: &Board, perspective: Color, features: &mut Vec<u16>) {
        assert!(features.is_empty());

        for (square, piece) in board.clone().into_iter() {
            let channel = if piece.color == perspective {
                match piece.role {
                    Role::Pawn => 0,
                    Role::Knight => 1,
                    Role::Bishop => 2,
                    Role::Rook => 3,
                    Role::Queen => 4,
                    Role::King => 5,
                }
            } else {
                match piece.role {
                    Role::Pawn => 6,
                    Role::Knight => 7,
                    Role::Bishop => 8,
                    Role::Rook => 9,
                    Role::Queen => 10,
                    Role::King => 11,
                }
            };

            features.push(channel * 64 + square as u16);
        }
    }

    fn changed_features(
        &self,
        board: &Board,
        _move: &Move,
        perspective: Color,
        added_features: &mut Vec<u16>,
        removed_features: &mut Vec<u16>,
    ) {
    }
}

#[cfg(test)]
mod tests {
    use crate::nnue::model::NnueModel;

    use super::FeatureSet;
    use super::*;
    use shakmaty::{fen::Fen, Chess, Position};

    #[test]
    fn test_basic_feature_set() {
        let fen: Fen = "4nrk1/3q1pp1/2n1p1p1/8/1P2Q3/7P/PB1N1PP1/2R3K1 w - - 5 26"
            .parse()
            .unwrap();
        let pos: Chess = fen.into_position(shakmaty::CastlingMode::Standard).unwrap();

        let basic: Box<dyn FeatureSet> = Box::new(Basic::new());
        let mut features_white = Vec::new();
        let mut features_black = Vec::new();

        basic.active_features(pos.board(), Color::White, &mut features_white);
        basic.active_features(pos.board(), Color::Black, &mut features_black);

        features_white.sort();
        features_black.sort();

        let mut buffer = Vec::new();
        let mut cursor = std::io::Cursor::new(&mut buffer);
        basic.encode(pos.board(), Color::White, &mut cursor);
        basic.encode(pos.board(), Color::Black, &mut cursor);

        println!("features_white: {:?}", features_white);
        println!("features_black: {:?}", features_black);
        println!("buffer: {:?}", buffer);

        let mut nnue_model =
            NnueModel::load("/mnt/c/Users/mlomb/Desktop/Tesis/cs-master-thesis/test_model.nn")
                .unwrap();

        nnue_model.refresh_accumulator(features_white.as_slice(), Color::White);
        nnue_model.refresh_accumulator(features_black.as_slice(), Color::Black);

        println!("model output: {:?}", nnue_model.forward(Color::White));
    }
}
