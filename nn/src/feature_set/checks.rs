use crate::nnue::model::NnueModel;

use super::FeatureSet;
use shakmaty::{fen::Fen, Chess, Color, Position};

pub(super) fn sanity_checks(feature_set: &dyn FeatureSet) {
    const FENS: [&str; 2] = [
        "4nrk1/3q1pp1/2n1p1p1/8/1P2Q3/7P/PB1N1PP1/2R3K1 w - - 5 26",
        "5r2/1p2ppkp/p2p1nP1/qn6/4P3/2r2B2/1PPQ1PP1/2KR3R w - - 0 21",
    ];

    for fen in FENS {
        let fen: Fen = fen.parse().unwrap();
        let pos: Chess = fen.into_position(shakmaty::CastlingMode::Standard).unwrap();

        check_flipped(&pos, feature_set);
        check_changed(&pos, Color::White, feature_set);
        check_changed(&pos, Color::Black, feature_set);
    }
}

/// Check that features are correctly the same when board if flipped
fn check_flipped(pos: &Chess, feature_set: &dyn FeatureSet) {
    let board_orig = pos.board().clone();

    let mut board_flip = board_orig.clone();
    board_flip.flip_vertical();
    board_flip.swap_colors();

    let mut feat_orig_white = vec![];
    let mut feat_orig_black = vec![];
    let mut feat_flip_white = vec![];
    let mut feat_flip_black = vec![];

    feature_set.active_features(&board_orig, Color::White, &mut feat_orig_white);
    feature_set.active_features(&board_orig, Color::Black, &mut feat_orig_black);
    feature_set.active_features(&board_flip, Color::White, &mut feat_flip_white);
    feature_set.active_features(&board_flip, Color::Black, &mut feat_flip_black);

    feat_orig_white.sort();
    feat_orig_black.sort();
    feat_flip_white.sort();
    feat_flip_black.sort();

    assert_eq!(feat_orig_white, feat_flip_black);
    assert_eq!(feat_orig_black, feat_flip_white);
}

/// Check changed_features assuming that active_features is right
fn check_changed(pos: &Chess, perspective: Color, feature_set: &dyn FeatureSet) {
    let mut pos_features = vec![];
    feature_set.active_features(pos.board(), perspective, &mut pos_features);

    for m in pos.legal_moves() {
        let mut pos_moved = pos.clone();
        pos_moved.play_unchecked(&m);

        let mut pos_moved_features = vec![];
        feature_set.active_features(pos_moved.board(), perspective, &mut pos_moved_features);

        let mut added_features = vec![];
        let mut removed_features = vec![];

        feature_set.changed_features(
            pos.board(),
            &m,
            perspective,
            &mut added_features,
            &mut removed_features,
        );
        let mut applied = apply_changes(&pos_features, &added_features, &removed_features);

        pos_moved_features.sort();
        applied.sort();

        assert_eq!(pos_moved_features, applied);
    }
}

/// Also checks that added do not exist in actual and removed do exist
fn apply_changes(actual: &Vec<u16>, added: &Vec<u16>, removed: &Vec<u16>) -> Vec<u16> {
    let mut result = vec![];

    for &x in actual {
        if !removed.contains(&x) {
            result.push(x);
        }
    }

    for x in added {
        result.push(*x);
        assert!(!actual.contains(x), "added feature is in actual");
        assert!(!removed.contains(x), "added feature is in remove");
    }

    for x in removed {
        assert!(actual.contains(x), "tried to remove a feature not present");
    }

    result
}
