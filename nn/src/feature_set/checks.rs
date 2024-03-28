use super::FeatureSet;
use shakmaty::{fen::Fen, Chess, Color, Position};

#[allow(dead_code)]
pub(super) fn sanity_checks(feature_set: &dyn FeatureSet) {
    const FENS: [&str; 20] = [
        "4nrk1/3q1pp1/2n1p1p1/8/1P2Q3/7P/PB1N1PP1/2R3K1 w - - 5 26",
        "5r2/1p2ppkp/p2p1nP1/qn6/4P3/2r2B2/1PPQ1PP1/2KR3R w - - 0 21",
        "2r2rk1/p2nqp2/1p1p1p1B/1bp5/3N4/8/PPPK1PPP/R2Q3R b - - 1 17",
        "r2q1b1r/p3kppp/2Q1pn2/3p4/3P4/2N1PN2/PPn2PPP/R1B2RK1 w - - 1 11",
        "rn2k2r/pp1qbppp/2p2n2/3p1b2/3P4/P1NBP3/1P3PPP/R1BQK1NR b KQkq - 1 9",
        "r2q1rk1/1pp1b1pp/p2p4/2PPpb2/PP5N/2N1B2P/5PP1/R2Q1RK1 b - - 0 16",
        "r2q1rk1/ppp2p2/4p1np/5p2/3P4/2PBP1Q1/P4PPP/R4RK1 w - - 0 16",
        "r3k2r/1pp2ppp/2nb1n2/pB1p4/P3pP1q/1P2P2P/1BPPQ2P/RN3K1R b kq - 0 12",
        "2r5/4r1kp/2pR2p1/p1P2p2/P1P1p3/4K1P1/7P/8 w - f6 0 34",
        "3rk1nr/1bqnppbp/pppp2p1/5P2/2PPP3/2NBBN2/PP4PP/R2QK2R w KQk - 1 11",
        "1nkr3r/pp2qpp1/2bp4/3Bp1bp/P1P1P3/5N2/5PPP/R2Q1RK1 b - - 0 17",
        "r1bqr1k1/pp3pbp/2n2np1/2pp4/4p3/PP1PP1PP/1BPNNPB1/R2Q1RK1 w - - 0 12",
        "3r1rk1/1p2q1p1/p1nb3p/2p1p1N1/3pP2P/P4P2/1PPB2P1/R2QR1K1 b - - 0 20",
        "r1bq1rk1/p2n1pbp/Ppp3p1/3p4/1PBP4/R3BN2/2P2PPP/3Q1RK1 w - - 0 15",
        "4r1k1/ppq2ppp/2pbr3/3p4/3Q4/1P3P1P/PBP3P1/2KR3R b - - 4 19",
        "r3kb1r/pppqnp1p/2n1p1p1/3p1b2/3P1P2/2P1PN2/PPB3PP/RNBQ1RK1 b kq - 3 8",
        "rn1q1rk1/pp3ppp/5n2/3p1b2/1b1P4/P1N2N2/1P2BPPP/R1BQR1K1 b - - 0 11",
        "8/p5Rp/8/4k3/8/4P2P/P1P5/2K5 w - - 1 31",
        "rn2k2r/pp2npp1/2pp3p/1P2p3/2BbP2q/P1NQ1P2/1BP2P1P/2KR3R b kq - 2 15",
        "r2q1rk1/1b1nbpp1/p1pp1n1p/Pp2p3/1P1PP3/1BP1BN1P/3N1PP1/R2QK2R b KQ - 0 13",
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

        //if feature_set.requires_refresh(&m) {
        //    continue;
        //}

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
        assert!(!actual.contains(x), "added feature {} is in actual", x);
        assert!(!removed.contains(x), "added feature {} is in remove", x);
    }

    for x in removed {
        assert!(actual.contains(x), "tried to remove a feature not present");
    }

    result
}
