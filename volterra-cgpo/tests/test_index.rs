use volterra_cgpo::index::{si, vi};

#[test]
fn scalar_and_vector_indices_match_layout() {
    let ly = 5;
    assert_eq!(si(2, 3, ly), 2 * 5 + 3);
    assert_eq!(vi(2, 3, ly, 0), (2 * 5 + 3) * 2);
    assert_eq!(vi(2, 3, ly, 1), (2 * 5 + 3) * 2 + 1);
}
