//! What a relaxed director field in the chain actually does.
//!
//! `tests/lobe_chain.rs` measures the charge the wall imposes, which is a
//! calculation on the boundary. This runs the interior: a random director field
//! relaxed inside the chain, with the cores counted where they settle.
//!
//! Total charge alone would prove nothing here. It is fixed by the anchoring
//! and would read correctly off an unrelaxed field, and off a diverged one: an
//! earlier version of this run blew the free energy from -5e6 to infinity in
//! seven steps and still reported the right total. The free energy is therefore
//! checked to have fallen, and the cores are counted rather than summed.

use volterra_dec::confined::{MeshOpts, confined_mesh};
use volterra_dec::confined_ldg::LdgProblem;
use volterra_dec::lobe_chain::LobeChain;
use volterra_dec::nematic_params::NematicParams;

/// Coherence length, at two and a half bulk elements, so a core is resolved.
const NCL: f64 = 0.25;
const H_BULK: f64 = 0.10;
/// Below the 0.29 that separates two cores across a neck, so they stay apart.
const MERGE: f64 = 0.05;

struct Relaxed {
    imposed: f64,
    positive: usize,
    negative: usize,
    total: f64,
    f_start: f64,
    f_end: f64,
    xs: Vec<f64>,
}

fn relax(n: usize, seed: u64) -> Relaxed {
    let chain = LobeChain::new(n, 0.25);
    let cm = confined_mesh(
        chain,
        MeshOpts {
            h_bulk: H_BULK,
            h_min: 0.008,
            cusp_edge: 0.01,
            ..Default::default()
        },
    );
    let (imposed, _w, _b) = cm.imposed_charge(1.0);
    let params = NematicParams::from_length_scales(f64::INFINITY, NCL, 128);
    let prob = LdgProblem::new(cm, params, 1.0).expect("the chain mesh assembles");
    let mut q = prob.random_state(seed);
    let f_start = prob.free_energy(&q);
    // 1e-4 is stable and converged: the same endpoint comes back at 1e-5 and
    // 1e-6, and 1e-3 diverges at this coherence length.
    prob.relax(&mut q, 1e-4, 20_000, 1e-10);
    let f_end = prob.free_energy(&q);
    let (positive, negative, total, cores) = prob.defect_summary(&q, MERGE);
    let mut xs: Vec<f64> = cores.iter().map(|c| c.0).collect();
    xs.sort_by(|a, b| a.partial_cmp(b).expect("core positions are finite"));
    Relaxed {
        imposed,
        positive,
        negative,
        total,
        f_start,
        f_end,
        xs,
    }
}

#[test]
fn the_relaxation_lowers_the_free_energy() {
    let r = relax(2, 1);
    assert!(
        r.f_end < r.f_start,
        "the free energy rose, {:.4e} to {:.4e}",
        r.f_start,
        r.f_end
    );
    assert!(
        r.f_end.is_finite(),
        "the relaxation diverged, ending at {}",
        r.f_end
    );
}

#[test]
fn each_lobe_settles_two_positive_half_cores() {
    for n in 1..=3 {
        for seed in [1_u64, 7, 23] {
            let r = relax(n, seed);
            assert!(
                r.f_end.is_finite() && r.f_end < r.f_start,
                "n = {n} seed = {seed} did not relax"
            );
            assert!(
                (r.total - n as f64).abs() < 1e-9,
                "n = {n} seed = {seed}: total {} against imposed {}",
                r.total,
                r.imposed
            );
            assert_eq!(
                r.positive,
                2 * n,
                "n = {n} seed = {seed}: expected two cores per lobe, got {} at x = {:?}",
                r.positive,
                r.xs
            );
            assert_eq!(
                r.negative, 0,
                "n = {n} seed = {seed} settled a negative core"
            );
        }
    }
}

#[test]
fn the_cores_pin_at_the_necks() {
    // A re-entrant cusp is where the wall anchoring turns the director hardest,
    // so the cores sit at the necks rather than in the middle of a lobe. That
    // is what makes the neck width the knob that sets their separation.
    let n = 3;
    let chain = LobeChain::new(n, 0.25);
    let lobe = chain.length() / n as f64;
    let r = relax(n, 1);
    assert_eq!(r.xs.len(), 2 * n);

    // The interior necks sit at multiples of the lobe length.
    for m in 1..n {
        let neck_x = m as f64 * lobe;
        let near =
            r.xs.iter()
                .filter(|&&x| (x - neck_x).abs() < 0.25 * lobe)
                .count();
        assert_eq!(
            near, 2,
            "the neck at x = {neck_x:.2} should have a core each side; cores at {:?}",
            r.xs
        );
    }
}

#[test]
fn the_merge_radius_is_what_decides_the_count_not_the_physics() {
    // The count is a statement about the merge radius and the total is not.
    // Two cores across a neck sit about 0.29 apart, so a radius above that
    // loses one while the winding sum keeps it. This is the trap that made an
    // earlier reading of this run report three cores where there are four.
    let chain = LobeChain::new(2, 0.25);
    let cm = confined_mesh(
        chain,
        MeshOpts {
            h_bulk: H_BULK,
            h_min: 0.008,
            cusp_edge: 0.01,
            ..Default::default()
        },
    );
    let params = NematicParams::from_length_scales(f64::INFINITY, NCL, 128);
    let prob = LdgProblem::new(cm, params, 1.0).expect("assembles");
    let mut q = prob.random_state(1);
    prob.relax(&mut q, 1e-4, 20_000, 1e-10);

    let (tight, _n1, total_tight, _c) = prob.defect_summary(&q, 0.05);
    let (loose, _n2, total_loose, _c) = prob.defect_summary(&q, 0.75);
    assert_eq!(tight, 4, "a tight radius resolves both cores at each neck");
    assert_eq!(loose, 3, "a loose radius merges the pair across the neck");
    assert!(
        (total_tight - total_loose).abs() < 1e-12,
        "the winding total must not depend on the merge radius: {total_tight} against {total_loose}"
    );
}
