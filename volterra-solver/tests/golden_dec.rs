//! Golden bit-for-bit oracles for the three DEC runners (dry, wet closed,
//! wet confined).
//!
//! These tests pin the full numerical output of each runner on a fixed, small,
//! fully deterministic fixture (a flat unit-square DEC mesh + a seeded random
//! initial Q field). They capture the final `q1`/`q2` vectors and the
//! per-snapshot `mean_s` trajectory as raw IEEE-754 bit patterns
//! (`f64::to_bits`) so that any drift in the refactored runners is caught
//! exactly, not within a tolerance.
//!
//! Fixture rationale: the existing `test_runner_dec*` integration tests all use
//! `FlatMesh::unit_square_grid` + `Euclidean<2>` + `Operators::from_mesh`, which
//! is the smallest fully deterministic DEC setup available (no Sphere operator
//! construction, no floating-point order dependence from mesh generation). We
//! reuse it here at grid resolution 4 (25 vertices) with 4 steps, snap_every 2.

use cartan_dec::mesh::FlatMesh;
use cartan_dec::Operators;
use cartan_manifolds::euclidean::Euclidean;
use volterra_core::ActiveNematicParams;
use volterra_dec::QFieldDec;
use volterra_solver::{
    run_dry_active_nematic_dec, run_wet_active_nematic_dec,
    run_wet_active_nematic_dec_confined,
};

/// Build the shared small fixture: a 4x4 flat unit-square grid, Euclidean<2>
/// operators, and a seeded random initial Q field.
fn fixture() -> (FlatMesh, Operators<Euclidean<2>, 3, 2>, ActiveNematicParams, QFieldDec) {
    let mesh = FlatMesh::unit_square_grid(4);
    let manifold = Euclidean::<2>;
    let ops = Operators::from_mesh(&mesh, &manifold);

    let mut params = ActiveNematicParams::default_test();
    params.dt = 0.005;

    let nv = mesh.n_vertices();
    let q0 = QFieldDec::random_perturbation(nv, 0.01, 42);
    (mesh, ops, params, q0)
}

/// Number of steps and snapshot cadence for all three oracles.
const N_STEPS: usize = 4;
const SNAP_EVERY: usize = 2;

/// Print a capture block (used once to generate the golden consts, then the
/// asserts below take over). Left in as a helper for re-capture if the fixture
/// ever changes intentionally.
#[allow(dead_code)]
fn dump(label: &str, q: &QFieldDec, mean_s: &[f64]) {
    eprintln!("// === {label} ===");
    eprint!("const {}_Q1: &[u64] = &[", label);
    for v in &q.q1 {
        eprint!("{},", v.to_bits());
    }
    eprintln!("];");
    eprint!("const {}_Q2: &[u64] = &[", label);
    for v in &q.q2 {
        eprint!("{},", v.to_bits());
    }
    eprintln!("];");
    eprint!("const {}_MEAN_S: &[u64] = &[", label);
    for v in mean_s {
        eprint!("{},", v.to_bits());
    }
    eprintln!("];");
}

fn assert_bits(label: &str, got: &[f64], expected: &[u64]) {
    assert_eq!(got.len(), expected.len(), "{label}: length mismatch");
    for (i, (g, e)) in got.iter().zip(expected).enumerate() {
        assert_eq!(
            g.to_bits(),
            *e,
            "{label}[{i}] drifted: got {g} (bits {}) expected bits {e}",
            g.to_bits()
        );
    }
}

// ── DRY ──────────────────────────────────────────────────────────────────────

const DRY_Q1: &[u64] = &[4555624793145840772,13785111507457842853,4569893760655585152,4567160511478147045,4571794405318749252,13767526792487686252,13788643058191708624,4559686910740573276,13786210678156354520,4567227091714007282,4563884619253302197,4564079435287163672,4543647545767508344,13792833733630019152,13788465251621107720,13785789884366390797,13788586035597395601,13792154605544168183,13786391579053196098,13777123672264539998,13790329036076100774,4558672923500548286,13775785347466553958,4557832200980663532,4563239419742634712,];
const DRY_Q2: &[u64] = &[13791698148543705349,4555180498063969417,4571160260050221872,4566737779565764986,4572684467463141113,13790255866542177347,4562975625863034268,4567709993449665660,4560037188808070116,4571550270443572301,4567252833120426307,4566921743255684865,4558856546651371070,4560772815278495260,4567414295013026671,4565765787807204239,13783133216487793395,13776915827689670742,4553634136547445346,13790617181696865192,4558544091072302640,4552820330495028678,4561156504361357299,4564351005606737145,13787615342129156952,];
const DRY_MEAN_S: &[u64] = &[4579558413260827438,4575666838901045153,4572228836388153058,];

#[test]
fn golden_dry_dec() {
    let (_mesh, ops, params, q0) = fixture();
    let (q_fin, stats) =
        run_dry_active_nematic_dec(&q0, &params, &ops, None, N_STEPS, SNAP_EVERY);
    let mean_s: Vec<f64> = stats.iter().map(|s| s.mean_s).collect();
    dump("DRY", &q_fin, &mean_s);
    assert_bits("DRY_Q1", &q_fin.q1, DRY_Q1);
    assert_bits("DRY_Q2", &q_fin.q2, DRY_Q2);
    assert_bits("DRY_MEAN_S", &mean_s, DRY_MEAN_S);
}

// ── WET (closed) ───────────────────────────────────────────────────────────

const WET_Q1: &[u64] = &[4573740314595536615,13793761691154987748,4576670801383931018,4571243910548835219,4573423523135674663,4565557293452421487,13798525956726337340,4566933077108072268,13796654443402561560,4576072885856882033,4563307490502315960,4574589058746440470,4570281387787912475,13799431165331980075,13789976837013109636,4561023005208986332,13796538661942754103,13799640487847975292,4564364233009114426,13782331542891184937,13797704777397497713,4574853040483124927,4569018903572878589,4558142963867533390,4573794124916470292,];
const WET_Q2: &[u64] = &[13798211351755037192,13787635746519233778,4576278684972358929,4557617422789307039,4575199492839700154,13799251222687686278,4568391935857381130,4571664947931900691,13794994741563030822,4576674723802346989,4571592428811750915,4573219203366419775,13789398297116708731,4560242281076818664,4573475796396293672,4574655490118704817,13797494706739217687,13782610860600355658,4565939615832235121,13799666562397257773,13788601232442198578,4568608245775119031,4564902507000336103,4572491600857756481,13795898438900266874,];
const WET_MEAN_S: &[u64] = &[4579558413260827438,4579500457880863532,4579443001981632198,];

#[test]
fn golden_wet_dec() {
    let (mesh, ops, mut params, q0) = fixture();
    // Activate the flow so the Stokes solve is exercised (zeta != 0).
    params.zeta_eff = 0.5;
    params.dt = 0.00005;
    let (q_fin, stats) =
        run_wet_active_nematic_dec(&q0, &params, &ops, &mesh, None, N_STEPS, SNAP_EVERY)
            .expect("wet dec runner");
    let mean_s: Vec<f64> = stats.iter().map(|s| s.mean_s).collect();
    dump("WET", &q_fin, &mean_s);
    assert_bits("WET_Q1", &q_fin.q1, WET_Q1);
    assert_bits("WET_Q2", &q_fin.q2, WET_Q2);
    assert_bits("WET_MEAN_S", &mean_s, WET_MEAN_S);
}

// ── WET confined ─────────────────────────────────────────────────────────────

const CONF_Q1: &[u64] = &[4573740257167848561,13793761693230482068,4576670809992160890,4571243889854186185,4573423537122797401,4565557270298245385,13798525956568105605,4566933070661974086,13796654445761384060,4576072891273497515,4563307487465005112,4574589057704625601,4570281385257397144,13799431164308584803,13789976837853742319,4561023004739801005,13796538661920924092,13799640487712518312,4564364237420450800,13782331541592967966,13797704777667297977,4574853041284391691,4569018905442595712,4558142964195038650,4573794125774285996,];
const CONF_Q2: &[u64] = &[13798211327860937378,13787635603082853686,4576278688366426215,4557617372158142460,4575199521605685092,13799251221748789393,4568391935944654278,4571664946154705866,13794994736813633442,4576674729078069330,4571592432125603112,4573219203410624148,13789398299667950452,4560242305803691114,4573475796172504734,4574655490589945520,13797494706404429909,13782610858064690537,4565939615208018748,13799666563344732614,13788601232845579805,4568608246547198809,4564902507980289319,4572491600843133446,13795898439232215228,];
const CONF_MEAN_S: &[u64] = &[4579558413260827438,4579500457315015374,4579443000871712390,];

#[test]
fn golden_confined_dec() {
    let (mesh, ops, mut params, q0) = fixture();
    params.zeta_eff = 0.5;
    params.dt = 0.00005;
    // A small, fixed set of boundary vertices for the Dirichlet stream-function
    // constraint. On the flat 4x4 grid these are valid vertex indices; the test
    // only pins the runner output, not a physical no-slip property.
    let boundary_vertices: Vec<usize> = vec![0, 1, 2, 3, 4];
    let (q_fin, stats) = run_wet_active_nematic_dec_confined(
        &q0,
        &params,
        &ops,
        &mesh,
        &boundary_vertices,
        None,
        N_STEPS,
        SNAP_EVERY,
    )
    .expect("wet confined dec runner");
    let mean_s: Vec<f64> = stats.iter().map(|s| s.mean_s).collect();
    dump("CONF", &q_fin, &mean_s);
    assert_bits("CONF_Q1", &q_fin.q1, CONF_Q1);
    assert_bits("CONF_Q2", &q_fin.q2, CONF_Q2);
    assert_bits("CONF_MEAN_S", &mean_s, CONF_MEAN_S);
}
