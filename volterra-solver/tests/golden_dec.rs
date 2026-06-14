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

const WET_Q1: &[u64] = &[4573740310704840944,13793761695795606926,4576670795720192208,4571243901326501712,4573423530324809453,4565557280555473925,13798525956097786710,4566933066568303307,13796654441449420104,4576072896402363487,4563307495889088259,4574589061026007402,4570281379409910878,13799431164385668836,13789976834809061057,4561023026214060197,13796538660016974579,13799640488470290259,4564364223270284320,13782331556645804608,13797704781548553856,4574853041121484056,4569018900749605243,4558142947506940526,4573794115426718656,];
const WET_Q2: &[u64] = &[13798211347787699479,13787635726465111843,4576278680575099298,4557617372398081726,4575199505743585492,13799251218570546595,4568391944417532436,4571664939116264995,13794994734937210312,4576674733826582070,4571592432088756362,4573219203246724562,13789398305204071882,4560242331254028192,4573475805008989983,4574655495204985876,13797494704560601947,13782610875100570401,4565939621417583577,13799666559701734929,13788601234743045758,4568608244531530825,4564902502817986440,4572491603795010837,13795898430253855811,];
const WET_MEAN_S: &[u64] = &[4579558413260827438,4579500457694256539,4579443001613830205,];

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

const CONF_Q1: &[u64] = &[4573740255393456546,13793761699023887934,4576670810382156446,4571243888556676464,4573423541241993553,4565557207602440576,13798525954333833359,4566933070397407938,13796654440515755946,4576072903825742222,4563307486892863693,4574589060736512745,4570281378213012034,13799431163369607519,13789976834686468712,4561023024068034104,13796538659990521549,13799640488441307749,4564364223323729723,13782331554911901500,13797704781334827840,4574853041792751282,4569018900751672515,4558142946368997578,4573794115486239886,];
const CONF_Q2: &[u64] = &[13798211328165392189,13787635588980068606,4576278688320601063,4557617346383854938,4575199525599771963,13799251210936161392,4568391963129934777,4571664938301758970,13794994729691964198,4576674740240544575,4571592436168063947,4573219203179223442,13789398307875836077,4560242344346003285,4573475805035422943,4574655494754891976,13797494704625708350,13782610874857952382,4565939621908945058,13799666559494839553,13788601235307196900,4568608245120181406,4564902502823343733,4572491604274883692,13795898429539399493,];
const CONF_MEAN_S: &[u64] = &[4579558413260827438,4579500457253100419,4579443000750509955,];

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
