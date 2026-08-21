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

//! **Regenerated 2026-08-20.** Two defects in `stokes_dec.rs` moved every wet
//! value: `velocity_from_psi` averaged directional derivatives instead of
//! solving for a gradient, which returned a quarter of the curl and never
//! converged; and `compute_vorticity_source_from_stress` took the circulation of
//! a piecewise-CONSTANT force around its own triangle, which is identically
//! zero, and so accumulated a source diverging as `1/h`. Both are now pinned
//! against exact solutions. `DRY_*` is unchanged, since the dry runner never
//! touches the flow.
//!
//! **Regenerated again, same day.** `advect_q` had the same defect as the curl:
//! it averaged directional derivatives over the incident edges and divided by
//! the valence, which is `u^T A_v grad Q` with `A_v = I/2` on an isotropic fan,
//! so it returned HALF of `u . grad Q` and held that factor under refinement.
//! `examples/dbg_advect.rs` measured 0.50 across a factor of eight in spacing
//! before, and 0.982 to 0.9997 after, and
//! `advection_recovers_the_directional_derivative_and_converges` pins both the
//! constant and the convergence. `advect_q_covariant` had it too. `DRY_*` is
//! again unchanged: the dry runner does not advect.

//!
//! **Regenerated 2026-08-20, a third time.** The biharmonic was driven with
//! `+curl f` where steady Stokes gives `-curl f`, so every velocity the wet
//! runners produced ran BACKWARDS. `DRY_*` is unchanged once more, since the dry
//! runner has no flow, and that is the signature the sign fix should leave: the
//! two wet blocks move and the dry one does not. See
//! `compute_vorticity_source_from_stress` for the derivation and for the three
//! tests that now pin the sign.

use cartan_dec::mesh::FlatMesh;
use cartan_dec::Operators;
use cartan_manifolds::euclidean::Euclidean;
use volterra_core::ActiveNematicParams;
use volterra_dec::QFieldDec;
use volterra_dec::{
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

const WET_Q1: &[u64] = &[4573740256338258859,13793761695944676397,4576670811807409380,4571243892156295667,4573423535788108759,4565557223608083069,13798525958558666742,4566933071088803945,13796654436591355122,4576072909032463982,4563307476080395507,4574589057509787638,4570281391410264059,13799431162183738188,13789976807606659755,4561023116014106212,13796538662003010687,13799640490954149643,4564364190495227955,13782331577951071061,13797704822886998838,4574852998530880481,4569018920361382974,4558142917502010889,4573794084829603021,];
const WET_Q2: &[u64] = &[13798211329734938041,13787635605961458555,4576278689150541707,4557617386998341324,4575199518220968366,13799251214452966632,4568391951644545431,4571664943396377195,13794994730241315693,4576674739976214046,4571592434102075175,4573219207348374359,13789398299111712858,4560242331366585680,4573475817732711726,4574655519621033837,13797494701637221076,13782610910015556533,4565939652600849627,13799666541565115505,13788601238611042900,4568608208607127810,4564902482932796778,4572491618042428764,13795898402529380568,];
const WET_MEAN_S: &[u64] = &[4579558413260827438,4579500457169706033,4579443000587757189,];

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

const CONF_Q1: &[u64] = &[4573740253404681459,13793761696690442846,4576670809560150379,4571243887271590884,4573423538071421652,4565557500930603306,13798525976657621326,4566933077241782335,13796654446594575144,4576072858123738014,4563307439326092684,4574589050602303087,4570281467368640695,13799431164981221069,13789976862315240258,4561023242588445408,13796538654884346580,13799640491706443371,4564364151483630894,13782331554151912096,13797704837962791749,4574852991811765626,4569018925271523286,4558142905026581891,4573794076689293346,];
const CONF_Q2: &[u64] = &[13798211329232134650,13787635595608935449,4576278687844831596,4557617354023398729,4575199522753191051,13799251259636609394,4568391858682954058,4571664976283995862,13794994753725053728,4576674696073096057,4571592416133267939,4573219228036327467,13789398253227815890,4560242120313790886,4573475802816098084,4574655558088574721,13797494691712216306,13782610984110235514,4565939675523179709,13799666525817018342,13788601235564070697,4568608200694827564,4564902475396655269,4572491623999528057,13795898390655280515,];
const CONF_MEAN_S: &[u64] = &[4579558413260827438,4579500457610074989,4579443001466426550,];

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
