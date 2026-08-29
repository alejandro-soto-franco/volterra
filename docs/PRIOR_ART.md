# Prior art and benchmark targets

`SUBSUMPTION.md` compares volterra against three codes from one group. This
document is the wider field: every public code found that solves the same
equations, what each one is, and what it can be benchmarked on.

The equations are the Beris-Edwards `Q`-tensor evolution coupled to
Navier-Stokes through an active stress. A code qualifies here when it
integrates both in time. An energy minimiser has no velocity field and no
physical time, so it benchmarks the free-energy half alone; an analysis suite
reads fields it never produced. Both appear below, labelled as such.

## Solvers of the same equations

| Code | Language | Licence | Last push | Method | Scope |
|---|---|---|---|---|---|
| [ludwig-cf/ludwig](https://github.com/ludwig-cf/ludwig) | C, MPI + CUDA | custom | 2026-08-29 | lattice Boltzmann | 3D complex fluids, LC hydrodynamics, colloids |
| [rmislam/lattice-boltzmann-active-nematics](https://github.com/rmislam/lattice-boltzmann-active-nematics) | C | none stated | 2025-02-23 | lattice Boltzmann | 2D active nematics, Beris-Edwards |
| [whpy/QActFlow](https://github.com/whpy/QActFlow) | CUDA | none stated | 2023-11-30 | pseudo-spectral, cuFFT | 2D `Q`-tensor, GPU |
| [tomidiy/ActiveNematics2D](https://github.com/tomidiy/ActiveNematics2D) | C | none stated | 2025-06-13 | finite difference | 2D Landau-de Gennes |
| [KamilFedorowicz/BE_model](https://github.com/KamilFedorowicz/BE_model) | C | none stated | 2022-03-07 | finite difference | Beris-Edwards, small |
| [Brandonkl/open-zetar](https://github.com/Brandonkl/open-zetar) | Python notebooks | none stated | 2026-06-01 | finite difference | 2D and 3D Beris-Edwards + Navier-Stokes |
| [liutc137/ActiveNematicDynamics](https://github.com/liutc137/ActiveNematicDynamics) | Python | none stated | 2025-09-16 | finite difference | 2D active nematics |
| [CunchengZhu/Riemannian-active-nematics-2024](https://github.com/CunchengZhu/Riemannian-active-nematics-2024) | mixed | MIT | 2025-02-06 | surface discretisation | active nematics on curved surfaces |
| `flow-solver.py` | Python | unreleased | local | finite difference | 2D confined, the CGPO reference |

## Solvers of part of the problem

| Code | Language | Licence | What it does |
|---|---|---|---|
| [sussmanLab/open-Qmin](https://github.com/sussmanLab/open-Qmin) | C++, CUDA | custom | Landau-de Gennes energy minimiser. No velocity field, no physical time; every updater descends from `equationOfMotion` and its `Time` is a step counter. Benchmarks the free energy and the defect detector, never the hydrodynamics. |
| [YingyouMa/ActiveBE_Validator](https://github.com/YingyouMa/ActiveBE_Validator) | notebooks | none stated | Residual checks for the active Beris-Edwards equations. A correctness oracle rather than a solver, and the most directly useful thing here for validation. |
| [joshichaitanya3/actnempy](https://github.com/joshichaitanya3/actnempy) | Python | see repo | Analysis of 2D active-nematic data. Reads fields; produces none. |

## Rust

No other Rust code in this class was found. Of 225 GitHub repositories matching
`nematic`, three are Rust: `morphym/nematic` is a WordNet sentence generator
whose name collides, `jonaspleyer/cr_nematic_structure` is an agent-based
bacterial model under `cellular_raza` with no continuum field, and the third is
this one. Repository searches for `Beris-Edwards`, `Landau-de Gennes` and
`liquid crystal` restricted to Rust return nothing, and crates.io lists only
`volterra-solver`.

The search sees public GitHub repositories and matches names, descriptions and
topics. A solver on GitLab, in a Zenodo deposit, or in a paper's supplementary
material would not appear in it, so the finding is "none indexed", never "none
exists".

## Licences

Six of the codes above state no licence at all, which under the Berne
Convention means all rights reserved: they may be read and run, and their
results may be cited, but no line may be copied into volterra and no fork may
be redistributed. Ludwig and open-Qmin state custom terms that need reading
before either is vendored. Only the Riemannian active nematics repository
has a licence that permits reuse outright. Benchmarking against a
no-licence code is fine; taking anything from one is not.

## Benchmark targets

Three separate questions, and different codes answer each.

**Correctness of the hydrodynamics.** `open-zetar` and
`lattice-boltzmann-active-nematics` integrate the same equations on a periodic
square, which is exactly volterra-fd's domain. A shared configuration at one
`ell_a` gives a field-by-field comparison. `ActiveBE_Validator` is better still
where it applies, since it checks residuals rather than agreement with another
code's discretisation error.

**Correctness of the free energy and the defect detector.** open-Qmin minimises
the same Landau-de Gennes functional on a lattice, so an equilibrium
configuration and its defect set transfer directly, with no hydrodynamics
involved on either side.

**Throughput.** Ludwig is the standard to beat on a large 3D domain and QActFlow
on a 2D GPU one. Neither comparison means anything without matched physics:
a lattice Boltzmann step and a projection-method step do different work, and the
resolution, the timestep and the convergence criterion of the pressure solve all
have to be stated for a rate to be comparable. State them, or report a
time-to-solution at a fixed accuracy instead of a step rate.

## Curved geometry

`Riemannian-active-nematics-2024` is the closest published comparison for
`volterra-dec`, which is the only other code here that runs active nematics on a
surface rather than a flat periodic box. It has an MIT licence, so unlike
most of the list its code may be reused as well as run.
