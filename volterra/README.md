# volterra

Covariant active nematics simulation on Riemannian manifolds, in Rust.

[![crates.io](https://img.shields.io/crates/v/volterra.svg)](https://crates.io/crates/volterra)
[![docs.rs](https://docs.rs/volterra/badge.svg)](https://docs.rs/volterra)

This is the facade crate: it re-exports the volterra workspace so downstream
users can depend on a single crate.

## Overview

`volterra` solves the Beris-Edwards nematohydrodynamics equations for active
liquid crystals on curved spaces. It is built on the
[cartan](https://crates.io/crates/cartan) Riemannian geometry library and uses
discrete exterior calculus (DEC) to discretize covariant differential operators
on arbitrary triangle meshes. The primary objects of study are the topological
defects of the nematic order parameter and their dynamics in confined and
curved geometries.

The facade re-exports:

- `volterra-core`: parameters, error types, and shared traits.
- `volterra-fields`: Q-tensor and velocity field containers.
- `volterra-solver`: time integration, Stokes solvers, and defect tracking.

Parallel transport of the spin-2 Q-tensor is carried covariantly through the
Levi-Civita connection, with the transport represented either as SO(d) matrices
or as geometric-algebra rotors from `cartan-core`.

## Python

A Python distribution is published separately on PyPI as `volterra-nematic`
(`pip install volterra-nematic`), imported as `import volterra`.

## License

[MIT](../LICENSE-MIT)
