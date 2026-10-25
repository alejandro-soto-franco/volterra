#!/bin/bash
# Rust -> Charon -> Aeneas -> Lean for volterra-core/src/fields/index.rs. Fails on a test
# failure, on drift between the Rust and the committed Lean translation, and on any axiom
# beyond Lean's standard three. Tools: the Aeneas nightly-2026.09.24-557f7a1 release at
# $AENEAS_HOME (default ~/local/aeneas), which bundles Charon at the commit Aeneas pins.
set -euo pipefail
cd "$(dirname "$0")"
A=${AENEAS_HOME:-$HOME/local/aeneas}

echo "== Rust tests"
(cd .. && cargo test --quiet -p volterra-core --lib fields::index)

echo "== Charon"
(cd index_core && "$A/charon" cargo --preset=aeneas --dest-file ../index_core.llbc)

echo "== Aeneas"
tmp=$(mktemp -d)
"$A/aeneas" -backend lean index_core.llbc -dest "$tmp" -split-files -loops-to-rec >/dev/null
rm -f index_core.llbc
for f in Funs.lean Types.lean; do
  diff -q "$tmp/$f" "lean/IndexCore/$f" >/dev/null || {
    rm -rf "$tmp"
    echo "generated $f differs from lean/IndexCore/$f: regenerate and re-prove"; exit 1; }
done
rm -rf "$tmp"

echo "== Lean"
(cd lean && lake build)
axioms=$(cd lean && lake env lean CheckAxioms.lean)
echo "$axioms"
if echo "$axioms" | grep -v "depends on axioms: \[propext, Classical.choice, Quot.sound\]$" | grep -q .; then
  echo "unexpected axiom or missing theorem"; exit 1
fi
echo "verified"
