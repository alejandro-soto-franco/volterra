/-
Specifications of volterra's periodic grid index maps, `volterra-core/src/fields/index.rs`,
as Aeneas translates them in `IndexCore/`.

Each theorem covers every grid whose vertex count fits in `usize`, on 32-bit and 64-bit
targets alike. Under that hypothesis no function panics or overflows, and each returns
the closed form stated. The signed index depends on each coordinate only through its
Euclidean residue and has period `nx` in `i` and `ny` in `j`.
-/
import Aeneas
import IndexCore
open Aeneas Aeneas.Std Result

theorem hcast_i64_val (n : Usize) (h : (n.val : Int) ≤ I64.max) :
    (UScalar.hcast .I64 n).val = n.val := by
  rw [UScalar.hcast_val_eq]
  rw [I64.max_eq] at h
  apply Int.bmod_eq_of_le
  all_goals simp
  all_goals omega

theorem hcast_usize_val (x : I64) (h0 : 0 ≤ x.val) (h1 : x.val ≤ Usize.max) :
    ((IScalar.hcast .Usize x).val : Int) = x.val := by
  rw [IScalar.hcast_val_eq]
  have hs : (Usize.max : Int) + 1 = 2 ^ UScalarTy.Usize.numBits := by
    simp [Usize.max, Usize.numBits]
  rw [Int.emod_eq_of_lt h0 (by rw [← hs]; omega)]
  omega

/-- The truncated remainder, corrected by `b` when negative, is the Euclidean one. -/
theorem tmod_fix (a b : Int) (hb : 0 < b) :
    (if Int.tmod a b < 0 then Int.tmod a b + b else Int.tmod a b) = a % b := by
  rw [Int.emod_eq_tmod]
  have hab : (b.natAbs : Int) = b := Int.natAbs_of_nonneg hb.le
  by_cases ha : 0 ≤ a
  · have := Int.tmod_nonneg b ha
    simp [ha]; omega
  · by_cases hd : b ∣ a
    · have : Int.tmod a b = 0 := Int.tmod_eq_zero_of_dvd hd
      simp [this, hd]
    · have h1 : Int.tmod a b ≤ 0 := by
        have := Int.tmod_nonneg b (show 0 ≤ -a by omega)
        rw [Int.neg_tmod] at this; omega
      have h2 : Int.tmod a b ≠ 0 := fun h => hd (Int.dvd_of_tmod_eq_zero h)
      simp only [ha, hd, false_or, if_false]
      have : Int.tmod a b < 0 := lt_of_le_of_ne h1 h2
      simp [this, hab]

namespace index_core

@[step]
theorem index.wrap_signed_spec (i : I64) (n : Usize) (hn : 0 < n.val)
    (hmax : (n.val : Int) ≤ I64.max) :
    index.wrap_signed i n ⦃ r => (r.val : Int) = i.val % n.val ∧ r.val < n.val ⦄ := by
  have hn1 := hcast_i64_val n hmax
  have hnpos : (0 : Int) < (UScalar.hcast .I64 n).val := by rw [hn1]; omega
  have hlo := Int.lt_tmod_of_pos i.val hnpos
  have hhi := Int.tmod_lt_of_pos i.val hnpos
  have hfix := tmod_fix i.val (UScalar.hcast .I64 n).val hnpos
  have hu : (n.val : Int) ≤ Usize.max := by scalar_tac
  unfold index.wrap_signed
  step*
  · subst n1_post
    rw [hn1] at hlo hhi hfix r_post i1_post
    have hr : (r.val : Int) < 0 := by scalar_tac
    rw [← r_post] at hfix
    simp only [hr, ↓reduceIte] at hfix
    have hc := hcast_usize_val i1 (by omega) (by omega)
    constructor <;> omega
  · subst n1_post
    rw [hn1] at hlo hhi hfix r_post
    have hr : ¬ (r.val : Int) < 0 := by scalar_tac
    rw [← r_post] at hfix
    simp only [hr, ↓reduceIte] at hfix
    have hc := hcast_usize_val r (by omega) (by omega)
    constructor <;> omega

/-- The row term of a flat index stays a full row below the grid's end. -/
theorem row_bound (a n m : Nat) (h : a < n) : a * m + m ≤ n * m := by
  have := Nat.mul_le_mul_right m (Nat.succ_le_of_lt h)
  rw [Nat.succ_mul] at this
  exact this

@[step]
theorem index.periodic_idx2_spec (i j nx ny : Usize) (hx : 0 < nx.val) (hy : 0 < ny.val)
    (hsz : nx.val * ny.val ≤ Usize.max) :
    index.periodic_idx2 i j nx ny ⦃ r =>
      r.val = i.val % nx.val * ny.val + j.val % ny.val ∧ r.val < nx.val * ny.val ⦄ := by
  have h1 := Nat.mod_lt i.val hx
  have h2 := Nat.mod_lt j.val hy
  have h3 := row_bound _ _ ny.val h1
  unfold index.periodic_idx2
  step*
  simp only [r_post, i2_post, i1_post, i3_post, true_and]
  generalize i.val % nx.val = a at *
  generalize j.val % ny.val = b at *
  omega

@[step]
theorem index.periodic_idx2_signed_spec (i j : I64) (nx ny : Usize) (hx : 0 < nx.val)
    (hy : 0 < ny.val) (hxm : (nx.val : Int) ≤ I64.max) (hym : (ny.val : Int) ≤ I64.max)
    (hsz : nx.val * ny.val ≤ Usize.max) :
    index.periodic_idx2_signed i j nx ny ⦃ r =>
      (r.val : Int) = i.val % nx.val * ny.val + j.val % ny.val ∧ r.val < nx.val * ny.val ⦄ := by
  unfold index.periodic_idx2_signed
  step*
  · have := row_bound _ _ ny.val i1_post1
    omega
  · have := row_bound _ _ ny.val i1_post1
    omega
  · have := row_bound _ _ ny.val i1_post1
    refine ⟨?_, by omega⟩
    rw [r_post, i2_post]
    push_cast
    rw [i1_post, i3_post]

@[step]
theorem index.periodic_idx3_spec (i j l nx ny nz : Usize) (hx : 0 < nx.val) (hy : 0 < ny.val)
    (hz : 0 < nz.val) (hsz : nx.val * ny.val * nz.val ≤ Usize.max) :
    index.periodic_idx3 i j l nx ny nz ⦃ r =>
      r.val = (i.val % nx.val * ny.val + j.val % ny.val) * nz.val + l.val % nz.val ∧
      r.val < nx.val * ny.val * nz.val ⦄ := by
  have h1 := Nat.mod_lt i.val hx
  have h2 := Nat.mod_lt j.val hy
  have h3 := Nat.mod_lt l.val hz
  have hxy := row_bound _ _ ny.val h1
  have hxy' : i.val % nx.val * ny.val + j.val % ny.val < nx.val * ny.val := by
    generalize i.val % nx.val = a at *
    generalize j.val % ny.val = b at *
    omega
  have hxyz := row_bound _ _ nz.val hxy'
  have hsz2 : nx.val * ny.val ≤ Usize.max := by
    have := Nat.le_mul_of_pos_right (nx.val * ny.val) hz
    omega
  unfold index.periodic_idx3
  step*
  · rw [i1_post]
    generalize i.val % nx.val = a at *
    omega
  · simp only [r_post, i5_post, i4_post, i2_post, i1_post, i3_post, i6_post, true_and]
    generalize i.val % nx.val * ny.val + j.val % ny.val = a at *
    generalize l.val % nz.val = c at *
    omega

@[step]
theorem index.coords3_spec (k ny nz : Usize) (hy : 0 < ny.val) (hz : 0 < nz.val) :
    index.coords3 k ny nz ⦃ c =>
      c.1.val = k.val / nz.val / ny.val ∧ c.2.1.val = k.val / nz.val % ny.val ∧
      c.2.2.val = k.val % nz.val ⦄ := by
  unfold index.coords3
  step*

/-- Dividing `x * q + r` by `q`, for `r < q`, recovers `x` and `r`. -/
theorem div_mod_of_lt (x q r : Nat) (hr : r < q) : (x * q + r) / q = x ∧ (x * q + r) % q = r := by
  have hq : 0 < q := by omega
  constructor
  · rw [Nat.add_comm, Nat.add_mul_div_right r x hq, Nat.div_eq_of_lt hr, Nat.zero_add]
  · rw [Nat.add_comm, Nat.add_mul_mod_self_right, Nat.mod_eq_of_lt hr]

/-- `periodic_idx3` inverts `coords3`: the coordinates of an index `k < nx * ny * nz`
map back to `k`. -/
theorem index.periodic_idx3_of_coords3 (k nx ny nz a b c : Usize) (hx : 0 < nx.val)
    (hy : 0 < ny.val) (hz : 0 < nz.val) (hsz : nx.val * ny.val * nz.val ≤ Usize.max)
    (hk : k.val < nx.val * ny.val * nz.val) (ha : a.val = k.val / nz.val / ny.val)
    (hb : b.val = k.val / nz.val % ny.val) (hc : c.val = k.val % nz.val) :
    index.periodic_idx3 a b c nx ny nz ⦃ r => r.val = k.val ⦄ := by
  have hk1 : k.val / nz.val < nx.val * ny.val := by
    rw [Nat.div_lt_iff_lt_mul hz]; exact hk
  have hk2 : k.val / nz.val / ny.val < nx.val := by
    rw [Nat.div_lt_iff_lt_mul hy]; exact hk1
  have hbn : k.val / nz.val % ny.val < ny.val := Nat.mod_lt _ hy
  have hcn : k.val % nz.val < nz.val := Nat.mod_lt _ hz
  apply WP.spec_mono (index.periodic_idx3_spec a b c nx ny nz hx hy hz hsz)
  rintro r ⟨hr, -⟩
  rw [hr, ha, hb, hc, Nat.mod_eq_of_lt hk2, Nat.mod_eq_of_lt hbn, Nat.mod_eq_of_lt hcn,
    Nat.div_add_mod' (k.val / nz.val) ny.val, Nat.div_add_mod' k.val nz.val]

/-- `coords3` inverts `periodic_idx3`: an index built from `(i, j, l)` has the
coordinates `(i % nx, j % ny, l % nz)`. -/
theorem index.coords3_of_periodic_idx3 (k i j l nx ny nz : Usize) (hy : 0 < ny.val)
    (hz : 0 < nz.val)
    (hk : k.val = (i.val % nx.val * ny.val + j.val % ny.val) * nz.val + l.val % nz.val) :
    index.coords3 k ny nz ⦃ c =>
      c.1.val = i.val % nx.val ∧ c.2.1.val = j.val % ny.val ∧ c.2.2.val = l.val % nz.val ⦄ := by
  have hl := div_mod_of_lt (i.val % nx.val * ny.val + j.val % ny.val) nz.val _ (Nat.mod_lt l.val hz)
  have hj := div_mod_of_lt (i.val % nx.val) ny.val _ (Nat.mod_lt j.val hy)
  apply WP.spec_mono (index.coords3_spec k ny nz hy hz)
  rintro c ⟨h1, h2, h3⟩
  rw [hk] at h1 h2 h3
  rw [hl.1, hj.1] at h1
  rw [hl.1, hj.2] at h2
  rw [hl.2] at h3
  exact ⟨h1, h2, h3⟩

end index_core
