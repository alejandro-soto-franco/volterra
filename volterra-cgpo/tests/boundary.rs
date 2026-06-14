use volterra_cgpo::nephroid_boundary;

/// Interior-cell count for Lx=Ly=100 must be within ±5 of the Python reference (5621).
#[test]
fn interior_count_100() {
    let b = nephroid_boundary(100, 100);
    let count = b.interior_count();
    println!("interior_count(100,100) = {count}");
    assert!(
        (count as i64 - 5621).abs() <= 5,
        "expected ~5621, got {count}"
    );
}

/// Interior-cell count for Lx=Ly=60 must be within ±5 of the Python reference (1965).
#[test]
fn interior_count_60() {
    let b = nephroid_boundary(60, 60);
    let count = b.interior_count();
    println!("interior_count(60,60) = {count}");
    assert!(
        (count as i64 - 1965).abs() <= 5,
        "expected ~1965, got {count}"
    );
}

/// Every outer-boundary cell must have a unit normal (|n| ≈ 1, tol 1e-2).
/// Every inner-boundary cell must have a unit normal.
/// Off-boundary cells must have [0, 0] normals.
#[test]
fn normals_unit_length_and_zero_off_boundary() {
    let b = nephroid_boundary(60, 60);
    let tol = 1e-2_f64;

    for x in 0..b.lx {
        for y in 0..b.ly {
            let idx = x * b.ly + y;

            let on = b.outer_normals[idx];
            let inn = b.inner_normals[idx];

            if b.is_outer[idx] {
                let mag = (on[0] * on[0] + on[1] * on[1]).sqrt();
                assert!(
                    (mag - 1.0).abs() < tol,
                    "outer normal at ({x},{y}) has |n|={mag}, expected ~1"
                );
            } else {
                assert_eq!(
                    on,
                    [0.0, 0.0],
                    "outer_normals[{x},{y}] should be [0,0] for non-outer cell"
                );
            }

            if b.is_inner[idx] {
                let mag = (inn[0] * inn[0] + inn[1] * inn[1]).sqrt();
                assert!(
                    (mag - 1.0).abs() < tol,
                    "inner normal at ({x},{y}) has |n|={mag}, expected ~1"
                );
            } else {
                assert_eq!(
                    inn,
                    [0.0, 0.0],
                    "inner_normals[{x},{y}] should be [0,0] for non-inner cell"
                );
            }
        }
    }
}

/// Every inner cell must have at least one outer 4-neighbour (consistency check).
#[test]
fn inner_cells_have_outer_neighbour() {
    let b = nephroid_boundary(60, 60);
    for x in 0..b.lx {
        for y in 0..b.ly {
            if !b.is_inner[x * b.ly + y] {
                continue;
            }
            let xi = x as i64;
            let yi = y as i64;
            let has_outer = [(xi + 1, yi), (xi - 1, yi), (xi, yi + 1), (xi, yi - 1)]
                .iter()
                .any(|&(nx, ny)| {
                    if nx < 0 || ny < 0 || nx >= b.lx as i64 || ny >= b.ly as i64 {
                        return false;
                    }
                    b.is_outer[nx as usize * b.ly + ny as usize]
                });
            assert!(
                has_outer,
                "inner cell ({x},{y}) has no outer 4-neighbour"
            );
        }
    }
}

/// Every outer cell must have at least one non-inside 4-neighbour.
#[test]
fn outer_cells_have_non_inside_neighbour() {
    let b = nephroid_boundary(60, 60);
    for x in 0..b.lx {
        for y in 0..b.ly {
            if !b.is_outer[x * b.ly + y] {
                continue;
            }
            let xi = x as i64;
            let yi = y as i64;
            let has_non_inside = [(xi + 1, yi), (xi - 1, yi), (xi, yi + 1), (xi, yi - 1)]
                .iter()
                .any(|&(nx, ny)| {
                    if nx < 0 || ny < 0 || nx >= b.lx as i64 || ny >= b.ly as i64 {
                        return true; // out of grid = non-inside
                    }
                    !b.inside[nx as usize * b.ly + ny as usize]
                });
            assert!(
                has_non_inside,
                "outer cell ({x},{y}) has no non-inside 4-neighbour"
            );
        }
    }
}
