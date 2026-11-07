//! Jonker-Volgenant (LAPJV) Linear Assignment Problem Solver.
//!
//! Re-architected for safety-critical execution:
//! - Pre-allocated fixed stack memory (zero dynamic heap allocation).
//! - Deterministic execution time bounds.
//! - Guarded array indices to prevent panics.

use heapless::Vec as StackVec;

pub const MAX_MATCH_DIM: usize = 64;

/// Result tuple: (matches, unmatched_rows, unmatched_cols)
pub struct LapMatchResult {
    pub matches: StackVec<(usize, usize), MAX_MATCH_DIM>,
    pub unmatched_rows: StackVec<usize, MAX_MATCH_DIM>,
    pub unmatched_cols: StackVec<usize, MAX_MATCH_DIM>,
}

/// Solves the Linear Assignment Problem with zero heap allocations.
///
/// `cost_matrix`: row-major slice of dimensions `num_rows * num_cols`.
pub fn lapjv_bounded(
    cost_matrix: &[f32],
    num_rows: usize,
    num_cols: usize,
    max_cost: f32,
) -> LapMatchResult {
    let mut result = LapMatchResult {
        matches: StackVec::new(),
        unmatched_rows: StackVec::new(),
        unmatched_cols: StackVec::new(),
    };

    if num_rows == 0 || num_cols == 0 {
        for r in 0..num_rows.min(MAX_MATCH_DIM) {
            let _ = result.unmatched_rows.push(r);
        }
        for c in 0..num_cols.min(MAX_MATCH_DIM) {
            let _ = result.unmatched_cols.push(c);
        }
        return result;
    }

    let nrows = num_rows.min(MAX_MATCH_DIM);
    let ncols = num_cols.min(MAX_MATCH_DIM);

    let mut row_sol = [None::<usize>; MAX_MATCH_DIM];
    let mut col_sol = [None::<usize>; MAX_MATCH_DIM];
    let mut u = [0.0f32; MAX_MATCH_DIM];
    let mut v = [0.0f32; MAX_MATCH_DIM];

    // Initial dual variable updates
    for i in 0..nrows {
        let mut min_val = f32::INFINITY;
        let mut min_c = 0;
        let row_offset = i * num_cols;

        for j in 0..ncols {
            let val = cost_matrix[row_offset + j];
            if val < min_val {
                min_val = val;
                min_c = j;
            }
        }
        u[i] = min_val;
        if col_sol[min_c].is_none() {
            col_sol[min_c] = Some(i);
            row_sol[i] = Some(min_c);
        }
    }

    let mut dist = [0.0f32; MAX_MATCH_DIM];
    let mut pred = [0usize; MAX_MATCH_DIM];
    let mut visited = [false; MAX_MATCH_DIM];

    // Augmenting path search with bounded iterations (JPL Rule 1 & 2)
    for cur_row in 0..nrows {
        if row_sol[cur_row].is_some() {
            continue;
        }

        dist.fill(f32::INFINITY);
        visited.fill(false);

        let mut active_cols = [0usize; MAX_MATCH_DIM];
        let mut active_len = ncols;
        for c in 0..ncols {
            active_cols[c] = c;
        }

        let mut last_col = 0;
        let mut i = cur_row;

        // Bounded loop invariant: maximum path length bounded by ncols
        for _path_step in 0..ncols {
            let row_offset = i * num_cols;
            let mut min_d = f32::INFINITY;
            let mut min_idx = 0;

            for idx in 0..active_len {
                let j = active_cols[idx];
                if !visited[j] {
                    let reduced = cost_matrix[row_offset + j] - u[i] - v[j];
                    if reduced < dist[j] {
                        dist[j] = reduced;
                        pred[j] = i;
                    }
                    if dist[j] < min_d {
                        min_d = dist[j];
                        min_idx = idx;
                        last_col = j;
                    }
                }
            }

            if min_d == f32::INFINITY {
                break;
            }

            // O(1) swap_remove instead of O(N) heap vector remove
            let c = active_cols[min_idx];
            active_cols[min_idx] = active_cols[active_len - 1];
            active_len -= 1;
            visited[c] = true;

            for j in 0..ncols {
                if visited[j] {
                    v[j] += dist[j] - min_d;
                }
            }
            u[cur_row] += min_d;
            for r in 0..nrows {
                if let Some(assigned_c) = row_sol[r] {
                    if assigned_c < MAX_MATCH_DIM && visited[assigned_c] {
                        u[r] -= dist[assigned_c] - min_d;
                    }
                }
            }

            if col_sol[last_col].is_none() {
                break;
            }
            if let Some(next_i) = col_sol[last_col] {
                i = next_i;
            } else {
                break;
            }
        }

        // Augment along the shortest path
        let mut c = last_col;
        for _step in 0..MAX_MATCH_DIM {
            if !visited[c] {
                break;
            }
            let r = pred[c];
            col_sol[c] = Some(r);
            let prev_c = row_sol[r];
            row_sol[r] = Some(c);
            if r == cur_row {
                break;
            }
            if let Some(next_c) = prev_c {
                c = next_c;
            } else {
                break;
            }
        }
    }

    let mut matched_cols = [false; MAX_MATCH_DIM];

    for r in 0..nrows {
        if let Some(c) = row_sol[r] {
            let cost = cost_matrix[r * num_cols + c];
            if cost <= max_cost {
                let _ = result.matches.push((r, c));
                matched_cols[c] = true;
            } else {
                let _ = result.unmatched_rows.push(r);
            }
        } else {
            let _ = result.unmatched_rows.push(r);
        }
    }

    for c in 0..ncols {
        if !matched_cols[c] {
            let _ = result.unmatched_cols.push(c);
        }
    }

    result
}