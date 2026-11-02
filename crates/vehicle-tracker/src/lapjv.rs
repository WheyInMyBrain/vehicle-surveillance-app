/// Solves the Linear Assignment Problem (min-cost bipartite matching).
///
/// `cost_matrix`: row-major slice of size `num_rows * num_cols`.
/// `max_cost`: assignments with cost exceeding this threshold are rejected.
///
/// Returns:
/// `(matches, unmatched_rows, unmatched_cols)`
pub fn lapjv(
    cost_matrix: &[f32],
    num_rows: usize,
    num_cols: usize,
    max_cost: f32,
) -> (Vec<(usize, usize)>, Vec<usize>, Vec<usize>) {
    if num_rows == 0 || num_cols == 0 {
        return (
            Vec::new(),
            (0..num_rows).collect(),
            (0..num_cols).collect(),
        );
    }

    let mut row_sol = vec![None; num_rows];
    let mut col_sol = vec![None; num_cols];
    let mut u = vec![0.0f32; num_rows];
    let mut v = vec![0.0f32; num_cols];

    // Dual updates & augmenting paths
    for i in 0..num_rows {
        let mut min_val = f32::INFINITY;
        let mut min_c = 0;
        let row_offset = i * num_cols;

        for j in 0..num_cols {
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

    // Augmenting path search for unassigned rows
    let mut dist = vec![0.0f32; num_cols];
    let mut pred = vec![0usize; num_cols];
    let mut visited = vec![false; num_cols];

    for cur_row in 0..num_rows {
        if row_sol[cur_row].is_some() {
            continue;
        }

        dist.fill(f32::INFINITY);
        visited.fill(false);

        let mut active_cols: Vec<usize> = (0..num_cols).collect();
        let mut last_col = 0;
        let mut i = cur_row;

        loop {
            let row_offset = i * num_cols;
            let mut min_d = f32::INFINITY;
            let mut min_idx = 0;

            for (idx, &j) in active_cols.iter().enumerate() {
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

            let c = active_cols.remove(min_idx);
            visited[c] = true;

            for j in 0..num_cols {
                if visited[j] {
                    v[j] += dist[j] - min_d;
                }
            }
            u[cur_row] += min_d;
            for r in 0..num_rows {
                if let Some(assigned_c) = row_sol[r] {
                    if visited[assigned_c] {
                        u[r] -= dist[assigned_c] - min_d;
                    }
                }
            }

            if col_sol[last_col].is_none() {
                break;
            }
            i = col_sol[last_col].unwrap();
        }

        // Augment along the shortest path
        let mut c = last_col;
        while visited[c] {
            let r = pred[c];
            col_sol[c] = Some(r);
            let prev_c = row_sol[r];
            row_sol[r] = Some(c);
            if r == cur_row {
                break;
            }
            c = prev_c.unwrap();
        }
    }

    let mut matches = Vec::with_capacity(num_rows.min(num_cols));
    let mut unmatched_rows = Vec::new();
    let mut unmatched_cols = Vec::new();

    for r in 0..num_rows {
        if let Some(c) = row_sol[r] {
            let cost = cost_matrix[r * num_cols + c];
            if cost <= max_cost {
                matches.push((r, c));
            } else {
                unmatched_rows.push(r);
            }
        } else {
            unmatched_rows.push(r);
        }
    }

    let mut matched_cols = vec![false; num_cols];
    for &(_, c) in &matches {
        matched_cols[c] = true;
    }
    for c in 0..num_cols {
        if !matched_cols[c] {
            unmatched_cols.push(c);
        }
    }

    (matches, unmatched_rows, unmatched_cols)
}