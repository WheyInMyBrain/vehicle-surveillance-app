use crate::config::ResolverConfig;
use crate::ensemble::RouteCandidate;

#[derive(Debug, Clone)]
pub struct EvaluatedTrajectory {
    pub candidate: RouteCandidate,
    pub probability_percent: f32,
    pub min_physical_sec: f32,
    pub expected_tensor_sec: f32,
    pub distance_m: f32,
    pub dark_alley_ratio: f32,
    pub is_dwell_detected: bool,
    pub excess_dwell_sec: f32,
    pub log_likelihood: f32,
}

pub fn fuse_bayesian_consensus(
    mut candidates: Vec<(RouteCandidate, f32, f32, f32, f32, f32, bool, f32)>,
    config: &ResolverConfig,
) -> Vec<EvaluatedTrajectory> {
    if candidates.is_empty() {
        return Vec::new();
    }

    // Step 1: Consensus Multiplier
    for (cand, log_l, _, _, _, _, _, _) in &mut candidates {
        let agreement_bonus = match cand.models_agreed.len() {
            3 => config.consensus_bonus_triple,
            2 => config.consensus_bonus_double,
            _ => 0.0,
        };
        *log_l += agreement_bonus;
    }

    // Step 2: Softmax Normalization
    let max_log = candidates
        .iter()
        .map(|(_, l, ..)| *l)
        .fold(f32::NEG_INFINITY, f32::max);
    let sum_exp: f32 = candidates
        .iter()
        .map(|(_, l, ..)| (l - max_log).exp())
        .sum();

    let mut evaluated = Vec::with_capacity(candidates.len());
    for (cand, log_l, min_t, exp_t, dist, dark, dwell, excess) in candidates {
        let prob = if sum_exp > 0.0 {
            (((log_l - max_log).exp() / sum_exp) * 100.0).round()
        } else {
            0.0
        };

        evaluated.push(EvaluatedTrajectory {
            candidate: cand,
            probability_percent: prob,
            min_physical_sec: min_t,
            expected_tensor_sec: exp_t,
            distance_m: dist,
            dark_alley_ratio: dark,
            is_dwell_detected: dwell,
            excess_dwell_sec: excess,
            log_likelihood: log_l,
        });
    }

    evaluated.sort_by(|a, b| b.probability_percent.partial_cmp(&a.probability_percent).unwrap());
    evaluated
}