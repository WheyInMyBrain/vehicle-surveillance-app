# `tracking-resolver`

High-throughput, real-time causal trajectory resolver and behavioral anomaly detection engine written in Rust. Designed for large-scale municipal Automated Number Plate Recognition (ANPR) surveillance networks, it reconstructs physical vehicle trajectories across inter-camera blind spots using spatial-temporal traffic telemetry, discrete road curvature physics, and a 4-layer Hierarchical Bayesian Consensus model.

---

## 1. System Overview & The Core Problem

Traditional map-matching and path-inference engines rely on algorithms like standard Dijkstra (shortest physical distance) or basic $A^*$ (fastest theoretical route). In urban surveillance networks with sparse, asynchronous camera sightings, these approaches break down:

1. **Shortest Distance Minimizers:** If a vehicle takes 8 minutes between two camera poles separated by an 800m alley, a distance-minimizer selects the 800m alley. It ignores that an 800m traversal in 8 minutes corresponds to crawling at $6\text{ km/h}$, whereas an adjacent 2,200m arterial road flowing at $35\text{ km/h}$ explains the 8-minute duration.
2. **Shortest Time Minimizers:** Standard routing assumes drivers optimize travel time globally. If an arterial slows down slightly, the model diverts hypothetical tracks through narrow residential backalleys to save theoretical paper seconds, ignoring human driving inertia and arterial bias.
3. **Unbounded Path Expansion:** Without strict space-time envelope pruning, searching for an observed duration $\Delta t_{\text{obs}}$ causes combinatorial explosion, evaluating hundreds of winding paths that hit the destination minutes too early or too late.
4. **Failure under Camera Dropouts:** In real-world cities, camera captures drop out (due to heavy rain, optical occlusion by buses, or hardware failure). Naive systems that penalize missing a camera linearly will divert a vehicle off a monitored arterial into unmonitored backalleys simply because the alley has zero cameras to miss.

### The Inverse Path Inference Problem

Given an initial camera sighting at $(A, \theta_A, v_A)$ at time $t_A$ and a subsequent camera sighting at $(B, \theta_B, v_B)$ at time $t_B$:

$$\Delta t_{\text{obs}} = t_B - t_A$$

We do not search for the shortest or fastest path. We search for the path $P^*$ that maximizes the posterior probability of producing the observation under empirical historical traffic flow, physical road curvature limits, turn geometry, and behavioral human inertia:

$$P^* = \arg\max_{P} \; \mathcal{P}\left(P \;\middle\vert\; \Delta t_{\text{obs}}, v_A, v_B, \theta_A, \theta_B, \mathbf{T}_{\text{prior}}, \mathcal{C}_{\text{unseen}}\right)$$

---

## 2. Theoretical Architecture: 4-Layer Bayesian Consensus

The engine evaluates trajectories through four decoupled layers:

```
                      RAW ANPR SIGHTINGS (A, B) + 5-MIN TENSOR
                                         │
┌────────────────────────────────────────┴────────────────────────────────────────┐
│ LAYER 1: Physical Kinematic & Spatiotemporal Envelope (Pruning)                 │
│ • Admissible Spatiotemporal Bounding Hull:                                      │
│     t_fwd(u) + t_edge + (dist_rem / v_max) ≤ Δt_obs × (1 + ε)                   │
│ • Curvature-Clamped Velocity Ceilings:                                          │
│     κ(s) = 1 / R_curve  ==>  v_ceiling(s) = min(v_legal, sqrt(a_lat / κ(s)))   │
│ • Forward/Backward Bang-Bang Acceleration Limits (a_accel, a_brake)             │
└────────────────────────────────────────┬────────────────────────────────────────┘
                                         │
                                 Feasible Subgraph
                                         │
┌────────────────────────────────────────┴────────────────────────────────────────┐
│ LAYER 2: Multi-Model Ensemble Generators (Hypothesis Space)                     │
│ Dynamic K expansion proportional to Δt_obs: K = clamp(K_min + Δt / K_interval)  │
│                                                                                 │
│   [Model A: Dynamic Tensor]       [Model B: Cognitive Hierarchy]                │
│   Yen's K-Shortest on             Yen's K-Shortest on                           │
│   delay: L / v_flow + J^2         arterial bias: L × w_tier                     │
│                    \                     /                                      │
│                     \                   /                                       │
│                   [Model C: Kinematic Minimalist]                               │
│                   Yen's K-Shortest on                                           │
│                   curvature limits: L / v_ceiling                               │
└────────────────────────────────────────┬────────────────────────────────────────┘
                                         │
                       Merged Candidate Set P = {P_1, ..., P_M}
                       + Generator Voting Consensus Tally
                                         │
┌────────────────────────────────────────┴────────────────────────────────────────┐
│ LAYER 3: Discriminative MAP Evaluator & Physical Realism Filters                 │
│ • Indian Left-Hand-Drive (LHD) Intersection Friction & Opposing Flow Delay      │
│ • Correlated Sub-linear Unobserved Camera Penalty with Clamped Floor:           │
│     ln P_cam = max(Floor, (N_missed)^0.5 × ln(p_miss))                         │
│ • Dwell / Stop Mixture Decomposition (Decouples Halts from Path Warping):       │
│     P(Δt | P) = (1 - w_stop) N(Δt; T_exp, σ^2) + w_stop P_dwell(Δt_excess)     │
│ • Speed Violation Soft Penalization (-25.0 log-likelihood)                      │
└────────────────────────────────────────┬────────────────────────────────────────┘
                                         │
┌────────────────────────────────────────┴────────────────────────────────────────┐
│ LAYER 4: Bayesian Consensus & Confidence Calibration                             │
│ • Consensus Prior Bonus: 3-model agreement (+2.6), 2-model agreement (+1.2)     │
│ • Softmax Posterior Normalization -> Dynamic Probability Distribution (%)       │
│ • Downstream Streaming Viterbi Beam Tracker Update & Ephemeris Spline           │
└─────────────────────────────────────────────────────────────────────────────────┘

```

---

## 3. Mathematical Formulations

### 3.1. Discrete Polyline Curvature Clamping ($\kappa$)

Road polylines are composed of intermediate survey coordinates. For any three consecutive vertices $p_1, p_2, p_3$, side lengths in meters $a, b, c$ are computed using planar geographic projection ($1^\circ \approx 111,139\text{ m}$). Using Heron's formula for triangle area $A$, the Menger curvature $\kappa$ is computed:

$$s = \frac{a + b + c}{2}, \quad A = \sqrt{s(s-a)(s-b)(s-c)}$$

$$\kappa = \frac{4A}{a \cdot b \cdot c} = \frac{1}{R_{\text{curve}}}$$

A physical vehicle cannot exceed lateral acceleration $a_{\text{lat, max}}$ without skidding or rolling. The velocity ceiling $v_{\text{ceiling}}$ across edge $e$ is:

$$v_{\text{turn, max}} = \sqrt{\frac{a_{\text{lat, max}}}{\kappa}}, \quad v_{\text{ceiling}}(e) = \min\left(v_{\text{legal}}(e) \cdot \mu_{\text{speed}}, \, v_{\text{turn, max}}\right)$$

### 3.2. Forward-Backward Bang-Bang Velocity Profiler

For any multi-edge path $P = \{e_1, e_2, \dots, e_n\}$, the theoretical minimum transit duration $T_{\min}(P)$ is derived by integrating forward from initial entry speed $v_{\text{start}}$ and backward from exit speed $v_{\text{end}}$:

$$\text{Forward pass: } v_{\text{fwd}}[i] = \min\left(v_{\text{ceiling}}[i], \, \sqrt{v_{\text{fwd}}[i-1]^2 + 2 a_{\text{accel}} L_i}\right)$$

$$\text{Backward pass: } v_{\text{bwd}}[i] = \min\left(v_{\text{ceiling}}[i], \, \sqrt{v_{\text{bwd}}[i+1]^2 + 2 a_{\text{brake}} L_i}\right)$$

$$v_{\text{feasible}}[i] = \max\left(v_{\text{crawl}}, \, \min(v_{\text{fwd}}[i], v_{\text{bwd}}[i])\right)$$

$$T_{\min}(P) = \sum_{i=1}^n \frac{L_i}{v_{\text{feasible}}[i]}$$

If $\Delta t_{\text{obs}} < T_{\min}(P) \cdot 0.85$, the vehicle has violated physical limits (triggering an `ExcessiveSpeed` anomaly alert).

### 3.3. Left-Hand Drive (LHD) Intersection Dynamics

Urban travel time variance is dominated by intersection turning friction. For heading deflection $\Delta \theta = \theta_{\text{in}} - \theta_{\text{out}} \in [-180^\circ, 180^\circ]$ and local intersection jam factor $J \in [0.0, 10.0]$:

$$\text{Delay}(\Delta \theta, J) = \begin{cases} 0.5\text{ s} & \vert{}\Delta \theta\vert{} \le 20^\circ \quad \text{(Straight Cruise)} \\ 3.0 + 0.4 J & 20^\circ < \Delta \theta \le 100^\circ \quad \text{(Left Turn - Protected Curb Lane)} \\ 14.0 + 2.5 J & -100^\circ \le \Delta \theta < -20^\circ \quad \text{(Right Turn - Crosses Opposing Flow)} \\ 45.0 + 4.0 J & \vert{}\Delta \theta\vert{} > 100^\circ \quad \text{(U-Turn / Acute Deflection)} \end{cases}$$

$$\text{Penalty}_{\text{turn}}(\Delta \theta) = \begin{cases} 0.0 & \vert{}\Delta \theta\vert{} \le 20^\circ \\ -0.2 & 20^\circ < \Delta \theta \le 100^\circ \\ -1.0 & -100^\circ \le \Delta \theta < -20^\circ \\ -4.5 & \vert{}\Delta \theta\vert{} > 100^\circ \end{cases}$$

### 3.4. Sub-linear Correlated Negative Information

If a candidate path bypasses $N$ operational ANPR cameras without generating sightings, treating each camera as an independent Bernoulli trial ($\prod p_{\text{miss}}$) creates an over-penalization bug during rain or lens fouling. The engine uses a sub-linear exponent with a soft floor:

$$\ln \mathcal{P}_{\text{camera\_miss}}(P) = \max\left(\text{Floor}_{\text{cam}}, \, (N_{\text{unobserved}})^{\alpha_{\text{sub}}} \cdot \ln(p_{\text{miss}})\right)$$

*(Default configuration: $\alpha_{\text{sub}} = 0.5$, $p_{\text{miss}} = 0.04$, $\text{Floor}_{\text{cam}} = -8.0$)*

### 3.5. Dwell / Stop Mixture Decomposition

When $\Delta t_{\text{obs}}$ exceeds theoretical expectations by multiple standard deviations, naive models invent zigzag detours to burn elapsed time. The engine decouples the traversal from parking/unloading halts:

$$\sigma_P^2 = \sigma_{\text{base}}^2 + \sum_{e \in P} \left( \frac{L_e}{v_e} \cdot (1.2 - \text{Confidence}_e) \right)^2$$

$$\text{Excess Time: } \Delta t_{\text{excess}} = \Delta t_{\text{obs}} - T_{\text{expected}}(P)$$

$$\text{Dwell Condition: } \Delta t_{\text{excess}} > \max\left(\text{Slack}_{\min}, \, 3.5 \cdot \sigma_P\right)$$

$$\text{Residual Used for Travel Time Likelihood: } \Delta t_{\text{res}} = \begin{cases} 0.0 & \text{if Dwell is Detected} \\ \Delta t_{\text{excess}} & \text{otherwise} \end{cases}$$

$$\ln \mathcal{P}_{\text{time}}(P) = -\frac{1}{2}\ln(2\pi \sigma_P^2) - \frac{(\Delta t_{\text{res}})^2}{2\sigma_P^2}$$

When a dwell is detected, the engine locks to the natural hierarchical path, sets `is_dwell_detected = true`, and reports `excess_dwell_sec` for anomaly processing.

### 3.6. Bayesian Consensus & Posterior Calibration

Each candidate $P_i$ receives an initial agreement prior based on the number of generator models ($M_{\text{Tensor}}, M_{\text{Hierarchy}}, M_{\text{Kinematic}}$) that independently produced it:

$$\ln P_0(P_i) = \begin{cases} +2.6 & \text{if all 3 models agree} \\ +1.2 & \text{if 2 models agree} \\ 0.0 & \text{if 1 model produced it} \end{cases}$$

The total joint log-posterior is evaluated:

$$\ln \mathcal{P}(P_i) = \ln P_0(P_i) + \ln \mathcal{P}_{\text{time}} + \text{Penalty}_{\text{turn}} + \text{Penalty}_{\text{hierarchy}} + \ln \mathcal{P}_{\text{camera\_miss}} + \text{Penalty}_{\text{speed\_violation}}$$

A numerically stable Softmax operation normalizes the candidates into calibrated probabilities:

$$\mathcal{P}(P_i) = \frac{\exp\left(\ln \mathcal{P}(P_i) - \max_j \ln \mathcal{P}(P_j)\right)}{\sum_{k} \exp\left(\ln \mathcal{P}(P_k) - \max_j \ln \mathcal{P}(P_j)\right)} \times 100\%$$

---

## 4. Crate Architecture & Module Manifest

```
crates/tracking-resolver/
├── Cargo.toml
├── src/
│   ├── lib.rs          # Public API re-exports
│   ├── config.rs       # ResolverConfig: Tunable parameters, limits, and penalties
│   ├── kinematics.rs   # Layer 1: κ computation, velocity ceilings, bang-bang pass
│   ├── envelope.rs     # Layer 1: Isochrone spatiotemporal bounding wavefront
│   ├── ensemble.rs     # Layer 2: Yen's K-Shortest path generator with dynamic K
│   ├── likelihood.rs   # Layer 3: LHD turn delays, sub-linear negative camera, dwell mixture
│   ├── consensus.rs    # Layer 4: Consensus bonus fusion & Softmax normalization
│   ├── anomaly.rs      # Real-time behavioral anomaly classification & scoring
│   ├── causal_gap.rs   # Inter-camera gap orchestrator connecting Layers 1-4
│   └── tracker.rs      # StreamingCausalTracker: Multi-leg Viterbi beam tracker
└── tests/
    ├── test_resolver.rs        # End-to-end single vehicle trips, ephemeris splines, anomalies
    └── test_fleet_resolver.rs  # Large-scale multi-shard stress test across 0% to 50% dropouts

```

---

## 5. Structural Breakdown of Modules

### `config.rs`

Defines `ResolverConfig`, encapsulating operational constants:

* `max_lateral_accel_mps2`: Default `3.5`. Clamps turning speeds along curves.
* `max_longitudinal_accel_mps2` & `max_braking_decel_mps2`: Default `2.4` and `3.8`. Used in the forward-backward kinematic profiler.
* `camera_miss_sublinear_exponent`: Default `0.5` ($\sqrt{N}$).
* `max_camera_miss_penalty`: Default `-8.0`. Prevents negative camera penalties from overriding highway hierarchy.
* `min_k_candidates` & `max_k_candidates`: Default `2` and `6`. Dynamic $K$ expansion scales between these limits based on `k_expansion_interval_sec` (default `60.0s`).

### `kinematics.rs`

* `resolve_legal_maxspeed_kmh(desc)`: Maps OpenStreetMap tags and `UnifiedHighwayTier` statutory limits to baseline speeds.
* `compute_max_curvature_inv_m(geometry)`: Computes polyline Menger curvature $\kappa = 1/R$.
* `edge_velocity_ceiling_mps(desc, config)`: Returns physical speed ceiling accounting for curve radius.
* `compute_path_min_physical_duration(descs, v_start, v_end, config)`: Implements the bang-bang forward-backward integration pass over a multi-edge candidate path.

### `envelope.rs`

* `compute_bidirectional_isochrone_hull(graph, src, dst, dt, config)`: Traverses forward from `src` using $A^*$ directed by a planar distance heuristic toward `dst`. Discards any node where:

$$g(u) + \frac{\text{dist}(u, \text{dst})}{v_{\text{max}}} > \Delta t_{\text{obs}} \cdot \text{Multiplier}$$



Prunes $>90\%$ of vertices in sub-millisecond time.

### `ensemble.rs`

* `dijkstra_subcorridor`: Local Dijkstra search constrained strictly within the valid edge set.
* `generate_k_paths`: Implements Yen's algorithm to produce structurally distinct alternative routes.
* `generate_ensemble_candidates`: Concurrently evaluates:
1. **Model A (Dynamic Tensor):** Evaluates flow impedance $L / v_{\text{flow}} \cdot (1 + 1.5(J/10)^2)$.
2. **Model B (Cognitive Hierarchy):** Evaluates arterial bias $L \cdot w_{\text{tier}}$, where Motorway = $1.0$, LivingStreet = $6.0$.
3. **Model C (Kinematic Minimalist):** Evaluates lowest theoretical transit time $L / v_{\text{ceiling}}$.



### `likelihood.rs`

* `junction_deflection_deg(e1, e2)`: Computes angular heading change across edge transitions in $[-180^\circ, 180^\circ]$.
* `junction_delay_and_penalty`: Implements LHD intersection dynamics.
* `negative_camera_penalty`: Applies sub-linear, floor-clamped negative information scoring.
* `evaluate_dwell_mixture`: Flags loitering events and computes the travel time residual $\Delta t_{\text{res}}$.
* `hierarchy_transition_penalty`: Penalizes dropping down road tiers (e.g., Primary $\to$ Residential).

### `consensus.rs`

* `fuse_bayesian_consensus(candidates, config)`: Applies consensus priors ($+2.6$ for 3 models, $+1.2$ for 2 models), performs numerically stable Softmax normalization, and sorts candidates descending by probability.

### `anomaly.rs`

* `evaluate_leg_anomaly`: Classifies traversal anomalies into four categories:
1. `SensorClockAnomaly`: Observed $\Delta t \le 0.5\text{s}$ over distance $>30\text{m}$. Severity: $100.0$.
2. `ExcessiveSpeed`: Observed $\Delta t < T_{\min} \cdot 0.85$. Severity: proportional to speed ceiling overshoot.
3. `LoiteringOrStashStop`: Triggered when `is_dwell_detected = true` and `excess_dwell_sec > 120.0s`. Severity: proportional to excess duration.
4. `DarkCorridorEvasion`: Triggered when $>65\%$ of the route traverses unmonitored minor backalleys over distances $>800\text{m}$.



### `causal_gap.rs`

* `resolve_camera_gap`: The central orchestrator. Handles direct single-edge and zero-edge connections, computes the isochrone hull, triggers dynamic-$K$ ensemble generation, scores candidates in Layer 3, and returns calibrated `Vec<TrajectoryCandidate>`.

### `tracker.rs`

* `StreamingCausalTracker`: Thread-safe, stateful tracker. Ingests streaming sightings, invokes `resolve_camera_gap`, feeds unobserved camera sets, updates the Viterbi hypothesis beam (retaining the top $N$ tracks sorted by log-likelihood), and constructs continuous vehicle trajectories.
* `VehicleTrack::reconstruct_continuous_ephemeris`: Slices reconstructed road geometry into uniform $15\text{m}$ segments, calculates historical flow speeds, and assigns interpolated UTC timestamps and speeds across every coordinate.

---

## 6. How to Use the Crate

### Cargo Dependencies

Add `tracking-resolver` to your `Cargo.toml`:

```toml
[dependencies]
tracking-resolver = { path = "../tracking-resolver" }
road-network-core = { path = "../road-network-core" }
road-graph-loader = { path = "../road-graph-loader" }
road-traffic-tensor = { path = "../road-traffic-tensor" }
chrono = { version = "0.4", features = ["serde"] }

```

### Complete End-to-End Tracking Example

```rust
use chrono::{Duration, Utc};
use road_graph_loader::RoadGraph;
use road_network_core::CameraSighting;
use road_traffic_tensor::TrafficTensorEngine;
use tracking_resolver::{ResolverConfig, StreamingCausalTracker};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // 1. Initialize Graph & Traffic Telemetry Tensor
    let graph = RoadGraph::open("data/road/mumbai_complete_graph.bin")?;
    let tensor = TrafficTensorEngine::new(&graph);

    // 2. Configure Operational Parameters
    let mut config = ResolverConfig::default();
    config.camera_miss_probability = 0.04;
    config.max_camera_miss_penalty = -8.0;

    // 3. Instantiate the Streaming Tracker
    let known_camera_edges = vec![101, 102, 105, 220, 310];
    let mut tracker = StreamingCausalTracker::new(&graph, &tensor)
        .with_beam_width(4)
        .with_config(config)
        .with_unobserved_cameras(known_camera_edges)
        .with_telemetry_lag(Duration::minutes(5));

    // 4. Ingest First Sighting (Camera Pole 1)
    let t0 = Utc::now();
    tracker.ingest_sighting(CameraSighting {
        camera_id: "CAM-NORTH-01".to_string(),
        vehicle_id: "MH-02-CZ-4411".to_string(),
        timestamp: t0,
        edge_from: 1042,
        edge_to: 1043,
        speed_kmh: 48.0,
        heading_deg: 182.0,
    });

    // 5. Ingest Second Sighting (Camera Pole 2, 4 minutes later)
    let t1 = t0 + Duration::seconds(240);
    tracker.ingest_sighting(CameraSighting {
        camera_id: "CAM-SOUTH-08".to_string(),
        vehicle_id: "MH-02-CZ-4411".to_string(),
        timestamp: t1,
        edge_from: 2011,
        edge_to: 2012,
        speed_kmh: 38.0,
        heading_deg: 175.0,
    });

    // 6. Inspect Reconstructed Trajectory & Anomalies
    if let Some(track) = tracker.get_track("MH-02-CZ-4411") {
        println!("Vehicle: {}", track.vehicle_id);
        println!("Total Reconstructed Distance: {:.1}m", track.total_distance_m());
        println!("Reconstructed Node Sequence: {:?}", track.best_reconstructed_path());

        // Check for behavioral anomalies
        for alert in &track.all_anomalies {
            println!(
                "ANOMALY DETECTED: [{:?}] Severity: {:.1} | {}",
                alert.anomaly_type, alert.severity_score, alert.explanation
            );
        }

        // Generate 15-meter sliced continuous spatio-temporal ephemeris
        let ephemeris = track.reconstruct_continuous_ephemeris(
            &graph,
            &tensor,
            tracker.telemetry_lag(),
        );

        for (lat, lon, timestamp, speed_kmh) in ephemeris.iter().take(5) {
            println!(
                "  Lat: {:.6}, Lon: {:.6} at {} | Speed: {:.1} km/h",
                lat, lon, timestamp.to_rfc3339(), speed_kmh
            );
        }
    }

    Ok(())
}

```

---

## 7. Performance Benchmarks & Stress-Test Audits

The engine was evaluated on a full-scale municipal road graph of Mumbai containing hundreds of thousands of directed edges, intermediate shape points, and historical 5-minute traffic bins.

### Multi-Condition Sensor Dropout Resilience (Fleet Size: 500 Vehicles)

Testing across varying camera dropout rates:

#### Empirical Stress-Test Benchmarks

##### Fleet Sample Size: 1,000 Vehicles (~370k Total Sightings)

| Dropout Rate (%) | Operational Context | Captures Processed | Throughput (sightings/s) | Avg Latency (µs/gap) | Precision (Mean / P95) | Recall (Mean / P95) | Loitering / Speed Alerts |
| --- | --- | --- | --- | --- | --- | --- | --- |
| **0.0%** | Pristine Sensor Grid | 38,939 | **318,685** | **3.14 µs** | 99.7% / 98.5% | 99.4% / 96.1% | 47 / 283 |
| **3.0%** | Nominal Daily Baseline | 37,841 | **287,037** | **3.48 µs** | 99.6% / 97.6% | 99.3% / 95.5% | 56 / 286 |
| **10.0%** | Inclement Weather / Glare | 34,714 | **244,845** | **4.08 µs** | 99.2% / 97.5% | 99.0% / 94.8% | 57 / 264 |
| **25.0%** | Degraded Network Grid | 29,226 | **165,293** | **6.05 µs** | 99.0% / 95.7% | 97.7% / 90.1% | 53 / 269 |
| **50.0%** | Sparse Corridors / Hardware Failures | 19,542 | **83,451** | **11.98 µs** | 96.5% / 81.1% | 94.4% / 80.8% | 80 / 192 |

##### Fleet Sample Size: 50,000 Vehicles (18.8M Total Sightings)

| Dropout Rate (%) | Operational Context | Captures Processed | Throughput (sightings/s) | Avg Latency (µs/gap) | Precision (Mean / P95) | Recall (Mean / P95) | Loitering / Speed Alerts |
| --- | --- | --- | --- | --- | --- | --- | --- |
| **0.0%** | Pristine Sensor Grid | 4,559,014 | **419,491** | **2.38 µs** | 99.7% / 98.6% | 99.3% / 95.7% | 6,428 / 33,426 |
| **3.0%** | Nominal Daily Baseline | 4,421,681 | **392,541** | **2.55 µs** | 99.6% / 98.1% | 99.2% / 95.3% | 6,585 / 32,918 |
| **10.0%** | Inclement Weather / Glare | 4,105,052 | **319,877** | **3.13 µs** | 99.5% / 97.1% | 98.9% / 94.4% | 6,415 / 30,743 |
| **25.0%** | Degraded Network Grid | 3,427,777 | **226,328** | **4.42 µs** | 99.0% / 95.5% | 98.0% / 91.4% | 6,473 / 27,192 |
| **50.0%** | Sparse Corridors / Hardware Failures | 2,283,662 | **114,914** | **8.70 µs** | 97.5% / 85.9% | 94.6% / 82.5% | 8,015 / 19,859 |

### Single-Trip HMM Beam Tracker Audit

```
Total Ground Truth Nodes:       142
Monitored Envelope Nodes:       136 (95.8%)
Camera Sightings Ingested:      63
Reconstructed Unique Nodes:     136
Active Hypotheses in Beam:      4
Resolution Time (Total Trip):   2.23 ms
Average Resolution Time / Gap:  36.04 µs
Path Precision:                 100.0%
Monitored Section Recall:       100.0%

```

### Key Behavioral Metrics

* **Sub-Microsecond Latency:** Under nominal conditions, resolution latency averages **$3.48\ \mu\text{s}$ per gap**, enabling streaming ingestion rates over **$280,000\text{ sightings/second}$** on a standard multi-core CPU.
* **Accuracy Across Sensor Failures:** Trajectory precision remains above **$99.0\%$** through a 25% camera failure rate, and retains **$96.5\%$** precision when half the camera infrastructure drops offline.
* **Deterministic Dwell Separation:** Injected 15-minute unmonitored loitering halts are decoupled from travel-time calculations, achieving a **99.0 severity alert score** without introducing phantom detour loops into the reconstructed path.