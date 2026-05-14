//! Thermodynamic helpers — adaptive temperature, free energy, relaxation time.
//!
//! Math fundamento:
//!
//! 1. Boltzmann distribution:
//!    P(s) = exp(-E(s)/kT) / Z
//!    Sampling com T: alto T = uniforme (criativo), baixo T = greedy (preciso).
//!
//! 2. Free Energy (Helmholtz):
//!    F = -kT log Z
//!    Predictive coding: brain minimiza F implicitly via WAL ingest.
//!
//! 3. Relaxation time (Langevin):
//!    τ = 1/γ onde γ é damping. Easy query alta γ (cache hit), hard query baixa γ.
//!
//! 4. Entropy:
//!    H(p) = -Σ p_i log p_i
//!    Mede uncertainty da resposta. Alta entropia → precisa Qwen.

#![allow(dead_code)]

/// Compute Shannon entropy of probability distribution (in nats).
/// p must sum to ~1.0.
pub fn entropy(p: &[f32]) -> f32 {
    p.iter()
        .filter(|&&x| x > 1e-9)
        .map(|&x| -x * x.ln())
        .sum()
}

/// Adaptive temperature based on query confidence.
/// confidence ∈ [0, 1]
///   - 1.0 (very confident) → T ≈ 0.1 (greedy)
///   - 0.5 (uncertain) → T ≈ 0.7 (default)
///   - 0.0 (no idea) → T ≈ 1.5 (exploratory)
pub fn adaptive_temperature(confidence: f32) -> f32 {
    let c = confidence.clamp(0.0, 1.0);
    // Smooth interp: high conf → low T
    0.1 + (1.0 - c).powf(2.0) * 1.4
}

/// Relaxation time τ (ms) for a query given its difficulty.
/// difficulty ∈ [0, 1]: 0 = easy (cache hit), 1 = hard (full Qwen).
/// Maps to budgeted compute time.
pub fn relaxation_tau_ms(difficulty: f32) -> f32 {
    let d = difficulty.clamp(0.0, 1.0);
    // Exponential growth: easy 5ms, hard 1000ms
    5.0 * (200.0_f32).powf(d)
}

/// Free energy delta after observing outcome (predictive coding).
/// surprise = -log P(observation | prior)
/// Higher = more learning signal.
pub fn surprise(observed_prob: f32) -> f32 {
    -((observed_prob.max(1e-9)).ln())
}

/// Softmax with temperature T.
pub fn softmax_t(logits: &[f32], t: f32) -> Vec<f32> {
    let t = t.max(0.01);
    let scaled: Vec<f32> = logits.iter().map(|&x| x / t).collect();
    let max = scaled.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let exps: Vec<f32> = scaled.iter().map(|&x| (x - max).exp()).collect();
    let sum: f32 = exps.iter().sum();
    if sum > 0.0 {
        exps.iter().map(|&v| v / sum).collect()
    } else {
        vec![1.0 / logits.len() as f32; logits.len()]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn entropy_uniform_maximal() {
        let p = vec![0.25, 0.25, 0.25, 0.25];
        let h = entropy(&p);
        let max_h = (4.0_f32).ln();  // log(N) for uniform N=4
        assert!((h - max_h).abs() < 0.01, "uniform entropy should be ln(4): {} vs {}", h, max_h);
    }

    #[test]
    fn entropy_delta_zero() {
        let p = vec![1.0, 0.0, 0.0, 0.0];
        let h = entropy(&p);
        assert!(h < 0.01, "delta dist entropy should be 0: {}", h);
    }

    #[test]
    fn adaptive_t_monotonic() {
        let t_high = adaptive_temperature(0.95);
        let t_mid = adaptive_temperature(0.5);
        let t_low = adaptive_temperature(0.1);
        assert!(t_high < t_mid, "high conf should yield lower T: {} vs {}", t_high, t_mid);
        assert!(t_mid < t_low, "low conf should yield higher T: {} vs {}", t_mid, t_low);
    }

    #[test]
    fn relaxation_tau_grows() {
        let easy = relaxation_tau_ms(0.0);
        let mid = relaxation_tau_ms(0.5);
        let hard = relaxation_tau_ms(1.0);
        assert!(easy < mid && mid < hard);
        assert!(easy < 10.0, "easy should be ~5ms: {}", easy);
        assert!(hard > 500.0, "hard should be ~1000ms: {}", hard);
    }

    #[test]
    fn softmax_t_temperature_effect() {
        let logits = vec![1.0, 2.0, 3.0];
        let p_low = softmax_t(&logits, 0.1);  // peaked
        let p_high = softmax_t(&logits, 10.0); // flat
        // Low T → top dominates more than high T
        assert!(p_low[2] > p_high[2]);
        // High T → more uniform
        let h_low = entropy(&p_low);
        let h_high = entropy(&p_high);
        assert!(h_high > h_low);
    }
}
