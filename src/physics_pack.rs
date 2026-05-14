//! Physics pack — FEP + Diffusion + Hamiltonian (MVP infrastructure).
//!
//! Implementa #5, #6, #8 do roadmap em forma minimal usável:
//! - FEP loss (Friston Free Energy Principle)
//! - Diffusion denoising score-matching
//! - Hamiltonian flow (energy-conserving integration)
//!
//! Math:
//!
//! FEP:
//!   F = D_KL(q(s|o) || p(s)) - E_q[log p(o|s)]
//!   Predictive coding: minimize prediction error + complexity
//!   Simplification: F ≈ -log p(o|context) = surprise
//!
//! Diffusion:
//!   Forward: x_t = x_0 + σ_t · ε  where ε ~ N(0, I)
//!   Reverse: ∂x = -∇log p(x) · dt + noise
//!   Score matching loss: ||score_pred - true_score||²
//!
//! Hamiltonian:
//!   H = T(p) + V(q)
//!   dq/dt = ∂H/∂p, dp/dt = -∂H/∂q
//!   Leapfrog integrator: 2nd order accurate, symplectic.

#![allow(dead_code)]

// ─────────────────────────────────────────────────────────────
// #5 FEP — Free Energy Principle loss
// ─────────────────────────────────────────────────────────────

/// FEP loss for predictive coding.
/// observation: actual byte/token sample
/// prior_probs: model's prediction distribution
/// returns variational free energy F ≈ surprise + KL
pub fn fep_loss(observation_idx: usize, prior_probs: &[f32], posterior_probs: &[f32]) -> f32 {
    if observation_idx >= prior_probs.len() { return f32::INFINITY; }
    let obs_likelihood = prior_probs[observation_idx].max(1e-9);
    let surprise = -obs_likelihood.ln();
    // KL divergence q (posterior) || p (prior)
    let mut kl = 0.0f32;
    for i in 0..prior_probs.len().min(posterior_probs.len()) {
        let q = posterior_probs[i].max(1e-9);
        let p = prior_probs[i].max(1e-9);
        kl += q * (q / p).ln();
    }
    surprise + kl
}

/// FEP-driven weight update: gradient descent on F.
/// Returns updated weight vector.
pub fn fep_update(weights: &[f32], gradient: &[f32], lr: f32) -> Vec<f32> {
    weights.iter().zip(gradient.iter())
        .map(|(w, g)| w - lr * g)
        .collect()
}

// ─────────────────────────────────────────────────────────────
// #8 Diffusion / Denoising
// ─────────────────────────────────────────────────────────────

/// Add Gaussian noise to byte signal (forward diffusion).
/// σ: noise std. t: timestep [0, 1]. Larger t → more noise.
pub fn add_noise(signal: &[f32], t: f32, seed: u64) -> Vec<f32> {
    let sigma = t.clamp(0.0, 1.0);
    let mut state = seed;
    signal.iter().map(|&x| {
        state ^= state << 13; state ^= state >> 7; state ^= state << 17;
        let u1 = ((state as u32) as f32 / u32::MAX as f32).max(1e-9);
        state ^= state << 13; state ^= state >> 7; state ^= state << 17;
        let u2 = (state as u32) as f32 / u32::MAX as f32;
        let z = (-2.0 * u1.ln()).sqrt() * (2.0 * std::f32::consts::PI * u2).cos();
        x + sigma * z
    }).collect()
}

/// Score matching loss: ||predicted_score - true_score||²
/// For diffusion training (offline)
pub fn score_matching_loss(predicted_score: &[f32], true_score: &[f32]) -> f32 {
    let mut sum = 0.0f32;
    let n = predicted_score.len().min(true_score.len());
    for i in 0..n {
        let d = predicted_score[i] - true_score[i];
        sum += d * d;
    }
    sum / n.max(1) as f32
}

// ─────────────────────────────────────────────────────────────
// #6 Hamiltonian flow
// ─────────────────────────────────────────────────────────────

/// Hamiltonian state: (position q, momentum p).
#[derive(Clone, Debug)]
pub struct PhaseSpace {
    pub q: Vec<f32>,
    pub p: Vec<f32>,
}

/// Leapfrog integrator step: energy-conserving 2nd order.
/// H = T(p) + V(q) where:
///   T(p) = ½ Σ p_i² / m_i (kinetic)
///   V(q) given by potential function
/// dq/dt = ∂H/∂p = p/m
/// dp/dt = -∂H/∂q = -∇V(q)
pub fn leapfrog_step<F>(state: &mut PhaseSpace, mass: &[f32], grad_v: F, dt: f32)
where F: Fn(&[f32]) -> Vec<f32>
{
    let n = state.q.len();
    // Half-step momentum
    let g = grad_v(&state.q);
    for i in 0..n {
        state.p[i] -= 0.5 * dt * g[i];
    }
    // Full-step position
    for i in 0..n {
        state.q[i] += dt * state.p[i] / mass[i].max(1e-9);
    }
    // Half-step momentum
    let g2 = grad_v(&state.q);
    for i in 0..n {
        state.p[i] -= 0.5 * dt * g2[i];
    }
}

/// Total Hamiltonian (energy).
pub fn hamiltonian(state: &PhaseSpace, mass: &[f32], potential: f32) -> f32 {
    let kinetic: f32 = state.p.iter().zip(mass.iter())
        .map(|(p, m)| p * p / (2.0 * m.max(1e-9)))
        .sum();
    kinetic + potential
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fep_low_surprise_correct_pred() {
        let prior = vec![0.9, 0.05, 0.05];
        let posterior = vec![0.95, 0.025, 0.025];
        let f = fep_loss(0, &prior, &posterior);
        // Observation = most-likely → low surprise
        assert!(f < 1.0, "low surprise expected: {}", f);
    }

    #[test]
    fn fep_high_surprise_wrong_pred() {
        let prior = vec![0.9, 0.05, 0.05];
        let posterior = vec![0.1, 0.45, 0.45];
        let f = fep_loss(2, &prior, &posterior);
        assert!(f > 1.0, "high surprise expected: {}", f);
    }

    #[test]
    fn diffusion_noise_increases_distance() {
        let signal = vec![1.0, 2.0, 3.0, 4.0];
        let noised_low = add_noise(&signal, 0.1, 42);
        let noised_high = add_noise(&signal, 1.0, 42);
        let d_low: f32 = signal.iter().zip(noised_low.iter())
            .map(|(a, b)| (a - b).abs()).sum();
        let d_high: f32 = signal.iter().zip(noised_high.iter())
            .map(|(a, b)| (a - b).abs()).sum();
        assert!(d_high > d_low, "higher t → more noise: {} vs {}", d_low, d_high);
    }

    #[test]
    fn hamiltonian_energy_conservation() {
        // Harmonic oscillator: V = ½ q², ∇V = q
        let mut state = PhaseSpace { q: vec![1.0], p: vec![0.0] };
        let mass = vec![1.0];
        let v_initial = 0.5 * state.q[0] * state.q[0];
        let h_initial = hamiltonian(&state, &mass, v_initial);
        // Integrate 100 steps
        for _ in 0..100 {
            leapfrog_step(&mut state, &mass, |q| vec![q[0]], 0.01);
        }
        let v_final = 0.5 * state.q[0] * state.q[0];
        let h_final = hamiltonian(&state, &mass, v_final);
        let drift = ((h_final - h_initial) / h_initial).abs();
        // Energy drift should be small (<5% for 100 steps)
        assert!(drift < 0.05, "energy should conserve: drift = {}", drift);
    }
}
