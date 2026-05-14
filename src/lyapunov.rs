//! Lyapunov chaos metric — measure brain sensitivity to perturbations.
//!
//! Math:
//!   λ = lim_{t→∞} (1/t) log |δ(t)/δ(0)|
//!
//! Interpretação:
//!   λ > 0: caótico (criativo, instável) — pequena mudança → output diverge
//!   λ < 0: estável (recall preciso) — mudança não afeta
//!   λ ≈ 0: edge of chaos (sweet spot) — equilíbrio
//!
//! Cálculo prático:
//!   Compare 2 outputs do brain p/ prompts levemente diferentes.
//!   δ(0) = string distance entre prompts (Hamming bits HV)
//!   δ(t) = string distance entre respostas
//!   λ = log(δ_resp / δ_prompt) / steps
//!
//! Aplicação:
//!   Ajustar temperature pra manter λ ≈ 0:
//!     λ > 0.5 → reduzir T (muito caótico)
//!     λ < -0.5 → aumentar T (muito determinístico)

#![allow(dead_code)]

/// Levenshtein distance between two byte strings (limited length).
pub fn lev_distance(a: &[u8], b: &[u8]) -> usize {
    let n = a.len().min(200);
    let m = b.len().min(200);
    if n == 0 { return m; }
    if m == 0 { return n; }
    let mut prev: Vec<usize> = (0..=m).collect();
    let mut curr = vec![0usize; m + 1];
    for i in 1..=n {
        curr[0] = i;
        for j in 1..=m {
            let cost = if a[i-1] == b[j-1] { 0 } else { 1 };
            curr[j] = (curr[j-1] + 1)
                .min(prev[j] + 1)
                .min(prev[j-1] + cost);
        }
        std::mem::swap(&mut prev, &mut curr);
    }
    prev[m]
}

/// Lyapunov exponent estimate from two prompt-response pairs.
/// `prompt_a`, `prompt_b`: perturbation pair (slightly different prompts)
/// `resp_a`, `resp_b`: corresponding brain responses
///
/// Returns λ ∈ ℝ. Stable: λ < 0. Chaotic: λ > 0.
pub fn lyapunov_estimate(prompt_a: &[u8], prompt_b: &[u8],
                          resp_a: &[u8], resp_b: &[u8]) -> f32 {
    let delta_in = lev_distance(prompt_a, prompt_b).max(1) as f32;
    let delta_out = lev_distance(resp_a, resp_b).max(1) as f32;
    // λ = ln(δ_out / δ_in) / "steps"
    // Steps = ratio of output to input length (rough)
    let n_steps = (resp_a.len().max(resp_b.len()).max(1) as f32) / 50.0;
    (delta_out / delta_in).ln() / n_steps.max(0.5)
}

/// Recommended temperature adjustment to push λ toward edge of chaos.
/// Returns multiplicative factor for current T:
///   λ > 0.5 (very chaotic): factor < 1 (reduce T)
///   λ < -0.5 (very stable): factor > 1 (increase T)
///   |λ| < 0.5: factor ≈ 1 (no adjust)
pub fn t_adjust_factor(lyapunov: f32) -> f32 {
    let lam = lyapunov.clamp(-3.0, 3.0);
    // Logistic curve centered at 0
    let target = 1.0 / (1.0 + (-lam * 0.5).exp());
    // Output: [0.5, 1.5] roughly
    1.5 - target
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lev_identical() {
        assert_eq!(lev_distance(b"abc", b"abc"), 0);
    }

    #[test]
    fn lev_one_char_diff() {
        assert_eq!(lev_distance(b"cat", b"bat"), 1);
    }

    #[test]
    fn lyapunov_stable() {
        // Same response → λ = 0 (or negative if δ_out tiny)
        let lam = lyapunov_estimate(b"hello world", b"hello worlx", b"resposta x", b"resposta x");
        assert!(lam < 1.0, "stable system should have low λ: {}", lam);
    }

    #[test]
    fn lyapunov_chaotic() {
        // Tiny prompt change → big response change → λ > 0
        let lam = lyapunov_estimate(
            b"oi", b"ai",
            b"resposta detalhada aqui sobre o problema X que estende muito",
            b"completamente outra resposta totalmente diferente sobre Y zwxq",
        );
        assert!(lam > 0.0, "chaotic system should have λ > 0: {}", lam);
    }

    #[test]
    fn t_adjust_monotonic() {
        let f_low = t_adjust_factor(-2.0);   // stable → increase T
        let f_mid = t_adjust_factor(0.0);    // balanced → ~1.0
        let f_high = t_adjust_factor(2.0);   // chaotic → reduce T
        assert!(f_low > f_mid && f_mid > f_high,
            "factor should be monotonic: {} > {} > {}", f_low, f_mid, f_high);
    }
}
