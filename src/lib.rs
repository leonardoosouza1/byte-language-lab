//! # byte-language-lab
//!
//! Lab empírico para a hipótese: **inteligência fala bytes estruturados**.
//!
//! Este crate implementa três variantes ("paths") de formato de byte estruturado
//! e mede empiricamente seu comportamento em representação numérica, aritmética,
//! aprendizado e composição multi-domínio.
//!
//! Ver `DESIGN.md` para o formalismo completo.

#![warn(missing_docs)]
#![warn(clippy::pedantic)]
#![warn(clippy::nursery)]
#![allow(clippy::module_name_repetitions)]
#![allow(clippy::cast_precision_loss)]
#![allow(clippy::cast_possible_truncation)]
#![allow(clippy::cast_sign_loss)]
#![allow(clippy::cast_possible_wrap)]

pub mod format;
pub mod arithmetic;
pub mod learn;
pub mod compose;
pub mod fsm_byte;
pub mod gpu;
pub mod wal;
pub mod gravity_forest;
pub mod curator;
pub mod wave_encoding;
pub mod thermo;
pub mod lyapunov;

#[cfg(feature = "gpu-aql")]
pub mod gpu_aql;

/// RNG determinístico minimalista (xorshift64) para reprodutibilidade total.
///
/// Sem dependência de `rand`: garante que cada experimento é byte-idêntico
/// dado o mesmo seed, em qualquer máquina.
pub struct Rng(u64);

impl Rng {
    /// Cria RNG com seed explícito. Seed 0 é proibido (xorshift colapsa em 0).
    #[must_use]
    pub const fn new(seed: u64) -> Self {
        // SAFETY-LIKE: seed 0 é estado degenerado do xorshift, fixamos para 1.
        Self(if seed == 0 { 1 } else { seed })
    }

    /// Retorna próximo u64 pseudo-aleatório.
    pub const fn next_u64(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.0 = x;
        x
    }

    /// Float em [0, 1) via top 53 bits.
    pub fn next_f64(&mut self) -> f64 {
        ((self.next_u64() >> 11) as f64) / ((1u64 << 53) as f64)
    }

    /// Float em [lo, hi).
    pub fn range_f64(&mut self, lo: f64, hi: f64) -> f64 {
        lo + (hi - lo) * self.next_f64()
    }

    /// Inteiro em [0, n).
    pub fn range_u32(&mut self, n: u32) -> u32 {
        (self.next_u64() % u64::from(n)) as u32
    }
}

#[cfg(test)]
mod rng_tests {
    use super::Rng;

    #[test]
    fn rng_deterministic() {
        let mut a = Rng::new(42);
        let mut b = Rng::new(42);
        for _ in 0..1000 {
            assert_eq!(a.next_u64(), b.next_u64());
        }
    }

    #[test]
    fn rng_distribution_floats_in_unit_interval() {
        let mut r = Rng::new(7);
        for _ in 0..10_000 {
            let v = r.next_f64();
            assert!((0.0..1.0).contains(&v), "f64 out of [0,1): {v}");
        }
    }
}
