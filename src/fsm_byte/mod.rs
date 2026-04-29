//! # FSM byte semantics — substrato unificado com fsm-cognitive-architecture
//!
//! O byte de estado FSM tem layout idêntico ao Path C State domain:
//!
//! ```text
//! bits 7-5: família (8 tipos)
//!   0 INTEGRADOR   - acumula até disparar
//!   1 DIFERENCIADOR - responde a mudança
//!   2 RESSONADOR   - sincroniza com frequência
//!   3 INIBIDOR     - suprime atividade vizinha
//!   4 AMPLIFICADOR - reforça sinal
//!   5 COMPARADOR   - dispara em diferença
//!   6 MEMORIZADOR  - mantém estado
//!   7 ALEATORIZADOR - injeta ruído
//!
//! bits 4-2: nível/sub-estado (8 níveis dentro da família)
//! bit 1:    escala temporal (0=reflexo 1ms, 1=contexto 100ms)
//! bit 0:    spike flag
//! ```
//!
//! Isso unifica o byte do `fsm-cognitive-architecture` com o byte-language
//! deste lab — o MESMO formato de 8 bits estruturados serve para "estado
//! cognitivo" (FSM) e "valor numérico" (Path A) dependendo do domain header.

/// As 8 famílias FSM canônicas (ADR-005 do fsm-cognitive-architecture).
#[repr(u8)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Familia {
    Integrador = 0,
    Diferenciador = 1,
    Ressonador = 2,
    Inibidor = 3,
    Amplificador = 4,
    Comparador = 5,
    Memorizador = 6,
    Aleatorizador = 7,
}

impl Familia {
    #[must_use]
    pub const fn from_u8(v: u8) -> Option<Self> {
        match v & 0x07 {
            0 => Some(Self::Integrador),
            1 => Some(Self::Diferenciador),
            2 => Some(Self::Ressonador),
            3 => Some(Self::Inibidor),
            4 => Some(Self::Amplificador),
            5 => Some(Self::Comparador),
            6 => Some(Self::Memorizador),
            7 => Some(Self::Aleatorizador),
            _ => None,
        }
    }

    #[must_use]
    pub const fn name(self) -> &'static str {
        match self {
            Self::Integrador => "INTEGRADOR",
            Self::Diferenciador => "DIFERENCIADOR",
            Self::Ressonador => "RESSONADOR",
            Self::Inibidor => "INIBIDOR",
            Self::Amplificador => "AMPLIFICADOR",
            Self::Comparador => "COMPARADOR",
            Self::Memorizador => "MEMORIZADOR",
            Self::Aleatorizador => "ALEATORIZADOR",
        }
    }
}

/// Escala temporal do neurônio FSM.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Escala {
    /// Reflexo (~1ms tick)
    Reflexo = 0,
    /// Contexto (~100ms tick)
    Contexto = 1,
}

/// Estado FSM completamente decodificado.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct EstadoFsm {
    pub familia: Familia,
    pub nivel: u8,    // 0..=7
    pub escala: Escala,
    pub spike: bool,
}

impl EstadoFsm {
    /// Constrói EstadoFsm a partir de byte raw.
    #[must_use]
    pub fn from_byte(b: u8) -> Self {
        let familia = Familia::from_u8(b >> 5).unwrap_or(Familia::Integrador);
        let nivel = (b >> 2) & 0x07;
        let escala = if (b >> 1) & 1 == 1 { Escala::Contexto } else { Escala::Reflexo };
        let spike = b & 1 == 1;
        Self { familia, nivel, escala, spike }
    }

    /// Codifica EstadoFsm como byte raw.
    #[must_use]
    pub const fn to_byte(self) -> u8 {
        let f = self.familia as u8;
        let n = self.nivel & 0x07;
        let s = match self.escala {
            Escala::Reflexo => 0,
            Escala::Contexto => 1,
        };
        let spike = if self.spike { 1 } else { 0 };
        (f << 5) | (n << 2) | (s << 1) | spike
    }

    /// Estado em repouso (vazio, sem spike).
    #[must_use]
    pub const fn rest(familia: Familia) -> Self {
        Self {
            familia,
            nivel: 0,
            escala: Escala::Reflexo,
            spike: false,
        }
    }

    /// Estado disparando no nível máximo.
    #[must_use]
    pub const fn spiking(familia: Familia, nivel: u8) -> Self {
        Self {
            familia,
            nivel,
            escala: Escala::Reflexo,
            spike: true,
        }
    }

    /// Decodifica para representação textual (debug-friendly).
    #[must_use]
    pub fn describe(self) -> String {
        format!(
            "{}[{}] {} {}",
            self.familia.name(),
            self.nivel,
            if matches!(self.escala, Escala::Contexto) { "ctx" } else { "rfx" },
            if self.spike { "🔥" } else { "·" }
        )
    }
}

/// Microcoluna = 8 neurônios FSM = 8 bytes (u64 packed).
#[repr(C, align(8))]
#[derive(Debug, Clone, Copy, Default)]
pub struct Microcoluna {
    pub neurons: [u8; 8],
}

impl Microcoluna {
    #[must_use]
    pub fn from_estados(estados: [EstadoFsm; 8]) -> Self {
        let mut neurons = [0_u8; 8];
        for (i, e) in estados.iter().enumerate() {
            neurons[i] = e.to_byte();
        }
        Self { neurons }
    }

    #[must_use]
    pub fn estados(&self) -> [EstadoFsm; 8] {
        let mut out = [EstadoFsm::rest(Familia::Integrador); 8];
        for (i, &b) in self.neurons.iter().enumerate() {
            out[i] = EstadoFsm::from_byte(b);
        }
        out
    }

    /// View como u64 (1 registrador GPU).
    #[must_use]
    pub fn as_u64(&self) -> u64 {
        u64::from_le_bytes(self.neurons)
    }

    /// Spike rate da microcoluna (popcount dos spikes / 8).
    #[must_use]
    pub fn spike_rate(&self) -> f32 {
        let mut count = 0_u32;
        for &b in &self.neurons {
            count += u32::from(b & 1);
        }
        count as f32 / 8.0
    }

    /// Família dominante (maior contagem entre os 8 neurônios).
    #[must_use]
    pub fn familia_dominante(&self) -> Familia {
        let mut counts = [0_u32; 8];
        for &b in &self.neurons {
            counts[(b >> 5) as usize] += 1;
        }
        let mut best = 0;
        let mut best_count = 0;
        for (i, &c) in counts.iter().enumerate() {
            if c > best_count {
                best_count = c;
                best = i;
            }
        }
        Familia::from_u8(best as u8).unwrap()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn byte_roundtrip_all() {
        // Todos 256 bytes possíveis devem round-trip via EstadoFsm
        for b in 0_u16..=255 {
            let b = b as u8;
            let est = EstadoFsm::from_byte(b);
            assert_eq!(est.to_byte(), b, "roundtrip failed for byte {:#x}", b);
        }
    }

    #[test]
    fn integrador_repouso() {
        let est = EstadoFsm::rest(Familia::Integrador);
        assert_eq!(est.to_byte(), 0x00);
    }

    #[test]
    fn aleatorizador_max_spiking() {
        let est = EstadoFsm {
            familia: Familia::Aleatorizador,
            nivel: 7,
            escala: Escala::Contexto,
            spike: true,
        };
        // 111 111 1 1 = 0xFF
        assert_eq!(est.to_byte(), 0xFF);
    }

    #[test]
    fn microcoluna_packed_in_u64() {
        let estados = [
            EstadoFsm::spiking(Familia::Integrador, 7),
            EstadoFsm::spiking(Familia::Diferenciador, 5),
            EstadoFsm::rest(Familia::Ressonador),
            EstadoFsm::rest(Familia::Inibidor),
            EstadoFsm::spiking(Familia::Amplificador, 3),
            EstadoFsm::rest(Familia::Comparador),
            EstadoFsm::rest(Familia::Memorizador),
            EstadoFsm::spiking(Familia::Aleatorizador, 7),
        ];
        let mc = Microcoluna::from_estados(estados);
        let u = mc.as_u64();
        // u64 deve ter 8 bytes não-zero (cada um um neurônio)
        assert!(u != 0);
        // Round-trip
        let est_back = mc.estados();
        for (a, b) in estados.iter().zip(est_back.iter()) {
            assert_eq!(a, b);
        }
    }

    #[test]
    fn spike_rate_correct() {
        let estados = [
            EstadoFsm::spiking(Familia::Integrador, 7),
            EstadoFsm::rest(Familia::Integrador),
            EstadoFsm::spiking(Familia::Integrador, 5),
            EstadoFsm::rest(Familia::Integrador),
            EstadoFsm::spiking(Familia::Integrador, 3),
            EstadoFsm::rest(Familia::Integrador),
            EstadoFsm::rest(Familia::Integrador),
            EstadoFsm::rest(Familia::Integrador),
        ];
        let mc = Microcoluna::from_estados(estados);
        // 3 de 8 spikes = 0.375
        assert!((mc.spike_rate() - 0.375).abs() < 1e-6);
    }
}
