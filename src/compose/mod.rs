//! # Multi-domain composition: o "byte-language" unificado
//!
//! Demonstra que números, texto, comandos, estados FSM e booleans podem
//! coexistir em UM único stream de bytes — a tese central que distingue
//! "byte-language" de tokens (que tem 1 vocabulário fixo) ou floats
//! (que são apenas números).
//!
//! Usa Path C (domain-tagged) como substrato.

pub use crate::format::path_c::Block;
use crate::format::path_c;
use crate::format::{DecodeError, EncodeError};

/// Sentença = sequência ordenada de blocos tipados.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct Sentence {
    pub blocks: Vec<Block>,
}

impl Sentence {
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    pub fn number(mut self, x: f64) -> Self {
        self.blocks.push(Block::Number(x));
        self
    }

    pub fn text(mut self, s: impl Into<String>) -> Self {
        self.blocks.push(Block::Text(s.into()));
        self
    }

    pub fn command(mut self, action: u8, magnitude: u8) -> Self {
        self.blocks.push(Block::Command { action, magnitude });
        self
    }

    pub fn state(mut self, fsm_byte: u8) -> Self {
        self.blocks.push(Block::State(fsm_byte));
        self
    }

    pub fn boolean(mut self, b: bool) -> Self {
        self.blocks.push(Block::Boolean(b));
        self
    }

    pub fn percent(mut self, p: u8) -> Self {
        self.blocks.push(Block::Percent(p));
        self
    }

    /// Encoda toda a sentença em um único stream de bytes.
    ///
    /// # Errors
    /// Propaga erros de encoding de Path C (apenas `NotFinite` ou range em Number).
    pub fn encode(&self) -> Result<Vec<u8>, EncodeError> {
        let mut out = Vec::new();
        for block in &self.blocks {
            let bytes = match block {
                Block::Number(x) => path_c::encode_number(*x)?,
                Block::Text(s) => path_c::encode_text(s)?,
                Block::Command { action, magnitude } => {
                    path_c::encode_command(*action, *magnitude)
                }
                Block::State(b) => path_c::encode_state(*b),
                Block::Boolean(b) => path_c::encode_boolean(*b),
                Block::Percent(p) => path_c::encode_percent(*p),
            };
            out.extend_from_slice(&bytes);
        }
        Ok(out)
    }

    /// Decoda stream → Sentence.
    ///
    /// # Errors
    /// Propaga erros de Path C decode_stream.
    pub fn decode(bytes: &[u8]) -> Result<Self, DecodeError> {
        Ok(Self {
            blocks: path_c::decode_stream(bytes)?,
        })
    }
}

/// Métricas de uma sentence: byte count, domínios usados, ratio header/payload.
#[derive(Debug, Clone)]
pub struct SentenceStats {
    pub total_bytes: usize,
    pub block_count: usize,
    pub mean_bytes_per_block: f64,
    pub domains_used: Vec<&'static str>,
    pub header_overhead_pct: f64,
}

#[must_use]
pub fn analyze(sentence: &Sentence) -> SentenceStats {
    let bytes = sentence.encode().unwrap_or_default();
    let total_bytes = bytes.len();
    let block_count = sentence.blocks.len();
    let mean_bytes = if block_count > 0 {
        total_bytes as f64 / block_count as f64
    } else {
        0.0
    };

    let mut domains_used = Vec::new();
    for block in &sentence.blocks {
        let name = match block {
            Block::Number(_) => "Number",
            Block::Text(_) => "Text",
            Block::Command { .. } => "Command",
            Block::State(_) => "State",
            Block::Boolean(_) => "Boolean",
            Block::Percent(_) => "Percent",
        };
        if !domains_used.contains(&name) {
            domains_used.push(name);
        }
    }

    // Header overhead: blocos curtos têm 1 byte header + payload.
    // Para length=0 long blocks: 1 header + 1 EOB = 2 bytes overhead.
    let headers = block_count + sentence.blocks.iter().filter(|b| match b {
        Block::Text(s) => s.len() > 15,
        _ => false,
    }).count();
    let header_overhead_pct = if total_bytes > 0 {
        100.0 * headers as f64 / total_bytes as f64
    } else {
        0.0
    };

    SentenceStats {
        total_bytes,
        block_count,
        mean_bytes_per_block: mean_bytes,
        domains_used,
        header_overhead_pct,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn build_and_roundtrip() {
        let s = Sentence::new()
            .number(42.0)
            .text("status:")
            .boolean(true)
            .percent(85)
            .command(3, 100)
            .state(0x9D);

        let bytes = s.encode().unwrap();
        let decoded = Sentence::decode(&bytes).unwrap();
        assert_eq!(decoded.blocks.len(), 6);
        assert!(matches!(decoded.blocks[0], Block::Number(n) if (n - 42.0).abs() < 1e-9));
        assert!(matches!(&decoded.blocks[1], Block::Text(t) if t == "status:"));
        assert!(matches!(decoded.blocks[2], Block::Boolean(true)));
        assert!(matches!(decoded.blocks[3], Block::Percent(85)));
        assert!(matches!(decoded.blocks[4], Block::Command { action: 3, magnitude: 100 }));
        assert!(matches!(decoded.blocks[5], Block::State(0x9D)));
    }

    #[test]
    fn analyze_counts_correctly() {
        let s = Sentence::new()
            .number(1.0)
            .text("hi")
            .boolean(false);
        let stats = analyze(&s);
        assert_eq!(stats.block_count, 3);
        assert!(stats.total_bytes > 3); // ao menos 1 byte por block
        assert_eq!(stats.domains_used.len(), 3);
    }

    #[test]
    fn driving_command_sequence() {
        // Demo: stream típico de uma "linguagem de inteligência"
        // contendo número, texto, comando e estado num único frame
        let s = Sentence::new()
            .text("frame:")
            .number(127.5)               // posição x
            .number(-3.2)                // velocidade
            .percent(60)                 // throttle
            .percent(0)                  // brake
            .number(0.15)                // steer angle
            .state(0x9D)                 // FSM cortex byte
            .boolean(false)              // emergency flag
            .command(2, 100);            // ação composta

        let bytes = s.encode().unwrap();
        let restored = Sentence::decode(&bytes).unwrap();
        assert_eq!(restored.blocks.len(), 9);
    }
}
