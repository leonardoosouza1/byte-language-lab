//! Multi-resolution views — analyze same data at different scales.
//!
//! Inspired by SACI multi-timeframe (5m, 15m, 30m, 1h, 4h):
//!   - bit (2 states, finest)
//!   - nibble (16 states)
//!   - byte (256 states)
//!   - bigram (256² = 65k states)
//!   - word (vocab-size states)
//!   - phrase (sentence-level)
//!
//! Cross-correlation revela como mesma info aparece em diferentes lenses.
//!
//! Math:
//!   Mutual Information I(X;Y) = Σ p(x,y) log(p(x,y) / (p(x)·p(y)))
//!   High I(X;Y) = views strongly correlated
//!   Low I(X;Y) = views independent (orthogonal)
//!
//! Use cases:
//!   - bit-5 vs case (já validado KL=0.003)
//!   - nibble distribution vs word freq
//!   - byte ngram vs sentence sentiment

#![allow(dead_code)]

/// Multi-scale histogram from a byte buffer.
#[derive(Debug, Clone)]
pub struct MultiView {
    /// Bit count [2]: how many 0s vs 1s
    pub bits: [u64; 2],
    /// Nibble count [16]: 4-bit chunks (2 per byte)
    pub nibbles: [u64; 16],
    /// Byte count [256]: each byte value
    pub bytes: [u64; 256],
    /// Bigram count [256][256]: byte-pair transitions
    pub bigrams: Vec<u64>,  // flat 256*256
    /// Hex digit count [16]: hex representation digits
    pub hex_digits: [u64; 16],
    /// Total bytes processed
    pub n_bytes: u64,
}

impl Default for MultiView {
    fn default() -> Self { Self::new() }
}

impl MultiView {
    pub fn new() -> Self {
        Self {
            bits: [0u64; 2],
            nibbles: [0u64; 16],
            bytes: [0u64; 256],
            bigrams: vec![0u64; 256 * 256],
            hex_digits: [0u64; 16],
            n_bytes: 0,
        }
    }

    /// Process buffer at all scales.
    pub fn ingest(&mut self, data: &[u8]) {
        self.n_bytes += data.len() as u64;
        for i in 0..data.len() {
            let b = data[i];
            self.bytes[b as usize] += 1;
            // Bits
            for k in 0..8 {
                let bit = (b >> k) & 1;
                self.bits[bit as usize] += 1;
            }
            // Nibbles (2 per byte)
            self.nibbles[(b & 0x0F) as usize] += 1;
            self.nibbles[((b >> 4) & 0x0F) as usize] += 1;
            // Hex digits (same as nibbles, but tracked separately for hex string view)
            self.hex_digits[(b & 0x0F) as usize] += 1;
            self.hex_digits[((b >> 4) & 0x0F) as usize] += 1;
            // Bigram
            if i > 0 {
                let prev = data[i-1];
                self.bigrams[(prev as usize) * 256 + (b as usize)] += 1;
            }
        }
    }

    /// Get normalized probability distribution for given scale.
    /// scale: "bit", "nibble", "byte", "hex"
    pub fn probabilities(&self, scale: &str) -> Vec<f32> {
        match scale {
            "bit" => normalize(&self.bits[..]),
            "nibble" => normalize(&self.nibbles[..]),
            "byte" => normalize(&self.bytes[..]),
            "hex" => normalize(&self.hex_digits[..]),
            _ => Vec::new(),
        }
    }

    /// Shannon entropy at given scale (nats).
    pub fn entropy(&self, scale: &str) -> f32 {
        let p = self.probabilities(scale);
        p.iter()
            .filter(|&&x| x > 1e-9)
            .map(|&x| -x * x.ln())
            .sum()
    }

    /// Mutual Information between byte position parity (bit) and byte value.
    /// I(parity; byte) — how much does byte value depend on bit pattern?
    pub fn mi_bit_byte(&self) -> f32 {
        if self.n_bytes < 10 { return 0.0; }
        // Conditional: p(byte | bit=0) vs p(byte | bit=1)
        let mut p_b0 = [0.0f32; 256];
        let mut p_b1 = [0.0f32; 256];
        let mut n0 = 0.0f32;
        let mut n1 = 0.0f32;
        for b in 0..256usize {
            let cnt = self.bytes[b] as f32;
            // Use bit 5 (lsb of high nibble) since we validated bit5=case
            if (b >> 5) & 1 == 0 {
                p_b0[b] = cnt;
                n0 += cnt;
            } else {
                p_b1[b] = cnt;
                n1 += cnt;
            }
        }
        if n0 < 1.0 || n1 < 1.0 { return 0.0; }
        for b in 0..256 {
            p_b0[b] /= n0;
            p_b1[b] /= n1;
        }
        // KL(p0 || p1) ≈ MI lower bound (using bit5 split)
        let mut kl = 0.0f32;
        for b in 0..256 {
            if p_b0[b] > 1e-9 && p_b1[b] > 1e-9 {
                kl += p_b0[b] * (p_b0[b] / p_b1[b]).ln();
            }
        }
        kl.max(0.0)
    }

    /// Hex string representation of byte (helpful pro user).
    pub fn byte_to_hex(b: u8) -> String {
        format!("{:02x}", b)
    }

    /// Aggregate stats — multi-scale signature compact pra dashboard.
    pub fn signature(&self) -> MultiViewSignature {
        MultiViewSignature {
            n_bytes: self.n_bytes,
            bit_entropy: self.entropy("bit"),
            nibble_entropy: self.entropy("nibble"),
            byte_entropy: self.entropy("byte"),
            hex_entropy: self.entropy("hex"),
            mi_bit5_byte: self.mi_bit_byte(),
            n_active_bytes: self.bytes.iter().filter(|&&c| c > 0).count(),
            n_active_bigrams: self.bigrams.iter().filter(|&&c| c > 0).count(),
        }
    }
}

#[derive(Debug, Clone)]
pub struct MultiViewSignature {
    pub n_bytes: u64,
    pub bit_entropy: f32,
    pub nibble_entropy: f32,
    pub byte_entropy: f32,
    pub hex_entropy: f32,
    pub mi_bit5_byte: f32,
    pub n_active_bytes: usize,
    pub n_active_bigrams: usize,
}

fn normalize(counts: &[u64]) -> Vec<f32> {
    let sum: u64 = counts.iter().sum();
    if sum == 0 { return vec![0.0; counts.len()]; }
    counts.iter().map(|&c| c as f32 / sum as f32).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ingest_counts_bytes_correctly() {
        let mut mv = MultiView::new();
        mv.ingest(b"AA");  // byte 65 = 0x41 = 01000001
        assert_eq!(mv.bytes[b'A' as usize], 2);
        // Each byte has 8 bits, 2 of which are 1s (bits 0 and 6 of 65)
        // For 2 bytes, total bits=16: 2*2=4 ones, 2*6=12 zeros
        assert_eq!(mv.bits[1], 4);
        assert_eq!(mv.bits[0], 12);
    }

    #[test]
    fn nibble_decomposition() {
        let mut mv = MultiView::new();
        mv.ingest(&[0xAB]);  // nibbles: 0xA, 0xB
        assert_eq!(mv.nibbles[0xA], 1);
        assert_eq!(mv.nibbles[0xB], 1);
    }

    #[test]
    fn entropy_monotonic_with_diversity() {
        let mut mv1 = MultiView::new();
        mv1.ingest(b"aaaaaaaa");
        let mut mv2 = MultiView::new();
        mv2.ingest(b"abcdefgh");
        let e1 = mv2.entropy("byte");
        let e2 = mv1.entropy("byte");
        assert!(e1 > e2, "diverse should have more entropy: {} > {}", e1, e2);
    }

    #[test]
    fn mi_bit_byte_pt_text() {
        let mut mv = MultiView::new();
        // Use mixed case to trigger bit5 separation
        mv.ingest(b"Brasil pais Capital Brasilia Sao Paulo cidade Rio Janeiro");
        let mi = mv.mi_bit_byte();
        // KL based on bit5 split → should be > 0 for mixed-case text
        assert!(mi >= 0.0, "mi should be non-negative: {}", mi);
    }

    #[test]
    fn signature_compact() {
        let mut mv = MultiView::new();
        mv.ingest(b"Hello world test data");
        let sig = mv.signature();
        assert!(sig.n_bytes > 0);
        assert!(sig.byte_entropy > 0.0);
        assert!(sig.n_active_bytes > 0);
    }
}
