//! # Learnability comparison: byte-positional vs char-string
//!
//! Hipótese: streams de byte estruturado têm distribuição mais previsível
//! (lower entropy) que strings de caracteres, tornando a próxima-byte
//! predição mais aprendível por modelos pequenos.
//!
//! Implementação: N-gram Markov model com Laplace smoothing.
//! Métrica: perplexity em test set held-out.

use std::collections::HashMap;

/// N-gram counter com Laplace smoothing.
pub struct NGramModel {
    n: usize,
    /// (n-1)-gram context → (next byte → count)
    counts: HashMap<Vec<u8>, HashMap<u8, u32>>,
    vocab_size: u32,
    alpha: f64, // smoothing
}

impl NGramModel {
    #[must_use]
    pub fn new(n: usize, vocab_size: u32) -> Self {
        assert!(n >= 2, "n must be >= 2");
        Self {
            n,
            counts: HashMap::new(),
            vocab_size,
            alpha: 0.1,
        }
    }

    /// Treina contando n-gramas no stream.
    pub fn train(&mut self, sequence: &[u8]) {
        if sequence.len() < self.n {
            return;
        }
        for i in 0..=sequence.len() - self.n {
            let context = sequence[i..i + self.n - 1].to_vec();
            let next = sequence[i + self.n - 1];
            *self
                .counts
                .entry(context)
                .or_default()
                .entry(next)
                .or_insert(0) += 1;
        }
    }

    /// Treina em múltiplos streams concatenados (com separator entre eles).
    pub fn train_batch(&mut self, sequences: &[Vec<u8>]) {
        for seq in sequences {
            self.train(seq);
        }
    }

    /// log P(next | context) com Laplace smoothing.
    #[must_use]
    pub fn log_prob(&self, context: &[u8], next: u8) -> f64 {
        let row = self.counts.get(context);
        let total: u32 = row.map_or(0, |r| r.values().sum());
        let count = row.and_then(|r| r.get(&next).copied()).unwrap_or(0);
        let p = (f64::from(count) + self.alpha)
            / (f64::from(total) + self.alpha * f64::from(self.vocab_size));
        p.ln()
    }

    /// Perplexity em test set: exp(− mean log P).
    /// Menor = mais previsível.
    #[must_use]
    pub fn perplexity(&self, sequences: &[Vec<u8>]) -> f64 {
        let mut total_log_prob = 0.0;
        let mut total_tokens = 0_u32;
        for seq in sequences {
            if seq.len() < self.n {
                continue;
            }
            for i in 0..=seq.len() - self.n {
                let context = &seq[i..i + self.n - 1];
                let next = seq[i + self.n - 1];
                total_log_prob += self.log_prob(context, next);
                total_tokens += 1;
            }
        }
        if total_tokens == 0 {
            return f64::INFINITY;
        }
        (-total_log_prob / f64::from(total_tokens)).exp()
    }

    #[must_use]
    pub fn distinct_contexts(&self) -> usize {
        self.counts.len()
    }
}

// ---------------- Dataset generator ----------------

/// Operação aritmética simples.
#[derive(Debug, Clone, Copy)]
pub enum Op {
    Add,
    Sub,
    Mul,
}

impl Op {
    pub const fn symbol(self) -> char {
        match self {
            Self::Add => '+',
            Self::Sub => '-',
            Self::Mul => '*',
        }
    }

    pub fn apply(self, a: f64, b: f64) -> f64 {
        match self {
            Self::Add => a + b,
            Self::Sub => a - b,
            Self::Mul => a * b,
        }
    }
}

/// Problem aritmético (a op b = result).
#[derive(Debug, Clone, Copy)]
pub struct Problem {
    pub a: f64,
    pub op: Op,
    pub b: f64,
    pub result: f64,
}

/// Gera N problemas aritméticos com inteiros pequenos (0-99).
#[must_use]
pub fn generate_dataset(n: usize, seed: u64) -> Vec<Problem> {
    let mut rng = crate::Rng::new(seed);
    let mut out = Vec::with_capacity(n);
    let ops = [Op::Add, Op::Sub, Op::Mul];
    for _ in 0..n {
        let a = f64::from(rng.range_u32(100));
        let b = f64::from(rng.range_u32(100));
        let op = ops[rng.range_u32(3) as usize];
        let result = op.apply(a, b);
        out.push(Problem { a, op, b, result });
    }
    out
}

// ---------------- Encoders ----------------

/// Encoda problem como string ASCII: "47+23=70" + terminator.
#[must_use]
pub fn encode_ascii(p: &Problem) -> Vec<u8> {
    format!(
        "{}{}{}={}\n",
        p.a as i64,
        p.op.symbol(),
        p.b as i64,
        p.result as i64
    )
    .into_bytes()
}

/// Encoda problem como byte-positional usando Path B (BCD ordenado).
///
/// Reusa nibble-encoded number representation. Operadores ocupam slots
/// dedicados no campo digit (>10). Terminator = SYM_END do Path B.
#[must_use]
pub fn encode_byte_positional(p: &Problem) -> Vec<u8> {
    use crate::format::path_b;

    let mut out = Vec::with_capacity(20);
    out.extend(path_b::encode(p.a).unwrap());
    // Operador ocupa 1 byte: digit field carrega símbolo, conf = 15
    let op_sym = match p.op {
        Op::Add => 14, // reservado: usamos 14 = ADD
        Op::Sub => 11, // SYM_NEG do path_b reusado contextualmente como op
        Op::Mul => 15, // 15 = MUL
    };
    out.push(path_b::pack(op_sym, path_b::CONF_MAX));
    out.extend(path_b::encode(p.b).unwrap());
    // Equals sign: usamos digit=12 (SYM_E reusado contextualmente como '=')
    out.push(path_b::pack(12, path_b::CONF_MAX));
    out.extend(path_b::encode(p.result).unwrap());
    // Terminator: byte distinto reservado
    out.push(0xEE);
    out
}

// ---------------- Comparison harness ----------------

#[derive(Debug, Clone)]
pub struct LearnabilityResult {
    pub encoding: &'static str,
    pub n_samples: usize,
    pub mean_bytes_per_sample: f64,
    pub vocab_size: usize,
    pub ngram_perplexity_n2: f64,
    pub ngram_perplexity_n3: f64,
    pub ngram_perplexity_n4: f64,
}

/// Treina e avalia n-gram models em um conjunto de sequências.
///
/// Treina em primeiros `train_pct%`, testa nos restantes.
#[must_use]
pub fn evaluate_encoding(
    encoding: &'static str,
    sequences: Vec<Vec<u8>>,
    train_pct: f64,
) -> LearnabilityResult {
    let n_samples = sequences.len();
    let total_bytes: usize = sequences.iter().map(Vec::len).sum();
    let mean_bytes = total_bytes as f64 / n_samples as f64;

    // Calcula vocab observado
    let mut vocab = std::collections::HashSet::new();
    for seq in &sequences {
        for &b in seq {
            vocab.insert(b);
        }
    }
    let vocab_size = vocab.len();

    let split = (sequences.len() as f64 * train_pct) as usize;
    let (train, test) = sequences.split_at(split);

    let mut perp = [0.0_f64; 3];
    for (i, n) in [2_usize, 3, 4].iter().enumerate() {
        let mut model = NGramModel::new(*n, 256);
        model.train_batch(&train.to_vec());
        perp[i] = model.perplexity(&test.to_vec());
    }

    LearnabilityResult {
        encoding,
        n_samples,
        mean_bytes_per_sample: mean_bytes,
        vocab_size,
        ngram_perplexity_n2: perp[0],
        ngram_perplexity_n3: perp[1],
        ngram_perplexity_n4: perp[2],
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ngram_train_predict() {
        let mut model = NGramModel::new(2, 256);
        // "abab" repeated: 'a' followed by 'b', 'b' followed by 'a'
        model.train(b"abababab");
        // P('b' | 'a') should be high
        let log_p_a_to_b = model.log_prob(b"a", b'b');
        let log_p_a_to_x = model.log_prob(b"a", b'x');
        assert!(log_p_a_to_b > log_p_a_to_x);
    }

    #[test]
    fn dataset_generation_deterministic() {
        let d1 = generate_dataset(10, 7);
        let d2 = generate_dataset(10, 7);
        assert_eq!(d1.len(), d2.len());
        for (a, b) in d1.iter().zip(d2.iter()) {
            assert!((a.a - b.a).abs() < 1e-9);
            assert!((a.b - b.b).abs() < 1e-9);
        }
    }

    #[test]
    fn ascii_encoding_works() {
        let p = Problem {
            a: 47.0,
            op: Op::Add,
            b: 23.0,
            result: 70.0,
        };
        let bytes = encode_ascii(&p);
        let s = std::str::from_utf8(&bytes).unwrap();
        assert!(s.contains("47+23=70"));
    }

    #[test]
    fn byte_positional_encoding_distinct_from_ascii() {
        let p = Problem {
            a: 47.0,
            op: Op::Add,
            b: 23.0,
            result: 70.0,
        };
        let ascii = encode_ascii(&p);
        let bp = encode_byte_positional(&p);
        // Both encode the same problem but with different byte content
        assert_ne!(ascii, bp);
    }

    #[test]
    fn learnability_comparison_runs() {
        let problems = generate_dataset(500, 42);
        let ascii_seqs: Vec<Vec<u8>> = problems.iter().map(encode_ascii).collect();
        let bp_seqs: Vec<Vec<u8>> = problems.iter().map(encode_byte_positional).collect();

        let r_ascii = evaluate_encoding("ascii", ascii_seqs, 0.8);
        let r_bp = evaluate_encoding("byte_positional", bp_seqs, 0.8);

        // Sanity: ambos produziram resultados finitos
        assert!(r_ascii.ngram_perplexity_n3.is_finite());
        assert!(r_bp.ngram_perplexity_n3.is_finite());
        // Ambos têm vocab pequeno
        assert!(r_ascii.vocab_size < 256);
        assert!(r_bp.vocab_size < 256);
    }
}
