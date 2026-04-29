//! Três variantes de formato de byte estruturado.
//!
//! - [`path_a`]: posicional puro (BCD com posição biased).
//! - [`path_b`]: símbolos ricos com flags compostáveis.
//! - [`path_c`]: domain-tagged (header+payload), suporta multi-domínio.
//!
//! Todas oferecem `encode(f64) -> Vec<u8>` e `decode(&[u8]) -> Result<f64, _>`,
//! e são validadas contra round-trip + tolerância numérica.

pub mod path_a;
pub mod path_b;
pub mod path_c;

/// Erro de encoding (número fora do range suportado pelo formato).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EncodeError {
    /// Magnitude muito grande para o formato representar.
    OutOfRangeHigh,
    /// Magnitude muito pequena (subnormal/underflow).
    OutOfRangeLow,
    /// Valor não-finito (NaN, ±Inf).
    NotFinite,
}

/// Erro de decoding (stream malformado).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DecodeError {
    /// Stream vazio.
    Empty,
    /// Byte com role/symbol desconhecido.
    InvalidByte(u8),
    /// Faltou marcador de fim antes do final do slice.
    UnexpectedEnd,
    /// Sequência inválida (ex: dois '.' seguidos).
    MalformedSequence,
}

/// Tolerância padrão para round-trip: ε absoluto de 1e-12 OU ε relativo de 1e-8.
///
/// Quando `expected == 0`, fallback para apenas absoluto (relativo indefinido).
/// Adequado para byte formats com ~10 dígitos significativos preservados.
#[must_use]
pub fn within_tolerance(expected: f64, got: f64) -> bool {
    if expected.is_nan() || got.is_nan() {
        return expected.is_nan() && got.is_nan();
    }
    if !expected.is_finite() || !got.is_finite() {
        return expected == got;
    }
    let abs = (expected - got).abs();
    if abs < 1e-12 {
        return true;
    }
    if expected == 0.0 {
        return false; // já cobrimos abs < 1e-12 acima
    }
    abs / expected.abs() < 1e-8
}

#[cfg(test)]
mod tolerance_tests {
    use super::within_tolerance;

    #[test]
    fn exact_match() {
        assert!(within_tolerance(1.0, 1.0));
        assert!(within_tolerance(0.0, 0.0));
        assert!(within_tolerance(-3.14, -3.14));
    }

    #[test]
    fn close_match() {
        assert!(within_tolerance(1.0, 1.0 + 1e-13));
        assert!(within_tolerance(1e6, 1e6 + 1e-3));
    }

    #[test]
    fn far_mismatch() {
        assert!(!within_tolerance(1.0, 1.001));
        assert!(!within_tolerance(0.0, 1e-10));
    }

    #[test]
    fn nan_special() {
        assert!(within_tolerance(f64::NAN, f64::NAN));
        assert!(!within_tolerance(f64::NAN, 0.0));
    }
}
