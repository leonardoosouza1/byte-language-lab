//! # Path A — Posicional puro (BCD com posição biased)
//!
//! Cada byte = `[digit:4][position:4]`.
//!
//! ```text
//! bits 7-4 (digit):
//!   0..=9  → dígito decimal
//!   10     → ponto decimal "."
//!   11     → sinal negativo "-"
//!   12     → "e" (notação científica, futuro)
//!   13     → END marker
//!   14-15  → reservado
//!
//! bits 3-0 (position):
//!   0..=14 → posição (real = encoded - BIAS, BIAS=7)
//!            real_pos ∈ [-7, +7]
//!   15     → POS_ANY (para markers que não têm posição)
//! ```
//!
//! **Range representável:** ~10^-7 a 10^7.
//! **Vantagem-chave:** posição é explícita no byte, addition entre streams
//! é literalmente um digit-by-digit add com carry. Sem FP-ALU.

use super::{DecodeError, EncodeError};

/// Tolerância de round-trip do Path A: 1e-6 relativo OU 5e-8 absoluto.
///
/// Path A tem ~7 dígitos significativos (limitado por MIN_POS=-7).
/// Valores com mais dígitos são truncados com round-half-even.
#[must_use]
pub fn path_a_tolerance(expected: f64, got: f64) -> bool {
    if expected.is_nan() || got.is_nan() {
        return expected.is_nan() && got.is_nan();
    }
    if !expected.is_finite() || !got.is_finite() {
        return expected == got;
    }
    let abs = (expected - got).abs();
    if abs < 5e-8 {
        return true;
    }
    if expected == 0.0 {
        return false;
    }
    abs / expected.abs() < 1e-6
}

/// Bias da posição. Encoded position 0..14 mapeia para real_pos -7..+7.
pub const BIAS: i32 = 7;

/// Posição máxima representável (cobre até 10^7).
pub const MAX_POS: i32 = 7;

/// Posição mínima representável (cobre até 10^-7).
pub const MIN_POS: i32 = -7;

/// Símbolo: ponto decimal (no campo digit).
pub const SYM_DOT: u8 = 10;
/// Símbolo: sinal negativo (no campo digit).
pub const SYM_NEG: u8 = 11;
/// Símbolo: marcador exponencial (reservado para extensão futura).
pub const SYM_E: u8 = 12;
/// Símbolo: end-of-number (delimitador).
pub const SYM_END: u8 = 13;

/// Position encoding usado para markers (sem posição relevante).
pub const POS_ANY: u8 = 15;

/// Empacota digit (4 bits) e posição encoded (4 bits) em 1 byte.
#[inline]
#[must_use]
pub const fn pack(digit: u8, encoded_pos: u8) -> u8 {
    debug_assert!(digit < 16);
    debug_assert!(encoded_pos < 16);
    (digit << 4) | (encoded_pos & 0x0F)
}

/// Desempacota byte em (digit, encoded_pos).
#[inline]
#[must_use]
pub const fn unpack(byte: u8) -> (u8, u8) {
    (byte >> 4, byte & 0x0F)
}

/// Cap absoluto de dígitos significativos (limite de precisão f64 ≈ 15-17).
const MAX_SIG_DIGITS: u32 = 15;

/// Computa posição do dígito mais significativo, defensivo contra precisão de log10.
fn compute_leading_pos(value: f64) -> i32 {
    debug_assert!(value > 0.0 && value.is_finite());
    let raw = value.log10().floor() as i32;
    if (10f64).powi(raw + 1) <= value * (1.0 + 1e-12) {
        raw + 1
    } else if (10f64).powi(raw) > value * (1.0 + 1e-12) {
        raw - 1
    } else {
        raw
    }
}

/// Encoda f64 em stream de bytes Path A.
///
/// Dígitos fracionários são truncados se excederem MIN_POS, com rounding.
/// Ex: 0.123456789 (10 dígitos) é truncado para 0.1234568 (7 dígitos = MIN_POS).
///
/// # Errors
/// - `NotFinite` se x for NaN ou ±Inf.
/// - `OutOfRangeHigh` se |x| ≥ 10^8 (leading_pos > MAX_POS).
/// - `OutOfRangeLow` se 0 < |x| < 10^-7.
pub fn encode(x: f64) -> Result<Vec<u8>, EncodeError> {
    if !x.is_finite() {
        return Err(EncodeError::NotFinite);
    }

    let mut out = Vec::with_capacity(20);

    if x == 0.0 {
        out.push(pack(0, BIAS as u8));
        out.push(pack(SYM_END, POS_ANY));
        return Ok(out);
    }

    let value = if x < 0.0 {
        out.push(pack(SYM_NEG, POS_ANY));
        -x
    } else {
        x
    };

    if value >= 1e8 {
        return Err(EncodeError::OutOfRangeHigh);
    }
    if value < 1e-7 {
        return Err(EncodeError::OutOfRangeLow);
    }

    let leading_pos = compute_leading_pos(value);
    if leading_pos > MAX_POS {
        return Err(EncodeError::OutOfRangeHigh);
    }

    // Máximo de dígitos representáveis: do leading_pos até MIN_POS, capped a f64 precision
    let representable = (leading_pos - MIN_POS + 1) as u32;
    let max_digits = representable.min(MAX_SIG_DIGITS);

    // Escala value para inteiro com max_digits dígitos (aplica round-half-to-even via .round())
    let scale_pow = max_digits as i32 - 1 - leading_pos;
    let scaled = value * (10f64).powi(scale_pow);
    let int_value = scaled.round() as u128;

    // Lida com overflow de rounding (ex: 9.999... → 10.000...)
    let limit = 10u128.pow(max_digits);
    let (int_value, leading_pos) = if int_value >= limit {
        (int_value / 10, leading_pos + 1)
    } else {
        (int_value, leading_pos)
    };

    // Re-checa range após possível shift de leading_pos pelo rounding
    if leading_pos > MAX_POS {
        return Err(EncodeError::OutOfRangeHigh);
    }

    // Extrai dígitos (LSD-first via modulo, depois reverse)
    let mut digits: Vec<u8> = Vec::with_capacity(max_digits as usize);
    let mut remaining = int_value;
    for _ in 0..max_digits {
        digits.push((remaining % 10) as u8);
        remaining /= 10;
    }
    digits.reverse();

    // Trim zeros à direita (insignificantes)
    while digits.len() > 1 && *digits.last().unwrap() == 0 {
        digits.pop();
    }

    // Emite zero inteiro se valor for fracionário puro
    if leading_pos < 0 {
        out.push(pack(0, BIAS as u8));
    }

    let mut dot_emitted = false;
    for (i, &digit) in digits.iter().enumerate() {
        let pos = leading_pos - (i as i32);
        if pos < 0 && !dot_emitted {
            out.push(pack(SYM_DOT, POS_ANY));
            dot_emitted = true;
        }
        out.push(pack(digit, (pos + BIAS) as u8));
    }

    out.push(pack(SYM_END, POS_ANY));
    Ok(out)
}

/// Decoda stream de bytes Path A → f64.
///
/// # Errors
/// - `Empty` se slice vazio.
/// - `InvalidByte` se símbolo desconhecido.
/// - `UnexpectedEnd` se faltou marcador END.
pub fn decode(bytes: &[u8]) -> Result<f64, DecodeError> {
    if bytes.is_empty() {
        return Err(DecodeError::Empty);
    }

    let mut value = 0.0_f64;
    let mut sign = 1.0_f64;
    let mut found_end = false;

    for &b in bytes {
        let (sym, pos_enc) = unpack(b);
        match sym {
            SYM_END => {
                found_end = true;
                break;
            }
            SYM_NEG => sign = -1.0,
            // DOT: posição é informação implícita (já no campo position de cada digit),
            // marker é puramente legibilidade humana / debug. Ignorável no decoder.
            SYM_DOT | SYM_E => (),
            0..=9 => {
                let real_pos = i32::from(pos_enc) - BIAS;
                value += f64::from(sym) * (10f64).powi(real_pos);
            }
            _ => return Err(DecodeError::InvalidByte(b)),
        }
    }

    if !found_end {
        return Err(DecodeError::UnexpectedEnd);
    }

    Ok(sign * value)
}

/// Tamanho típico em bytes para representar `x`. Útil para compactness benchmarks.
#[must_use]
pub fn encoded_size(x: f64) -> Option<usize> {
    encode(x).ok().map(|v| v.len())
}

#[cfg(test)]
mod tests {
    use super::*;
    use super::path_a_tolerance as within_tolerance;

    #[test]
    fn pack_unpack_roundtrip() {
        for d in 0..16 {
            for p in 0..16 {
                let b = pack(d, p);
                let (du, pu) = unpack(b);
                assert_eq!(du, d);
                assert_eq!(pu, p);
            }
        }
    }

    #[test]
    fn encode_zero() {
        let bytes = encode(0.0).unwrap();
        assert_eq!(bytes.len(), 2); // [0,0] + END
        let v = decode(&bytes).unwrap();
        assert_eq!(v, 0.0);
    }

    #[test]
    fn encode_simple_integer() {
        let bytes = encode(47.0).unwrap();
        // Esperamos: [4 at pos 1][7 at pos 0][END]
        assert_eq!(bytes.len(), 3);
        let v = decode(&bytes).unwrap();
        assert!(within_tolerance(47.0, v), "got {v}");
    }

    #[test]
    fn encode_simple_fractional() {
        let bytes = encode(47.83).unwrap();
        // Esperamos: [4][7][.][8][3][END]
        let v = decode(&bytes).unwrap();
        assert!(within_tolerance(47.83, v), "got {v}");
    }

    #[test]
    fn encode_negative() {
        let bytes = encode(-3.14).unwrap();
        // Esperamos: [-][3][.][1][4][END]
        let v = decode(&bytes).unwrap();
        assert!(within_tolerance(-3.14, v), "got {v}");
    }

    #[test]
    fn encode_small_fraction() {
        let bytes = encode(0.000_5).unwrap();
        let v = decode(&bytes).unwrap();
        assert!(within_tolerance(0.000_5, v), "got {v}");
    }

    #[test]
    fn encode_user_example() {
        // Exemplo central da hipótese: 0.4847
        let bytes = encode(0.4847).unwrap();
        let v = decode(&bytes).unwrap();
        assert!(within_tolerance(0.4847, v), "got {v}");
    }

    #[test]
    fn encode_out_of_range_high() {
        assert_eq!(encode(1e8), Err(EncodeError::OutOfRangeHigh));
        assert_eq!(encode(-1e9), Err(EncodeError::OutOfRangeHigh));
    }

    #[test]
    fn encode_out_of_range_low() {
        assert_eq!(encode(1e-8), Err(EncodeError::OutOfRangeLow));
        assert_eq!(encode(-1e-10), Err(EncodeError::OutOfRangeLow));
    }

    #[test]
    fn encode_not_finite() {
        assert_eq!(encode(f64::NAN), Err(EncodeError::NotFinite));
        assert_eq!(encode(f64::INFINITY), Err(EncodeError::NotFinite));
        assert_eq!(encode(f64::NEG_INFINITY), Err(EncodeError::NotFinite));
    }

    #[test]
    fn decode_empty() {
        assert_eq!(decode(&[]), Err(DecodeError::Empty));
    }

    #[test]
    fn decode_unexpected_end() {
        // sem marcador END
        let bytes = vec![pack(4, (1 + BIAS) as u8), pack(7, BIAS as u8)];
        assert_eq!(decode(&bytes), Err(DecodeError::UnexpectedEnd));
    }

    #[test]
    fn property_roundtrip_integers() {
        // Testa todos inteiros em [-1000, 1000]
        for i in -1000..=1000 {
            let x = f64::from(i);
            let bytes = encode(x).unwrap();
            let y = decode(&bytes).unwrap();
            assert!(within_tolerance(x, y), "x={x} y={y}");
        }
    }

    #[test]
    fn property_roundtrip_random() {
        let mut rng = crate::Rng::new(42);
        let mut failures = 0;
        let mut tested = 0;
        let n = 10_000;
        for _ in 0..n {
            let raw = rng.range_f64(-1e6, 1e6);
            // Encode-or-skip: aceita que valores fora de range são erro esperado
            let bytes = match encode(raw) {
                Ok(b) => b,
                Err(_) => continue,
            };
            tested += 1;
            let y = decode(&bytes).unwrap();
            if !within_tolerance(raw, y) {
                failures += 1;
                if failures <= 5 {
                    eprintln!("MISMATCH: x={raw} y={y} bytes={bytes:?}");
                }
            }
        }
        assert!(tested > 9_000, "{tested}/{n} encoded successfully");
        assert_eq!(failures, 0, "{failures} round-trip failures of {tested}");
    }

    #[test]
    fn property_roundtrip_small_fractions() {
        let mut rng = crate::Rng::new(7);
        let mut failures = 0;
        let mut tested = 0;
        for _ in 0..1000 {
            let x = rng.range_f64(-1.0, 1.0);
            let bytes = match encode(x) {
                Ok(b) => b,
                Err(_) => continue,
            };
            tested += 1;
            let y = decode(&bytes).unwrap();
            if !within_tolerance(x, y) {
                failures += 1;
                if failures <= 3 {
                    eprintln!("MISMATCH: x={x} y={y}");
                }
            }
        }
        assert!(tested > 800, "{tested}/1000 encoded");
        assert_eq!(failures, 0, "{failures} small-fraction failures");
    }

    #[test]
    fn property_roundtrip_large_integers() {
        let mut rng = crate::Rng::new(123);
        for _ in 0..1000 {
            let x = rng.range_f64(1.0, 1e7).round();
            let bytes = encode(x).unwrap();
            let y = decode(&bytes).unwrap();
            assert!(within_tolerance(x, y), "x={x} y={y}");
        }
    }
}
