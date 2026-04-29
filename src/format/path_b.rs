//! # Path B — BCD ordenado com confidence
//!
//! Cada byte = `[digit:4][confidence:4]`.
//!
//! Posição é IMPLÍCITA (vem da ordem dos bytes). Confidence (0-15) é explícita
//! e útil para output de modelos probabilísticos. Mais compacto que Path A
//! para escrita sequencial.

use super::{DecodeError, EncodeError};

pub const SYM_DOT: u8 = 10;
pub const SYM_NEG: u8 = 11;
pub const SYM_E: u8 = 12;
pub const SYM_END: u8 = 13;
pub const CONF_MAX: u8 = 15;

#[inline]
#[must_use]
pub const fn pack(digit: u8, conf: u8) -> u8 {
    debug_assert!(digit < 16);
    debug_assert!(conf < 16);
    (digit << 4) | (conf & 0x0F)
}

#[inline]
#[must_use]
pub const fn unpack(byte: u8) -> (u8, u8) {
    (byte >> 4, byte & 0x0F)
}

/// Encoda f64 em Path B com confidence máxima.
///
/// # Errors
/// `NotFinite` para NaN/Inf.
pub fn encode(x: f64) -> Result<Vec<u8>, EncodeError> {
    encode_with_confidence(x, CONF_MAX)
}

/// Encoda com confidence customizada por byte (uniforme).
///
/// # Errors
/// `NotFinite` para NaN/Inf.
pub fn encode_with_confidence(x: f64, conf: u8) -> Result<Vec<u8>, EncodeError> {
    if !x.is_finite() {
        return Err(EncodeError::NotFinite);
    }
    let conf = conf.min(CONF_MAX);
    let mut out = Vec::with_capacity(20);

    if x == 0.0 {
        out.push(pack(0, conf));
        out.push(pack(SYM_END, conf));
        return Ok(out);
    }

    let value = if x < 0.0 {
        out.push(pack(SYM_NEG, conf));
        -x
    } else {
        x
    };

    // Usa f64 → exp formatação para extrair dígitos com precisão máxima.
    let s = format!("{value:.15e}");
    let parts: Vec<&str> = s.splitn(2, 'e').collect();
    let mantissa = parts[0];
    let exp: i32 = parts[1].parse().map_err(|_| EncodeError::NotFinite)?;

    let chars: Vec<char> = mantissa.chars().collect();
    let lead_digit = chars[0].to_digit(10).unwrap() as u8;
    let mut frac_digits: Vec<u8> = chars[2..]
        .iter()
        .map(|c| c.to_digit(10).unwrap() as u8)
        .collect();
    while !frac_digits.is_empty() && *frac_digits.last().unwrap() == 0 {
        frac_digits.pop();
    }

    if exp >= 0 {
        let int_digits = (exp + 1) as usize;
        let mut all_digits = vec![lead_digit];
        all_digits.extend_from_slice(&frac_digits);
        while all_digits.len() < int_digits {
            all_digits.push(0);
        }
        for &d in &all_digits[..int_digits] {
            out.push(pack(d, conf));
        }
        if all_digits.len() > int_digits {
            let frac = &all_digits[int_digits..];
            if frac.iter().any(|&d| d != 0) {
                out.push(pack(SYM_DOT, conf));
                let mut last_nonzero = frac.len();
                while last_nonzero > 0 && frac[last_nonzero - 1] == 0 {
                    last_nonzero -= 1;
                }
                for &d in &frac[..last_nonzero] {
                    out.push(pack(d, conf));
                }
            }
        }
    } else {
        out.push(pack(0, conf));
        out.push(pack(SYM_DOT, conf));
        for _ in 0..(-exp - 1) {
            out.push(pack(0, conf));
        }
        out.push(pack(lead_digit, conf));
        for &d in &frac_digits {
            out.push(pack(d, conf));
        }
    }

    out.push(pack(SYM_END, conf));
    Ok(out)
}

/// Decoda Path B → f64.
///
/// # Errors
/// `Empty`, `InvalidByte`, `UnexpectedEnd` para streams malformados.
pub fn decode(bytes: &[u8]) -> Result<f64, DecodeError> {
    if bytes.is_empty() {
        return Err(DecodeError::Empty);
    }

    let mut value = 0.0_f64;
    let mut sign = 1.0_f64;
    let mut in_fractional = false;
    let mut frac_pos = -1_i32;
    let mut found_end = false;

    for &b in bytes {
        let (sym, _conf) = unpack(b);
        match sym {
            SYM_END => {
                found_end = true;
                break;
            }
            SYM_NEG => sign = -1.0,
            SYM_DOT => in_fractional = true,
            SYM_E => return Err(DecodeError::InvalidByte(b)),
            d if d <= 9 => {
                if in_fractional {
                    value += f64::from(d) * (10f64).powi(frac_pos);
                    frac_pos -= 1;
                } else {
                    value = value * 10.0 + f64::from(d);
                }
            }
            _ => return Err(DecodeError::InvalidByte(b)),
        }
    }

    if !found_end {
        return Err(DecodeError::UnexpectedEnd);
    }

    Ok(sign * value)
}

#[must_use]
pub fn encoded_size(x: f64) -> Option<usize> {
    encode(x).ok().map(|v| v.len())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn within(expected: f64, got: f64) -> bool {
        if expected == 0.0 {
            return got.abs() < 1e-12;
        }
        let abs = (expected - got).abs();
        if abs < 1e-12 {
            return true;
        }
        abs / expected.abs() < 1e-9
    }

    #[test]
    fn roundtrip_specific() {
        for &x in &[0.0, 1.0, -1.0, 47.83, -3.14, 0.4847, 123.456, 1e6, 1e-3, -1e-5] {
            let bytes = encode(x).unwrap();
            let y = decode(&bytes).unwrap();
            assert!(within(x, y), "x={x} y={y}");
        }
    }

    #[test]
    fn confidence_preserved() {
        let bytes = encode_with_confidence(0.5, 7).unwrap();
        for &b in &bytes {
            let (_d, c) = unpack(b);
            assert_eq!(c, 7);
        }
    }

    #[test]
    fn roundtrip_random() {
        let mut rng = crate::Rng::new(42);
        for _ in 0..1000 {
            let x = rng.range_f64(-1e6, 1e6);
            let bytes = encode(x).unwrap();
            let y = decode(&bytes).unwrap();
            assert!(within(x, y), "x={x} y={y}");
        }
    }
}
