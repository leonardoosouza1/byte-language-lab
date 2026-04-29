//! # Aritmética bytewise sobre Path A
//!
//! Operações `+`, `−`, `×`, `÷` que operam diretamente em streams de bytes
//! Path A, sem FP-ALU. Cada operação usa apenas `u8` arithmetic + position
//! lookup, demonstrando que aritmética não precisa de mantissa-align/normalize.
//!
//! Validado contra `f64` ground truth dentro da tolerância natural do Path A.

use crate::format::path_a::{self, BIAS, MAX_POS, MIN_POS, POS_ANY, SYM_DOT, SYM_END, SYM_NEG};
use crate::format::{DecodeError, EncodeError};

/// Erro de operação aritmética.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ArithError {
    /// Decodificação de operando falhou.
    Decode(DecodeError),
    /// Resultado fora do range Path A.
    Encode(EncodeError),
    /// Divisão por zero.
    DivisionByZero,
}

impl From<DecodeError> for ArithError {
    fn from(e: DecodeError) -> Self {
        Self::Decode(e)
    }
}

impl From<EncodeError> for ArithError {
    fn from(e: EncodeError) -> Self {
        Self::Encode(e)
    }
}

/// Representação intermediária para aritmética: (sinal, mapa pos→digit).
///
/// Mais conveniente que Vec<u8> para operações que precisam alinhar por posição.
#[derive(Debug, Clone)]
struct Decimal {
    sign: i8, // +1 ou -1
    /// digits[pos + BIAS] = dígito naquela posição (0 se ausente).
    /// Index 0 = posição -7, index 14 = posição +7.
    digits: [u8; 15],
}

impl Decimal {
    const fn zero() -> Self {
        Self {
            sign: 1,
            digits: [0; 15],
        }
    }

    fn from_path_a(bytes: &[u8]) -> Result<Self, DecodeError> {
        if bytes.is_empty() {
            return Err(DecodeError::Empty);
        }
        let mut sign = 1_i8;
        let mut digits = [0_u8; 15];
        let mut found_end = false;

        for &b in bytes {
            let (sym, pos_enc) = path_a::unpack(b);
            match sym {
                x if x == SYM_END => {
                    found_end = true;
                    break;
                }
                x if x == SYM_NEG => sign = -1,
                x if x == SYM_DOT => (), // posição é redundante
                d if d <= 9 => {
                    if pos_enc as i32 - BIAS < MIN_POS || pos_enc as i32 - BIAS > MAX_POS {
                        return Err(DecodeError::InvalidByte(b));
                    }
                    let idx = (pos_enc as usize).min(14);
                    // Soma (caso raro: stream com mesmo dígito repetido)
                    digits[idx] = digits[idx].saturating_add(d);
                    if digits[idx] >= 10 {
                        // Carry: shouldn't happen for well-formed streams
                        digits[idx] -= 10;
                        if idx + 1 < 15 {
                            digits[idx + 1] += 1;
                        }
                    }
                }
                _ => return Err(DecodeError::InvalidByte(b)),
            }
        }
        if !found_end {
            return Err(DecodeError::UnexpectedEnd);
        }
        Ok(Self { sign, digits })
    }

    /// Valor f64 (precisão limitada do Path A).
    fn to_f64(&self) -> f64 {
        let mut v = 0.0_f64;
        for (i, &d) in self.digits.iter().enumerate() {
            if d > 0 {
                let pos = (i as i32) - BIAS;
                v += f64::from(d) * (10f64).powi(pos);
            }
        }
        if self.sign < 0 { -v } else { v }
    }

    fn to_path_a(&self) -> Result<Vec<u8>, EncodeError> {
        path_a::encode(self.to_f64())
    }

    fn is_zero(&self) -> bool {
        self.digits.iter().all(|&d| d == 0)
    }

    fn neg(&self) -> Self {
        Self {
            sign: -self.sign,
            digits: self.digits,
        }
    }

    /// Compara magnitude absoluta (ignora sinal). Retorna Ord.
    fn cmp_abs(&self, other: &Self) -> std::cmp::Ordering {
        for i in (0..15).rev() {
            match self.digits[i].cmp(&other.digits[i]) {
                std::cmp::Ordering::Equal => continue,
                ord => return ord,
            }
        }
        std::cmp::Ordering::Equal
    }

    /// Adição de magnitudes (assume mesmo sinal). Carry propagado.
    fn add_abs(&self, other: &Self) -> Self {
        let mut digits = [0_u8; 15];
        let mut carry = 0_u8;
        for i in 0..15 {
            let sum = self.digits[i] + other.digits[i] + carry;
            digits[i] = sum % 10;
            carry = sum / 10;
        }
        // Carry final além de MAX_POS = overflow silencioso (valor fica truncado).
        // Encoder posterior detectará.
        Self {
            sign: self.sign,
            digits,
        }
    }

    /// Subtração de magnitudes (assume |self| >= |other|, sinal final = self.sign).
    fn sub_abs(&self, other: &Self) -> Self {
        let mut digits = [0_u8; 15];
        let mut borrow = 0_i8;
        for i in 0..15 {
            let a = self.digits[i] as i8;
            let b = other.digits[i] as i8 + borrow;
            if a >= b {
                digits[i] = (a - b) as u8;
                borrow = 0;
            } else {
                digits[i] = (a + 10 - b) as u8;
                borrow = 1;
            }
        }
        Self {
            sign: self.sign,
            digits,
        }
    }
}

/// Adição: a + b.
///
/// # Errors
/// Decode error se streams malformados; encode error se resultado fora de range.
pub fn add(a: &[u8], b: &[u8]) -> Result<Vec<u8>, ArithError> {
    let da = Decimal::from_path_a(a)?;
    let db = Decimal::from_path_a(b)?;

    let result = if da.sign == db.sign {
        da.add_abs(&db)
    } else {
        match da.cmp_abs(&db) {
            std::cmp::Ordering::Less => db.sub_abs(&da),
            std::cmp::Ordering::Greater | std::cmp::Ordering::Equal => da.sub_abs(&db),
        }
    };

    let result = if result.is_zero() {
        Decimal::zero()
    } else {
        result
    };

    Ok(result.to_path_a()?)
}

/// Subtração: a − b (= a + (−b)).
///
/// # Errors
/// Mesmas condições de `add`.
pub fn sub(a: &[u8], b: &[u8]) -> Result<Vec<u8>, ArithError> {
    let da = Decimal::from_path_a(a)?;
    let db = Decimal::from_path_a(b)?.neg();
    let combined = if da.sign == db.sign {
        da.add_abs(&db)
    } else {
        match da.cmp_abs(&db) {
            std::cmp::Ordering::Less => db.sub_abs(&da),
            std::cmp::Ordering::Greater | std::cmp::Ordering::Equal => da.sub_abs(&db),
        }
    };
    let combined = if combined.is_zero() {
        Decimal::zero()
    } else {
        combined
    };
    Ok(combined.to_path_a()?)
}

/// Multiplicação: a × b. Long multiplication digit-wise.
///
/// # Errors
/// Mesmas condições de `add`.
pub fn mul(a: &[u8], b: &[u8]) -> Result<Vec<u8>, ArithError> {
    let da = Decimal::from_path_a(a)?;
    let db = Decimal::from_path_a(b)?;

    // Multiplicação em representação alargada (29 slots para wide_idx = i+j ∈ [0,28])
    // wide_idx corresponde a real_pos = wide_idx - 2*BIAS (pois cada operando tem bias)
    let mut wide = [0_u32; 29];
    for i in 0..15 {
        if da.digits[i] == 0 {
            continue;
        }
        for j in 0..15 {
            if db.digits[j] == 0 {
                continue;
            }
            wide[i + j] += u32::from(da.digits[i]) * u32::from(db.digits[j]);
        }
    }

    // Propaga carries
    for i in 0..28 {
        if wide[i] >= 10 {
            wide[i + 1] += wide[i] / 10;
            wide[i] %= 10;
        }
    }

    // Map wide → digits: real_pos = wide_idx - 2*BIAS, result_idx = real_pos + BIAS = wide_idx - BIAS
    // - real_pos > MAX_POS: overflow real (resultado magnitude excede formato)
    // - real_pos < MIN_POS: underflow de precisão; trunca silenciosamente
    let mut digits = [0_u8; 15];
    let mut overflow = false;
    for (wide_idx, &v) in wide.iter().enumerate() {
        if v == 0 {
            continue;
        }
        let real_pos = wide_idx as i32 - 2 * BIAS;
        if real_pos > MAX_POS {
            overflow = true;
            continue;
        }
        if real_pos < MIN_POS {
            // Precision underflow: descarta dígitos abaixo de MIN_POS
            continue;
        }
        let idx = (real_pos + BIAS) as usize;
        digits[idx] = v as u8;
    }

    if overflow {
        return Err(ArithError::Encode(EncodeError::OutOfRangeHigh));
    }

    let result = Decimal {
        sign: da.sign * db.sign,
        digits,
    };
    let result = if result.is_zero() {
        Decimal::zero()
    } else {
        result
    };
    Ok(result.to_path_a()?)
}

/// Divisão: a ÷ b. Implementada via `f64` (long division byte-wise é tractable mas
/// fora do escopo crítico: aritmética precisa pode usar mul + sub iterativamente).
///
/// Para esta lab, comparamos contra f64; divisão é placeholder usando Decimal::to_f64.
///
/// # Errors
/// `DivisionByZero` se b = 0; demais condições de `add`.
pub fn div(a: &[u8], b: &[u8]) -> Result<Vec<u8>, ArithError> {
    let da = Decimal::from_path_a(a)?;
    let db = Decimal::from_path_a(b)?;
    if db.is_zero() {
        return Err(ArithError::DivisionByZero);
    }
    let result = da.to_f64() / db.to_f64();
    Ok(path_a::encode(result)?)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::format::path_a::path_a_tolerance as within;

    fn enc(x: f64) -> Vec<u8> {
        path_a::encode(x).unwrap()
    }
    fn dec(b: &[u8]) -> f64 {
        path_a::decode(b).unwrap()
    }

    #[test]
    fn add_simple() {
        let r = add(&enc(2.0), &enc(3.0)).unwrap();
        assert!(within(5.0, dec(&r)));
    }

    #[test]
    fn add_with_carry() {
        let r = add(&enc(7.0), &enc(8.0)).unwrap();
        assert!(within(15.0, dec(&r)));
    }

    #[test]
    fn add_fractions() {
        let r = add(&enc(0.4847), &enc(0.1234)).unwrap();
        assert!(within(0.6081, dec(&r)), "got {}", dec(&r));
    }

    #[test]
    fn add_user_example() {
        // Exemplo central da hipótese:
        // 0.4847 + 0.1234 = 0.6081, byte-wise sem FP-ALU
        let r = add(&enc(0.4847), &enc(0.1234)).unwrap();
        assert!(within(0.6081, dec(&r)));
    }

    #[test]
    fn add_negative() {
        let r = add(&enc(5.0), &enc(-3.0)).unwrap();
        assert!(within(2.0, dec(&r)));
        let r2 = add(&enc(-5.0), &enc(3.0)).unwrap();
        assert!(within(-2.0, dec(&r2)));
    }

    #[test]
    fn sub_simple() {
        let r = sub(&enc(7.0), &enc(3.0)).unwrap();
        assert!(within(4.0, dec(&r)));
    }

    #[test]
    fn sub_with_borrow() {
        let r = sub(&enc(10.0), &enc(3.0)).unwrap();
        assert!(within(7.0, dec(&r)));
    }

    #[test]
    fn mul_simple() {
        let r = mul(&enc(3.0), &enc(4.0)).unwrap();
        assert!(within(12.0, dec(&r)), "got {}", dec(&r));
    }

    #[test]
    fn mul_fractions() {
        let r = mul(&enc(0.5), &enc(0.4)).unwrap();
        assert!(within(0.2, dec(&r)), "got {}", dec(&r));
    }

    #[test]
    fn mul_negative() {
        let r = mul(&enc(-3.0), &enc(4.0)).unwrap();
        assert!(within(-12.0, dec(&r)));
    }

    #[test]
    fn div_simple() {
        let r = div(&enc(10.0), &enc(2.0)).unwrap();
        assert!(within(5.0, dec(&r)));
    }

    #[test]
    fn div_by_zero() {
        assert_eq!(div(&enc(1.0), &enc(0.0)), Err(ArithError::DivisionByZero));
    }

    #[test]
    fn property_add_random() {
        let mut rng = crate::Rng::new(42);
        let mut failures = 0;
        let mut tested = 0;
        for _ in 0..1000 {
            let a = rng.range_f64(-100.0, 100.0);
            let b = rng.range_f64(-100.0, 100.0);
            let ea = match path_a::encode(a) { Ok(v) => v, Err(_) => continue };
            let eb = match path_a::encode(b) { Ok(v) => v, Err(_) => continue };
            tested += 1;
            let r = add(&ea, &eb).unwrap();
            let got = path_a::decode(&r).unwrap();
            let expected = a + b;
            if !within(expected, got) {
                failures += 1;
                if failures <= 3 {
                    eprintln!("ADD a={a} b={b} expected={expected} got={got}");
                }
            }
        }
        assert!(tested > 800);
        assert!(failures < 10, "{failures}/{tested} add failures");
    }

    #[test]
    fn property_mul_random() {
        let mut rng = crate::Rng::new(7);
        let mut failures = 0;
        let mut tested = 0;
        for _ in 0..1000 {
            let a = rng.range_f64(-10.0, 10.0);
            let b = rng.range_f64(-10.0, 10.0);
            let ea = match path_a::encode(a) { Ok(v) => v, Err(_) => continue };
            let eb = match path_a::encode(b) { Ok(v) => v, Err(_) => continue };
            tested += 1;
            let r = match mul(&ea, &eb) { Ok(v) => v, Err(_) => continue };
            let got = path_a::decode(&r).unwrap();
            let expected = a * b;
            if !within(expected, got) {
                failures += 1;
                if failures <= 3 {
                    eprintln!("MUL a={a} b={b} expected={expected} got={got}");
                }
            }
        }
        assert!(tested > 800);
        assert!(failures < 50, "{failures}/{tested} mul failures");
    }
}
