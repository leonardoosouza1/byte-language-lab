//! # Path C — Domain-tagged (header + payload)
//!
//! Cada bloco semântico começa com um byte HEADER `[domain:4][length:4]`,
//! seguido pelo payload (bytes interpretados conforme domínio).
//!
//! Suporta multi-domínio em um único stream — a chave para o "byte language"
//! unificado: número, texto, comando, estado FSM, boolean, percent coexistem.

use super::{DecodeError, EncodeError, path_a};

#[repr(u8)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Domain {
    Number = 0,
    Hex = 1,
    Text = 2,
    Command = 3,
    State = 4,
    Boolean = 5,
    Percent = 6,
}

impl Domain {
    #[must_use]
    pub const fn from_u8(v: u8) -> Option<Self> {
        match v {
            0 => Some(Self::Number),
            1 => Some(Self::Hex),
            2 => Some(Self::Text),
            3 => Some(Self::Command),
            4 => Some(Self::State),
            5 => Some(Self::Boolean),
            6 => Some(Self::Percent),
            _ => None,
        }
    }
}

pub const EOB: u8 = 0xFF;

#[derive(Debug, Clone, PartialEq)]
pub enum Block {
    Number(f64),
    Text(String),
    Command { action: u8, magnitude: u8 },
    State(u8),
    Boolean(bool),
    Percent(u8),
}

#[inline]
#[must_use]
pub const fn pack_header(domain: u8, length: u8) -> u8 {
    debug_assert!(domain < 16);
    debug_assert!(length < 16);
    (domain << 4) | (length & 0x0F)
}

#[inline]
#[must_use]
pub const fn unpack_header(byte: u8) -> (u8, u8) {
    (byte >> 4, byte & 0x0F)
}

/// Encoda f64 como bloco Number usando Path A no payload.
///
/// # Errors
/// Propaga erro do Path A (NotFinite, OutOfRange).
pub fn encode_number(x: f64) -> Result<Vec<u8>, EncodeError> {
    let payload = path_a::encode(x)?;
    if payload.len() > 15 {
        let mut out = Vec::with_capacity(payload.len() + 2);
        out.push(pack_header(Domain::Number as u8, 0));
        out.extend_from_slice(&payload);
        out.push(EOB);
        Ok(out)
    } else {
        let mut out = Vec::with_capacity(payload.len() + 1);
        out.push(pack_header(Domain::Number as u8, payload.len() as u8));
        out.extend_from_slice(&payload);
        Ok(out)
    }
}

/// Encoda string como bloco Text.
///
/// # Errors
/// Não há erro intrínseco; assinatura uniforme com `encode_number`.
pub fn encode_text(s: &str) -> Result<Vec<u8>, EncodeError> {
    let bytes = s.as_bytes();
    if bytes.len() > 15 {
        let mut out = Vec::with_capacity(bytes.len() + 2);
        out.push(pack_header(Domain::Text as u8, 0));
        out.extend_from_slice(bytes);
        out.push(EOB);
        Ok(out)
    } else {
        let mut out = Vec::with_capacity(bytes.len() + 1);
        out.push(pack_header(Domain::Text as u8, bytes.len() as u8));
        out.extend_from_slice(bytes);
        Ok(out)
    }
}

#[must_use]
pub fn encode_command(action: u8, magnitude: u8) -> Vec<u8> {
    vec![pack_header(Domain::Command as u8, 2), action, magnitude]
}

#[must_use]
pub fn encode_state(fsm_byte: u8) -> Vec<u8> {
    vec![pack_header(Domain::State as u8, 1), fsm_byte]
}

#[must_use]
pub fn encode_boolean(b: bool) -> Vec<u8> {
    vec![pack_header(Domain::Boolean as u8, 1), u8::from(b)]
}

#[must_use]
pub fn encode_percent(p: u8) -> Vec<u8> {
    vec![pack_header(Domain::Percent as u8, 1), p.min(100)]
}

/// Decoda múltiplos blocos consecutivos no stream.
///
/// # Errors
/// `InvalidByte` se domain desconhecido. `MalformedSequence` se payload mal formatado.
/// `UnexpectedEnd` se EOB esperado não for encontrado.
pub fn decode_stream(bytes: &[u8]) -> Result<Vec<Block>, DecodeError> {
    let mut blocks = Vec::new();
    let mut i = 0;

    while i < bytes.len() {
        let header = bytes[i];
        i += 1;

        let (domain_raw, length) = unpack_header(header);
        let dom = Domain::from_u8(domain_raw).ok_or(DecodeError::InvalidByte(header))?;

        let (payload_end, skip_eob) = if length == 0 {
            let mut e = i;
            while e < bytes.len() && bytes[e] != EOB {
                e += 1;
            }
            if e >= bytes.len() {
                return Err(DecodeError::UnexpectedEnd);
            }
            (e, true)
        } else {
            (i + length as usize, false)
        };

        if payload_end > bytes.len() {
            return Err(DecodeError::UnexpectedEnd);
        }

        let payload = &bytes[i..payload_end];
        let block = match dom {
            Domain::Number => Block::Number(path_a::decode(payload)?),
            Domain::Text => Block::Text(
                String::from_utf8(payload.to_vec()).map_err(|_| DecodeError::InvalidByte(0))?,
            ),
            Domain::Command if payload.len() == 2 => Block::Command {
                action: payload[0],
                magnitude: payload[1],
            },
            Domain::State if payload.len() == 1 => Block::State(payload[0]),
            Domain::Boolean if payload.len() == 1 => Block::Boolean(payload[0] != 0),
            Domain::Percent if payload.len() == 1 => Block::Percent(payload[0]),
            _ => return Err(DecodeError::MalformedSequence),
        };
        blocks.push(block);

        i = payload_end + (if skip_eob { 1 } else { 0 });
    }

    Ok(blocks)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn number_roundtrip() {
        let bytes = encode_number(47.83).unwrap();
        let blocks = decode_stream(&bytes).unwrap();
        assert_eq!(blocks.len(), 1);
        match blocks[0] {
            Block::Number(n) => assert!((n - 47.83).abs() < 1e-7),
            _ => panic!("expected Number"),
        }
    }

    #[test]
    fn multi_domain_stream() {
        let mut stream = Vec::new();
        stream.extend(encode_number(42.0).unwrap());
        stream.extend(encode_text("hello").unwrap());
        stream.extend(encode_command(3, 100));
        stream.extend(encode_state(0x9D));
        stream.extend(encode_boolean(true));
        stream.extend(encode_percent(75));

        let blocks = decode_stream(&stream).unwrap();
        assert_eq!(blocks.len(), 6);
        assert!(matches!(blocks[0], Block::Number(n) if (n - 42.0).abs() < 1e-9));
        assert!(matches!(&blocks[1], Block::Text(s) if s == "hello"));
        assert!(matches!(blocks[2], Block::Command { action: 3, magnitude: 100 }));
        assert!(matches!(blocks[3], Block::State(0x9D)));
        assert!(matches!(blocks[4], Block::Boolean(true)));
        assert!(matches!(blocks[5], Block::Percent(75)));
    }

    #[test]
    fn long_text_uses_eob() {
        let s = "this is a longer text that exceeds 15 bytes";
        let bytes = encode_text(s).unwrap();
        assert_eq!(unpack_header(bytes[0]).1, 0);
        let blocks = decode_stream(&bytes).unwrap();
        assert!(matches!(&blocks[0], Block::Text(t) if t == s));
    }
}
