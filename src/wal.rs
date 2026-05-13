//! WAL — binary append-only Write-Ahead Log for auto-learning.
//!
//! File layout:
//!   Header (12 bytes): magic "IWAL" | version u32 | reserved u32
//!   Record (variable, 22 bytes overhead):
//!     type u8 (1=dialog, 2=curiosity)
//!     flags u8 (bit0=from_qwen, bit1=from_curiosity, bit2=high_quality)
//!     ts u64 (unix seconds)
//!     score f32 (quality 0..1)
//!     user_len u32, model_len u32
//!     user_bytes, model_bytes
//!
//! Multiple writers must hold the Mutex<File>. Reads are atomic snapshot.

use std::fs::{File, OpenOptions};
use std::io::{Read, Write};
use std::path::Path;

pub const MAGIC: &[u8; 4] = b"IWAL";
pub const VERSION: u32 = 1;
pub const HEADER_SIZE: u64 = 12;

pub const TYPE_DIALOG: u8 = 1;
pub const TYPE_CURIOSITY: u8 = 2;
pub const FLAG_FROM_QWEN: u8 = 1 << 0;
pub const FLAG_FROM_CURIOSITY: u8 = 1 << 1;
pub const FLAG_HIGH_QUALITY: u8 = 1 << 2;

pub struct Record {
    pub rtype: u8,
    pub flags: u8,
    pub ts: u64,
    pub score: f32,
    pub user: Vec<u8>,
    pub model: Vec<u8>,
}

pub fn open_for_append(path: &str) -> std::io::Result<File> {
    let exists = Path::new(path).exists();
    let mut f = OpenOptions::new().create(true).append(true).read(true).open(path)?;
    if !exists {
        let mut hdr = Vec::with_capacity(12);
        hdr.extend_from_slice(MAGIC);
        hdr.extend_from_slice(&VERSION.to_le_bytes());
        hdr.extend_from_slice(&0u32.to_le_bytes());
        f.write_all(&hdr)?;
    }
    Ok(f)
}

pub fn append(file: &mut File, rec: &Record) -> std::io::Result<()> {
    let mut buf = Vec::with_capacity(22 + rec.user.len() + rec.model.len());
    buf.push(rec.rtype);
    buf.push(rec.flags);
    buf.extend_from_slice(&rec.ts.to_le_bytes());
    buf.extend_from_slice(&rec.score.to_le_bytes());
    buf.extend_from_slice(&(rec.user.len() as u32).to_le_bytes());
    buf.extend_from_slice(&(rec.model.len() as u32).to_le_bytes());
    buf.extend_from_slice(&rec.user);
    buf.extend_from_slice(&rec.model);
    file.write_all(&buf)
}

pub fn read_all(path: &str) -> std::io::Result<Vec<Record>> {
    if !Path::new(path).exists() { return Ok(Vec::new()); }
    let mut f = File::open(path)?;
    let mut data = Vec::new();
    f.read_to_end(&mut data)?;
    if data.len() < HEADER_SIZE as usize { return Ok(Vec::new()); }
    if &data[0..4] != MAGIC { return Err(std::io::Error::other("bad magic")); }
    let mut out = Vec::new();
    let mut i = HEADER_SIZE as usize;
    while i + 22 <= data.len() {
        let rtype = data[i];
        let flags = data[i+1];
        let ts = u64::from_le_bytes(data[i+2..i+10].try_into().unwrap());
        let score = f32::from_le_bytes(data[i+10..i+14].try_into().unwrap());
        let user_len = u32::from_le_bytes(data[i+14..i+18].try_into().unwrap()) as usize;
        let model_len = u32::from_le_bytes(data[i+18..i+22].try_into().unwrap()) as usize;
        let body_start = i + 22;
        let body_end = body_start + user_len + model_len;
        if body_end > data.len() { break; }
        let user = data[body_start..body_start+user_len].to_vec();
        let model = data[body_start+user_len..body_end].to_vec();
        out.push(Record { rtype, flags, ts, score, user, model });
        i = body_end;
    }
    Ok(out)
}

pub fn count_records(path: &str) -> usize {
    read_all(path).map(|v| v.len()).unwrap_or(0)
}

pub fn size_bytes(path: &str) -> u64 {
    std::fs::metadata(path).map(|m| m.len()).unwrap_or(0)
}

pub fn archive(path: &str) -> std::io::Result<String> {
    let ts = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs()).unwrap_or(0);
    let arch = format!("{}.{}.archive", path, ts);
    std::fs::rename(path, &arch)?;
    Ok(arch)
}

pub fn now_ts() -> u64 {
    std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs()).unwrap_or(0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn header_format() {
        assert_eq!(MAGIC, b"IWAL");
        assert_eq!(VERSION, 1);
        assert_eq!(HEADER_SIZE, 12);
    }

    #[test]
    fn roundtrip_record() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("test.wal");
        let path_s = path.to_str().unwrap();

        let mut f = open_for_append(path_s).unwrap();
        append(&mut f, &Record {
            rtype: TYPE_DIALOG,
            flags: FLAG_FROM_QWEN,
            ts: 1234567,
            score: 0.95,
            user: b"hello".to_vec(),
            model: b"world".to_vec(),
        }).unwrap();
        drop(f);

        let recs = read_all(path_s).unwrap();
        assert_eq!(recs.len(), 1);
        assert_eq!(recs[0].rtype, TYPE_DIALOG);
        assert_eq!(recs[0].user, b"hello");
        assert_eq!(recs[0].model, b"world");
    }
}
