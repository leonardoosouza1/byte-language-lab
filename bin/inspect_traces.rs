//! Lê data/qwen_traces.bin e imprime os primeiros N records pra inspeção.
use std::io::Read;

fn main() -> std::io::Result<()> {
    let mut f = std::fs::File::open("data/qwen_traces.bin")?;
    let mut buf = Vec::new();
    f.read_to_end(&mut buf)?;
    assert_eq!(&buf[0..4], b"QTRC");
    let _ver = u32::from_le_bytes(buf[4..8].try_into().unwrap());
    let n = u32::from_le_bytes(buf[8..12].try_into().unwrap()) as usize;
    println!("n_records = {n}");
    let mut c = 12usize;
    let mut total_resp_len = 0usize;
    let mut max_resp = 0usize;
    let mut len_hist = [0usize; 8]; // 0..16, 16..32, 32..64, 64..128, 128..256, 256..512, 512..1024, 1024+
    for i in 0..n {
        let pl = u32::from_le_bytes(buf[c..c + 4].try_into().unwrap()) as usize;
        c += 4;
        let p = std::str::from_utf8(&buf[c..c + pl]).unwrap_or("?");
        c += pl;
        let rl = u32::from_le_bytes(buf[c..c + 4].try_into().unwrap()) as usize;
        c += 4;
        let r = std::str::from_utf8(&buf[c..c + rl]).unwrap_or("?");
        c += rl;
        total_resp_len += rl;
        max_resp = max_resp.max(rl);
        let bucket = match rl {
            0..=15 => 0, 16..=31 => 1, 32..=63 => 2, 64..=127 => 3,
            128..=255 => 4, 256..=511 => 5, 512..=1023 => 6, _ => 7,
        };
        len_hist[bucket] += 1;
        if i < 8 || i >= n - 4 {
            let r_disp: String = r.chars().take(120).collect();
            println!("[{i:>3}] ({rl:>4}b) {p:?}\n     → {r_disp:?}");
        }
    }
    println!("\navg resp len = {:.1}, max = {}", total_resp_len as f64 / n as f64, max_resp);
    println!("len buckets: <16:{} 16-31:{} 32-63:{} 64-127:{} 128-255:{} 256-511:{} 512-1023:{} 1024+:{}",
        len_hist[0], len_hist[1], len_hist[2], len_hist[3],
        len_hist[4], len_hist[5], len_hist[6], len_hist[7]);
    Ok(())
}
