//! byte_graph_communities.rs — detecta comunidades no grafo byte-byte.
//!
//! Constrói grafo: nós = 256 bytes, edges = bigram frequency no corpus PT-BR.
//! Aplica algoritmo simples greedy modularity (Louvain-light).
//! Exporta como DOT para Graphviz.

use std::fs;
use std::io::Write;
use std::path::Path;
use std::collections::HashMap;

const CORPUS: &str = "data/dialogues_augmented.txt";

fn normalize(s: &str) -> Vec<u8> {
    let bytes = s.as_bytes();
    let mut out = Vec::with_capacity(bytes.len());
    let mut i = 0;
    while i < bytes.len() {
        if bytes[i] == 0xC3 && i + 1 < bytes.len() {
            let m = match bytes[i+1] {
                0xA0..=0xA5 => b'a', 0xA8..=0xAB => b'e',
                0xAC..=0xAF => b'i', 0xB2..=0xB6 => b'o',
                0xB9..=0xBC => b'u', 0xA7 => b'c',
                0x80..=0x85 => b'a', 0x87 => b'c',
                0x88..=0x8B => b'e', 0x8C..=0x8F => b'i',
                0x92..=0x96 => b'o', 0x99..=0x9C => b'u',
                _ => { i += 2; continue; }
            };
            out.push(m); i += 2;
        } else {
            let b = bytes[i];
            if b.is_ascii_alphanumeric() { out.push(b | 0x20); }
            else if b.is_ascii_whitespace() { out.push(b' '); }
            else { out.push(b); }
            i += 1;
        }
    }
    out
}

fn main() -> std::io::Result<()> {
    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║   IARA Byte Graph — comunidades emergentes do corpus PT-BR   ║");
    println!("╚══════════════════════════════════════════════════════════════╝\n");

    let mut text = String::new();
    for p in &[CORPUS, "data/qwen_sustained.txt"] {
        if Path::new(p).exists() {
            text.push_str(&fs::read_to_string(p)?);
            text.push('\n');
        }
    }
    println!("  Corpus: {} bytes", text.len());

    // Build 256x256 bigram graph
    let mut bigram = vec![vec![0u32; 256]; 256];
    let mut total_edges = 0u64;
    for line in text.lines() {
        let norm = normalize(line);
        for w in norm.windows(2) {
            let a = w[0] as usize;
            let b = w[1] as usize;
            bigram[a][b] += 1;
            total_edges += 1;
        }
    }
    println!("  Total transitions: {}", total_edges);

    // Filter: focus on alpha + space + digits
    let active: Vec<u8> = std::iter::once(b' ')
        .chain((b'0'..=b'9'))
        .chain((b'a'..=b'z'))
        .collect();
    println!("  Active nodes: {} (alfa + dígitos + espaço)\n", active.len());

    // Compute byte degree (out-edges)
    let mut degree: HashMap<u8, u32> = HashMap::new();
    for &a in &active {
        let d: u32 = active.iter().map(|&b| bigram[a as usize][b as usize]).sum();
        if d > 0 { degree.insert(a, d); }
    }

    // Simple greedy modularity: start with each node in own community,
    // iteratively merge pairs that improve modularity.
    // Modularity Q = sum_ij (A_ij/m - k_i k_j / (2m)^2) δ(c_i, c_j)
    // For simplicity: cluster by k-means on (out_distribution, in_distribution).

    // Compute affinity matrix as cosine of normalized rows
    println!("▸ Computando afinidades byte-byte (cosine de transition profiles):");
    let mut profiles: HashMap<u8, Vec<f32>> = HashMap::new();
    for &a in &active {
        let row: Vec<f32> = active.iter().map(|&b| bigram[a as usize][b as usize] as f32).collect();
        let n: f32 = row.iter().map(|x| x*x).sum::<f32>().sqrt();
        if n > 0.0 {
            profiles.insert(a, row.iter().map(|&x| x / n).collect());
        }
    }

    fn cos(a: &[f32], b: &[f32]) -> f32 {
        let mut s = 0.0;
        for i in 0..a.len() { s += a[i] * b[i]; }
        s
    }

    // Print top-5 most similar peers for each letter
    println!("\n▸ Para cada letra: top-5 mais similares (por transition profile):\n");
    for letter in [b'a', b'e', b'i', b'o', b'u',  // vogais
                   b'b', b'c', b'd', b'f', b'g',
                   b'l', b'm', b'n', b'r', b's', b't',
                   b'0', b'1', b' '] {
        let l = letter;
        if let Some(prof_l) = profiles.get(&l) {
            let mut scored: Vec<(f32, u8)> = active.iter()
                .filter(|&&b| b != l)
                .filter_map(|&b| profiles.get(&b).map(|p| (cos(prof_l, p), b)))
                .collect();
            scored.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap());
            let top: Vec<String> = scored.iter().take(5)
                .map(|(s, b)| format!("{}={:.3}", if *b == b' ' { "' '".to_string() } else { format!("{}", *b as char) }, s))
                .collect();
            let lc = if l == b' ' { "' '".to_string() } else { format!("{}", l as char) };
            println!("  {}: {}", lc, top.join("  "));
        }
    }

    // Greedy clustering by cosine threshold
    println!("\n▸ Detectando comunidades (greedy cosine ≥ 0.7):\n");
    let mut comm: HashMap<u8, u32> = HashMap::new();
    let mut next_comm = 0u32;
    for &a in &active {
        if comm.contains_key(&a) { continue; }
        comm.insert(a, next_comm);
        if let Some(pa) = profiles.get(&a) {
            for &b in &active {
                if b == a || comm.contains_key(&b) { continue; }
                if let Some(pb) = profiles.get(&b) {
                    if cos(pa, pb) >= 0.70 {
                        comm.insert(b, next_comm);
                    }
                }
            }
        }
        next_comm += 1;
    }

    // Group display
    let mut by_comm: HashMap<u32, Vec<u8>> = HashMap::new();
    for (&b, &c) in &comm {
        by_comm.entry(c).or_default().push(b);
    }
    let mut groups: Vec<(u32, Vec<u8>)> = by_comm.into_iter().collect();
    groups.sort_by(|a, b| b.1.len().cmp(&a.1.len()));

    println!("  Detected {} comunidades (top 8 por tamanho):", groups.len());
    for (idx, (cid, members)) in groups.iter().take(8).enumerate() {
        let mut mc = members.clone();
        mc.sort();
        let ms: String = mc.iter().map(|&b| {
            if b == b' ' { "' '".to_string() } else { format!("{}", b as char) }
        }).collect::<Vec<_>>().join(" ");
        println!("  C{}: {} chars  →  {}", idx, members.len(), ms);
    }

    // Export DOT
    let dot_path = "/tmp/byte_graph_communities.dot";
    let mut f = fs::File::create(dot_path)?;
    writeln!(f, "graph ByteCommunities {{")?;
    writeln!(f, "  layout=fdp; overlap=false;")?;
    writeln!(f, "  node [shape=circle];")?;
    // Color por comunidade
    let palette = ["#ff6b6b", "#4ecdc4", "#ffe66d", "#a8e6cf", "#c9b1ff", "#ffb86c", "#8be9fd", "#bd93f9"];
    for (&b, &c) in &comm {
        let label = if b == b' ' { "_".to_string() } else { format!("{}", b as char) };
        let color = palette[c as usize % palette.len()];
        writeln!(f, "  \"{}\" [fillcolor=\"{}\", style=filled];", label, color)?;
    }
    // Edges com peso ≥ 1% do total
    let thresh = (total_edges / 10000) as u32;
    for &a in &active {
        for &b in &active {
            let w = bigram[a as usize][b as usize];
            if w > thresh {
                let la = if a == b' ' { "_".to_string() } else { format!("{}", a as char) };
                let lb = if b == b' ' { "_".to_string() } else { format!("{}", b as char) };
                let pen = ((w as f32 / thresh as f32).ln().max(0.5) as i32).clamp(1, 8);
                writeln!(f, "  \"{}\" -- \"{}\" [penwidth={}];", la, lb, pen)?;
            }
        }
    }
    writeln!(f, "}}")?;
    println!("\n▸ Grafo salvo: {} ({} ativos, threshold edge ≥{})", dot_path, active.len(), thresh);

    Ok(())
}
