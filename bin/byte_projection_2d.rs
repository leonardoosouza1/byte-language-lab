//! byte_projection_2d.rs — projeta bytes em 2D via PCA do bigram profile.
//!
//! Cada byte tem vetor 37-D (transition profile pra outros bytes ativos).
//! PCA via power iteration simples.
//! Output: ASCII plot 50×30 + SVG.

use std::fs;
use std::io::Write;
use std::path::Path;

fn normalize_byte(b: u8) -> u8 {
    if b.is_ascii_alphabetic() { b | 0x20 }
    else if b.is_ascii_whitespace() { b' ' }
    else { b }
}

fn main() -> std::io::Result<()> {
    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║   byte_projection_2d — bytes plotados via PCA do bigram      ║");
    println!("╚══════════════════════════════════════════════════════════════╝\n");

    // Carrega bigrams do corpus
    let mut text = String::new();
    for p in &["data/dialogues_augmented.txt", "data/qwen_sustained.txt"] {
        if Path::new(p).exists() {
            text.push_str(&fs::read_to_string(p)?);
            text.push('\n');
        }
    }
    println!("  Corpus: {} bytes", text.len());

    let active: Vec<u8> = std::iter::once(b' ')
        .chain(b'0'..=b'9')
        .chain(b'a'..=b'z')
        .collect();
    let n = active.len();
    let idx_of: std::collections::HashMap<u8, usize> =
        active.iter().enumerate().map(|(i, &b)| (b, i)).collect();

    // Build NxN bigram matrix
    let mut m = vec![vec![0u32; n]; n];
    for line in text.lines() {
        let nb: Vec<u8> = line.bytes().map(normalize_byte).collect();
        for w in nb.windows(2) {
            if let (Some(&a), Some(&b)) = (idx_of.get(&w[0]), idx_of.get(&w[1])) {
                m[a][b] += 1;
            }
        }
    }

    // Convert to f32, log-transform (handle skewed dist)
    let mut x: Vec<Vec<f32>> = m.iter().map(|row| {
        row.iter().map(|&v| (1.0 + v as f32).ln()).collect()
    }).collect();

    // Center
    let mean: Vec<f32> = (0..n).map(|j| {
        let s: f32 = x.iter().map(|r| r[j]).sum();
        s / n as f32
    }).collect();
    for row in &mut x {
        for j in 0..n {
            row[j] -= mean[j];
        }
    }

    // Power iteration for top-2 eigenvectors of X^T X
    // We want: 2D projection of rows
    let mut v1 = vec![0f32; n];
    let mut v2 = vec![0f32; n];
    for i in 0..n { v1[i] = ((i * 31) % 7) as f32 + 1.0; v2[i] = ((i * 17) % 5) as f32 + 1.0; }

    fn dot(a: &[f32], b: &[f32]) -> f32 { a.iter().zip(b).map(|(x, y)| x*y).sum() }
    fn norm(a: &[f32]) -> f32 { dot(a, a).sqrt() }
    fn normalize_inplace(a: &mut [f32]) {
        let n = norm(a);
        if n > 1e-9 { for x in a.iter_mut() { *x /= n; } }
    }
    fn project_off(v: &mut [f32], u: &[f32]) {
        let p = dot(v, u);
        for i in 0..v.len() { v[i] -= p * u[i]; }
    }
    fn xtx_mul(x: &[Vec<f32>], v: &[f32]) -> Vec<f32> {
        let n = v.len();
        let xv: Vec<f32> = x.iter().map(|r| dot(r, v)).collect();
        // multiply X^T by xv
        let mut out = vec![0f32; n];
        for i in 0..x.len() {
            let xv_i = xv[i];
            for j in 0..n { out[j] += x[i][j] * xv_i; }
        }
        out
    }

    normalize_inplace(&mut v1);
    for _ in 0..50 {
        v1 = xtx_mul(&x, &v1);
        normalize_inplace(&mut v1);
    }
    normalize_inplace(&mut v2);
    for _ in 0..50 {
        v2 = xtx_mul(&x, &v2);
        project_off(&mut v2, &v1);
        normalize_inplace(&mut v2);
    }

    // Project each byte: (x · v1, x · v2)
    let mut pts: Vec<(u8, f32, f32)> = Vec::new();
    for i in 0..n {
        let p1 = dot(&x[i], &v1);
        let p2 = dot(&x[i], &v2);
        pts.push((active[i], p1, p2));
    }

    let xmin = pts.iter().map(|(_,a,_)| *a).fold(f32::INFINITY, f32::min);
    let xmax = pts.iter().map(|(_,a,_)| *a).fold(f32::NEG_INFINITY, f32::max);
    let ymin = pts.iter().map(|(_,_,b)| *b).fold(f32::INFINITY, f32::min);
    let ymax = pts.iter().map(|(_,_,b)| *b).fold(f32::NEG_INFINITY, f32::max);

    // ASCII plot
    let w = 70usize;
    let h = 24usize;
    let mut grid = vec![vec![' '; w]; h];
    for (b, px, py) in &pts {
        let gx = (((*px - xmin) / (xmax - xmin + 1e-9)) * (w as f32 - 1.0)) as usize;
        let gy = (h as f32 - 1.0 - ((*py - ymin) / (ymax - ymin + 1e-9)) * (h as f32 - 1.0)) as usize;
        let gy = gy.min(h - 1);
        let gx = gx.min(w - 1);
        let ch = if *b == b' ' { '_' } else { *b as char };
        grid[gy][gx] = ch;
    }

    println!("\n▸ Projeção 2D (PCA top-2 do log-bigram profile):\n");
    println!("    PC2 ↑");
    for row in &grid {
        print!("    ");
        for c in row { print!("{}", c); }
        println!();
    }
    println!("    PC1 →");

    // Print coords
    println!("\n▸ Coordenadas (top vs bottom em cada PC):\n");
    let mut by_pc1 = pts.clone();
    by_pc1.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
    println!("  Top 5 PC1: {}",
        by_pc1.iter().take(5).map(|(b,_,_)| if *b == b' ' { "_".to_string() } else { format!("{}", *b as char) }).collect::<Vec<_>>().join(" "));
    println!("  Btm 5 PC1: {}",
        by_pc1.iter().rev().take(5).map(|(b,_,_)| if *b == b' ' { "_".to_string() } else { format!("{}", *b as char) }).collect::<Vec<_>>().join(" "));
    let mut by_pc2 = pts.clone();
    by_pc2.sort_by(|a, b| b.2.partial_cmp(&a.2).unwrap());
    println!("  Top 5 PC2: {}",
        by_pc2.iter().take(5).map(|(b,_,_)| if *b == b' ' { "_".to_string() } else { format!("{}", *b as char) }).collect::<Vec<_>>().join(" "));
    println!("  Btm 5 PC2: {}",
        by_pc2.iter().rev().take(5).map(|(b,_,_)| if *b == b' ' { "_".to_string() } else { format!("{}", *b as char) }).collect::<Vec<_>>().join(" "));

    // SVG export
    let svg_path = "/tmp/byte_projection.svg";
    let mut f = fs::File::create(svg_path)?;
    let svg_w = 800;
    let svg_h = 600;
    writeln!(f, "<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 {} {}'>", svg_w, svg_h)?;
    writeln!(f, "  <rect width='{}' height='{}' fill='#0a0a0a'/>", svg_w, svg_h)?;
    writeln!(f, "  <text x='10' y='20' fill='#00ff88' font-family='monospace'>IARA Byte Projection 2D (PCA log-bigram)</text>")?;
    for (b, px, py) in &pts {
        let cx = 50.0 + ((*px - xmin) / (xmax - xmin + 1e-9)) * (svg_w as f32 - 100.0);
        let cy = 50.0 + (1.0 - (*py - ymin) / (ymax - ymin + 1e-9)) * (svg_h as f32 - 100.0);
        let label = if *b == b' ' { "_".to_string() } else { format!("{}", *b as char) };
        let color = if b.is_ascii_digit() { "#ffe66d" }
                    else if b.is_ascii_alphabetic() { "#4ecdc4" }
                    else { "#ff6b6b" };
        writeln!(f, "  <circle cx='{}' cy='{}' r='14' fill='{}' opacity='0.7'/>", cx, cy, color)?;
        writeln!(f, "  <text x='{}' y='{}' fill='black' font-family='monospace' font-size='14' text-anchor='middle' alignment-baseline='middle'>{}</text>",
                 cx, cy + 4.0, label)?;
    }
    writeln!(f, "</svg>")?;
    println!("\n▸ SVG exportado: {}", svg_path);

    Ok(())
}
