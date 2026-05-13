#!/usr/bin/env python3
"""Clean + dedupe corpus dialogs.
- Remove Qwen scaffolding boilerplate
- Strip citation refs [1, 2]
- Dedupe by user+model pair
- Cap repetitions of common templates
"""
import sys, re, hashlib, glob, os

CORPUS_FILES = [
    'data/dialogues_augmented.txt',
    'data/qwen_sustained.txt',
    'data/online_learning.txt',
    'data/dialogues_pt.txt',
    'data/corpus_literario_pt.txt',
    'data/teste_livro_pt.txt',
    'data/pagina_longa_pt.txt',
    'data/knowledge_extra.txt',
] + sorted(glob.glob('data/nlm_research_*.txt'))

BOILERPLATE = [
    r'Aqui estão alguns pontos importantes sobre',
    r'Aqui estão algumas informações',
    r'Espero que isso ajude',
    r'Vou te explicar',
    r'Você pode verificar essas informações',
    r'\[[\d,\s\-]+\]',  # citation refs
]

def strip_boilerplate(text):
    for pattern in BOILERPLATE:
        text = re.sub(pattern, '', text, flags=re.IGNORECASE)
    return re.sub(r'\s+', ' ', text).strip()

def extract_pairs(text):
    out = []
    chunks = re.findall(r'USER:\s*(.+?)(?=MODEL:)\s*MODEL:\s*(.+?)(?=USER:|$)', text, re.DOTALL)
    for u, m in chunks:
        u = strip_boilerplate(u)
        m = strip_boilerplate(m)
        if 3 <= len(u) <= 200 and 5 <= len(m) <= 400:
            out.append((u, m))
    return out

def main():
    all_pairs = []
    seen = set()
    raw_count = 0
    for path in CORPUS_FILES:
        if not os.path.exists(path):
            continue
        with open(path, 'rb') as f:
            text = f.read().decode('utf-8', errors='ignore')
        pairs = extract_pairs(text)
        raw_count += len(pairs)
        for u, m in pairs:
            # Dedupe by lowercase user (cap repetitions)
            key = hashlib.md5(u.lower().encode()).hexdigest()[:12]
            if key not in seen:
                seen.add(key)
                all_pairs.append((u, m))
    out_path = 'data/cleaned_dialogs.txt'
    with open(out_path, 'w', encoding='utf-8') as f:
        for u, m in all_pairs:
            f.write(f'USER: {u}\nMODEL: {m}\n\n')
    print(f'Raw pairs: {raw_count}')
    print(f'Deduped:   {len(all_pairs)}')
    print(f'Reduction: {(raw_count-len(all_pairs))/max(raw_count,1)*100:.0f}%')
    print(f'Output:    {out_path}  ({os.path.getsize(out_path)} bytes)')

if __name__ == '__main__':
    main()
