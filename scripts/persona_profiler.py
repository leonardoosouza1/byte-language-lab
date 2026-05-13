#!/usr/bin/env python3
"""PERSONA PROFILER — extrai perfil de usuário a partir de texto.
Combina:
- Métricas stylometric (TTR, sentence len, pontuação)
- Qwen embeddings (estilo geral)
- Detecção emoção/formalidade
- Vocabulário range (formal/informal/code/etc)
"""
import sys, re, struct, json, numpy as np
from collections import Counter

# Sample input — pode vir de stdin ou arg
if len(sys.argv) > 1:
    sample_text = sys.argv[1]
else:
    sample_text = """oi blz td bem? to estudando aqui sobre IA mt loko esse trampo
    quer ver q legal, hoje peguei o qwen e tipo desmontei ele pra ver o q tinha dentro
    achei q ia ser bom pra usar mas não, kkkk só lixo mesmo
    1 só layer não da, precisa todos os 28 senão fica ruim"""

print("═══════════════════════════════════════════════════════════")
print("  PERSONA PROFILE — análise estilométrica")
print("═══════════════════════════════════════════════════════════\n")
print(f"Sample:\n{sample_text}\n")

# ── BASIC STYLOMETRICS ──
text = sample_text.strip()
words = re.findall(r"\b[\w']+\b", text.lower())
sentences = re.split(r'[.!?]+', text)
sentences = [s.strip() for s in sentences if len(s.strip()) > 0]

# 1. TTR (type-token ratio) — diversidade vocab
ttr = len(set(words)) / max(len(words), 1)

# 2. Tamanho médio sentença
avg_sent_len = np.mean([len(re.findall(r"\b[\w']+\b", s)) for s in sentences]) if sentences else 0

# 3. Tamanho médio palavra
avg_word_len = np.mean([len(w) for w in words]) if words else 0

# 4. Ratio pontuação
punct = re.findall(r'[.,!?;:]', text)
punct_ratio = len(punct) / max(len(words), 1)

# 5. Exclamações + interrogações
excl = text.count('!')
quest = text.count('?')

# 6. Caps ratio
caps_chars = sum(1 for c in text if c.isupper())
caps_ratio = caps_chars / max(len(text), 1)

# 7. Internet slang detection
slang_terms = {'kkk', 'kkkk', 'rs', 'rsrs', 'lol', 'lmao', 'wtf', 'cara', 'mano',
               'blz', 'td', 'tb', 'tbm', 'pq', 'q', 'vc', 'mt', 'pra', 'pro',
               'to', 'ta', 'loko', 'trampo', 'tipo', 'sa', 'oi', 'olá'}
slang_count = sum(1 for w in words if w in slang_terms)
slang_ratio = slang_count / max(len(words), 1)

# 8. Code/tech markers
tech_terms = {'ia', 'modelo', 'algoritmo', 'rust', 'python', 'qwen', 'gpu',
              'cpu', 'layer', 'token', 'embed', 'attention', 'transformer',
              'neural', 'dataset', 'treinar', 'inference', 'param', 'tensor',
              'matriz', 'vetor', 'memória', 'memoria', 'bytes', 'sistema'}
tech_count = sum(1 for w in words if w in tech_terms)
tech_ratio = tech_count / max(len(words), 1)

# 9. Numbers
nums = re.findall(r'\d+', text)

# 10. Emotion markers
positive = {'legal', 'massa', 'top', 'show', 'foda', 'maneiro', 'bacana', 'incrível',
            'lindo', 'amei', 'amo', 'gosto', 'feliz', 'alegre', 'ótimo', 'otimo'}
negative = {'ruim', 'lixo', 'merda', 'odeio', 'chato', 'triste', 'porra', 'caralho',
            'pessimo', 'horrível', 'horrivel', 'péssimo'}
emphatic = {'mt', 'muito', 'super', 'mega', 'hiper', 'pra caralho', 'demais'}
pos_count = sum(1 for w in words if w in positive)
neg_count = sum(1 for w in words if w in negative)
emp_count = sum(1 for w in words if w in emphatic)

print("─── MÉTRICAS BASE ───")
print(f"  Total palavras:      {len(words)}")
print(f"  Total frases:        {len(sentences)}")
print(f"  TTR (diversidade):   {ttr:.3f}  ({'alta' if ttr > 0.7 else 'média' if ttr > 0.5 else 'baixa'})")
print(f"  Tamanho médio frase: {avg_sent_len:.1f} palavras")
print(f"  Tamanho médio word:  {avg_word_len:.2f} chars")
print(f"  Pontuação ratio:     {punct_ratio:.3f}")
print(f"  Exclamações: {excl}, Interrogações: {quest}")
print(f"  Caps ratio:          {caps_ratio:.3f}")
print(f"  Números encontrados: {len(nums)}")

print("\n─── PERFIL ESTILÍSTICO ───")
print(f"  Internet slang:   {slang_count}/{len(words)} ({slang_ratio*100:.1f}%)")
print(f"  Tech vocab:       {tech_count}/{len(words)} ({tech_ratio*100:.1f}%)")
print(f"  Emoção positiva:  {pos_count}")
print(f"  Emoção negativa:  {neg_count}")
print(f"  Ênfase:           {emp_count}")

# Classificação automática
print("\n─── PERFIL PERSONA INFERIDO ───")
formality = "informal"
if slang_ratio > 0.05 or excl > 2:
    formality = "casual/informal"
if punct_ratio > 0.15 and slang_ratio < 0.02:
    formality = "formal"
if avg_word_len > 5.5 and tech_ratio < 0.05:
    formality = "literário/erudito"

print(f"  Formalidade:      {formality}")

domain = "geral"
if tech_ratio > 0.05:
    domain = "tech/programação"
if "ia" in words or "modelo" in words or "neural" in words:
    domain = "AI/ML"

print(f"  Domínio:          {domain}")

mood = "neutro"
if pos_count > neg_count and pos_count > 1:
    mood = "positivo"
elif neg_count > pos_count and neg_count > 1:
    mood = "negativo/crítico"
if excl > 3:
    mood += " entusiasmado"

print(f"  Humor:            {mood}")

# Idade aproximada (heurística)
age_signal = "indeterminado"
if slang_ratio > 0.10:
    age_signal = "jovem (alta gíria internet)"
elif slang_ratio < 0.02 and avg_word_len > 5:
    age_signal = "adulto formal"
else:
    age_signal = "adulto comunicativo"
print(f"  Faixa estimada:   {age_signal}")

# Personalidade traits
print(f"\n─── TRAÇOS DETECTADOS ───")
traits = []
if quest > 3 or quest > len(sentences) * 0.4:
    traits.append("✓ Curioso (muitas perguntas)")
if excl > 3:
    traits.append("✓ Expressivo (muitas exclamações)")
if tech_ratio > 0.1:
    traits.append("✓ Técnico/Engenheiro")
if slang_ratio > 0.10:
    traits.append("✓ Comunicador casual")
if avg_sent_len > 15:
    traits.append("✓ Articulado (frases longas)")
if avg_sent_len < 8:
    traits.append("✓ Direto/conciso (frases curtas)")
if caps_ratio > 0.1:
    traits.append("✓ Enfático (CAPS)")
if pos_count + emp_count > 3:
    traits.append("✓ Otimista/entusiasta")

if not traits:
    traits.append("Texto curto demais pra traços fortes")
for t in traits:
    print(f"  {t}")

print("\n─── TOP TERMOS USADOS ───")
word_freq = Counter(words)
common = {'a', 'o', 'que', 'e', 'de', 'do', 'da', 'um', 'uma', 'em', 'para', 'pra',
          'com', 'no', 'na', 'se', 'eu', 'tu', 'ele', 'ela', 'é', 'foi'}
unique_words = [(w, c) for w, c in word_freq.most_common() if w not in common and len(w) >= 3][:8]
for w, c in unique_words:
    print(f"  {w!r}: {c}x")

print("\n═══════════════════════════════════════════════════════════")
print("""  USO PRÁTICO no brain:
  - Detectar persona do USER em runtime
  - Adaptar response style (formal/casual/técnico)
  - Memoizar perfil entre sessões
  - Filtrar tópicos relevantes ao domínio
""")
