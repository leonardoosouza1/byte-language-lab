import random
import os

ARQUIVO = "perguntas.txt"
TOTAL_DESEJADO = 5000
CHECKPOINTS = [0, 10, 50, 100, 500, 1000, 2000, 3000, 4000, 5000]

# 1. Ações (Verbos)
verbos = [
    "explique", "descreva", "analise", "resuma", "defina", "discorra sobre", 
    "detalhe", "esclareça", "comente sobre", "apresente os fundamentos de",
    "quais as bases de", "mostre o impacto de", "contextualize", "avalie"
]

# 2. Itens (Temas específicos)
itens = {
    "Tech_LowLevel": ["Rust ownership", "ponteiros C", "instruções AVX-512", "eBPF", "drivers de GPU", "LNNs", "Triton kernels", "escalonamento RTOS", "pila de protocolos Zigbee", "instruções RISC-V", "Vulkan API", "alocação SLAB"],
    "IA_Avancada": ["backpropagation", "transformers", "atenção esparsa", "redes neurais líquidas", "inferência quantizada", "embeddings de grafos", "fine-tuning LoRA", "RAG (Retrieval-Augmented Generation)", "convoluções 3D", "GANs"],
    "Medicina": ["oncogênese", "farmacocinética", "neurotransmissores", "sistema linfático", "ciclo de Krebs", "fisiologia renal", "imunologia", "anatomia patológica", "virologia", "bioquímica clínica", "cardiologia", "endocrinologia"],
    "Engenharia": ["termodinâmica", "mecânica de fluidos", "resistência dos materiais", "leis de Maxwell", "circuitos integrados", "hidráulica", "aerodinâmica", "automação CLP", "estruturas metálicas"],
    "Direito_Sociedade": ["LGPD", "direito civil", "propriedade intelectual", "geopolítica", "macroeconomia", "sociologia", "antropologia", "ética na IA", "contratos inteligentes", "teoria do valor"],
    "Artes_Lutas": ["biomecânica do Hapkido", "Taekwondo", "Jiu-Jitsu", "harmonia musical", "pintura renascentista", "arquitetura moderna", "cinema noir", "escultura", "composição de jazz"],
    "Outros": ["extração de café", "máquinas Nespresso", "fermentação de pães", "astronomia", "cultivo de bonsai", "relojoaria", "cutelaria", "fotografia analógica", "hidroponia", "paleontologia"]
}

# 3. Contextos (O multiplicador secreto)
contextos = [
    "no cenário atual", "sob uma perspectiva técnica", "para iniciantes", "em detalhes profundos",
    "considerando a história", "na prática do dia a dia", "visando performance", "em um contexto global",
    "com foco em eficiência", "segundo a literatura acadêmica", "de forma simplificada", "de forma avançada"
]

perguntas_existentes = set()
if os.path.exists(ARQUIVO):
    with open(ARQUIVO, "r", encoding="utf-8") as f:
        perguntas_existentes = {linha.strip() for linha in f if linha.strip()}

print(f"🚀 Iniciando geração inesgotável. Meta: {TOTAL_DESEJADO}")

with open(ARQUIVO, "a", encoding="utf-8") as f:
    tentativas_vazias = 0
    
    while len(perguntas_existentes) < TOTAL_DESEJADO and tentativas_vazias < 10000:
        categoria = random.choice(list(itens.keys()))
        item = random.choice(itens[categoria])
        verbo = random.choice(verbos)
        contexto = random.choice(contextos)
        
        # A combinação tripla cria milhões de possibilidades
        nova_pergunta = f"{verbo} {item} {contexto}".strip()

        if nova_pergunta not in perguntas_existentes:
            f.write(nova_pergunta + "\n")
            f.flush()
            perguntas_existentes.add(nova_pergunta)
            tentativas_vazias = 0
            
            atual = len(perguntas_existentes)
            if atual in CHECKPOINTS or (atual > 100 and atual % 500 == 0):
                print(f"📦 Log: {atual} perguntas validadas... [{categoria}]")
        else:
            tentativas_vazias += 1

print(f"\n✅ Concluído! O arquivo '{ARQUIVO}' agora tem {len(perguntas_existentes)} perguntas únicas.")
