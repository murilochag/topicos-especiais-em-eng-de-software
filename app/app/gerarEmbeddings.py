import os
from sentence_transformers import SentenceTransformer
from pinecone import Pinecone

# Caminho da pasta com os arquivos .txt
pasta_txt = "./documentos/"

textos = []
nomes_arquivos = []

if not os.path.exists(pasta_txt):
    print(f"Erro: O diretório {pasta_txt} não existe!")
    exit(1)

arquivos = os.listdir(pasta_txt)
if not arquivos:
    print(f"Erro: Nenhum arquivo encontrado em {pasta_txt}!")
    exit(1)

for arquivo in arquivos:
    if arquivo.endswith(".txt"):
        caminho_arquivo = os.path.join(pasta_txt, arquivo)
        with open(caminho_arquivo, "r", encoding="utf-8") as f:
            conteudo = f.read().strip()
            if conteudo:  # Verifica se o arquivo não está vazio
                textos.append(conteudo)
                nomes_arquivos.append(arquivo)

if not textos:
    print("Erro: Nenhum arquivo válido encontrado!")
    exit(1)

print(f"Total de arquivos carregados: {len(textos)}")


modelo = SentenceTransformer('all-MiniLM-L6-v2')

embeddings = modelo.encode(textos)

if embeddings.shape[0] == 0:
    print("Erro: Nenhum embedding foi gerado!")
    exit(1)

print(f"Embeddings gerados: {embeddings.shape}")  # Formato esperado: (n_textos, dimensão_do_embedding)

PINECONE_API_KEY = ""
INDEX_NAME = "java-books-index"

pc = Pinecone(api_key=PINECONE_API_KEY)

indices_disponiveis = pc.list_indexes().names()
if INDEX_NAME not in indices_disponiveis:
    pc.create_index(
        name=INDEX_NAME,
        dimension=384,  
        metric="cosine"
    )


indice = pc.Index(INDEX_NAME)


dados = [
    (f"id_{i}", embedding.tolist(), {"texto": texto, "arquivo": nome_arquivo})
    for i, (embedding, texto, nome_arquivo) in enumerate(zip(embeddings, textos, nomes_arquivos))
]


indice.upsert(dados)

print("Embeddings armazenados no Pinecone com sucesso!")
