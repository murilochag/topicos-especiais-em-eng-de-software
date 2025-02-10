from sentence_transformers import SentenceTransformer
from pinecone import Pinecone, ServerlessSpec
from langchain_ollama import OllamaLLM
import os

# Configuração do Pinecone (Recuperador)
pc = Pinecone(api_key="", 
              environment="us-east1-aws")

# Nome do índice e criação, se necessário
nome_indice = "java-books-index"
if nome_indice not in pc.list_indexes().names():
    pc.create_index(
        name=nome_indice,
        dimension=1536,  # Ajuste a dimensão conforme seu modelo
        metric='euclidean',
        spec=ServerlessSpec(cloud='aws', region='us-east-1')
    )

# Acessar o índice
indice = pc.Index(nome_indice)

# Carregar o modelo de embeddings (Sentence Transformer)
modelo_embedding = SentenceTransformer('all-MiniLM-L6-v2')

# Função para buscar documentos no Pinecone
def buscar_documentos(query, top_k=5):
    # Converter a consulta em embeddings
    query_embedding = modelo_embedding.encode([query]).tolist()
    
    # Consultar o Pinecone para documentos semelhantes
    resultados = indice.query(vector=query_embedding, top_k=top_k, include_metadata=True)
    
    # Extrair os documentos dos resultados
    documentos = [match["metadata"]["texto"] for match in resultados["matches"]]
    return documentos


# Função para gerar a resposta com o modelo Ollama (Gerador)
def gerar_resposta(query, documentos):
    contexto = " ".join(documentos)
    prompt = f"Contexto: {contexto}\nPergunta: {query}\nResposta:"
    
    # Usar o Ollama para gerar a resposta com base no contexto
    try:
        model = OllamaLLM(model="llama3.1:latest")
        resposta = model.invoke(input=prompt)  # Passando o prompt para o modelo
    except Exception as e:
        resposta = f"Erro ao gerar resposta: {str(e)}"
    
    return resposta

# Função principal para integrar os dois componentes
def qa_rag(query):
    # Passo 1: Recuperar documentos
    documentos = buscar_documentos(query)
    if not documentos:
        return "Não há informações suficientes para responder à pergunta."
    
    # Passo 2: Gerar resposta com base nos documentos recuperados
    resposta = gerar_resposta(query, documentos)
    return resposta

# Teste da função principal
query = "O que é o null safe ou segurança nula?"
resposta = qa_rag(query)
print(resposta)
