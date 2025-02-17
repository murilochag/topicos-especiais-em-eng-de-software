from sentence_transformers import SentenceTransformer
from pinecone import Pinecone, ServerlessSpec
from langchain_ollama import OllamaLLM
import re
import os

pc = Pinecone(api_key="", # me contate em murilo.carvalho@icen.ufpa.br
              environment="us-east1-aws")
nome_indice = "kotlin-doc"
indice = pc.Index(nome_indice)

# Carregar o modelo de embeddings (Sentence Transformer)
modelo_embedding = SentenceTransformer('all-MiniLM-L6-v2')

# Função para buscar documentos no Pinecone
def buscar_documentos(query, top_k=5):
    query_embedding = modelo_embedding.encode([query]).tolist()

    resultados = indice.query(vector=query_embedding, top_k=top_k, include_metadata=True)

    documentos = [match["metadata"].get("paragrafo", "Parágrafo não disponível") for match in resultados["matches"]]
    print(documentos)
    return documentos

def gerar_resposta(query, documentos):
    contexto = " ".join(documentos)
    prompt = f"Contexto: {contexto}\nPergunta: {query}\nResposta:"
    
    try:
        model = OllamaLLM(model="llama3.1:latest")
        resposta = model.invoke(input=prompt)
    except Exception as e:
        resposta = f"Erro ao gerar resposta: {str(e)}"
    
    return resposta

def qa_rag(query):
    documentos = buscar_documentos(query)
    if not documentos:
        return "Não há informações suficientes para responder à pergunta."
    resposta = gerar_resposta(query, documentos)
    return resposta

query = input("pergunta:")
resposta = qa_rag(query)
print(resposta)