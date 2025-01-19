from sentence_transformers import SentenceTransformer

from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

model = SentenceTransformer("all-MiniLM-L6-v2")


# informações sobre meia entrada do site ingressos.com
faq = [
    {
        "pergunta": "Criança paga ingresso no cinema?", 
        "resposta": "A Lei da Meia-Entrada garante o benefício para estudantes, pessoas com deficiência (e acompanhante, quando necessário) e jovens de baixa renda com idade entre 15 e 29 anos inscritos no CadÚnico. A Lei não faz menção sobre isenção de pagamento para crianças.\nCom base no Estatuto da Criança e do Adolescente (Lei nº 8.069/1990), a maioria das redes de cinema concede o benefício de meia-entrada para a faixa etária de 3 a 12 anos. Antes de realizar sua compra, você deve conferir a política de cada estabelecimento entrando em contato com as respectivas redes de cinemas."
    },
    {
        "pergunta": "Sou idoso e possuo direito a meia-entrada, como devo fazer minha compra?", 
        "resposta": "Idosos maiores de 60 anos tem direito à meia-entrada e para realizar sua compra conosco, basta escolher o tipo de ingresso meia-entrada (ou Senior, se houver essa opção) durante o processo de compra e, no campo do nº do documento, coloque a numeração do seu RG ou de um documento oficial com foto. Não esqueça de levar esse documento para mostrar na entrada da sala do cinema."
    },
    {
        "pergunta": "Sou estudante, mas não tenho carteirinha. Posso comprar meia-entrada?",
        "resposta": "Não, somente estudantes devidamente identificados poderão usufruir do direito de compra de ingressos meia-entrada."
    },
    {
        "pergunta": "Gostaria saber sobre a política de Meia-Entrada",
        "resposta": "A política de meia-entrada é regulamentada pela Lei Federal nº 12.933/2013, que garante o direito de pagar metade do valor do ingresso em eventos culturais, esportivos e de lazer\nSaiba mais em: https://atendimento.ingresso.com/portal/pt-br/kb/articles/politica-de-meia-entrada"
    },
    {
        "pergunta": "Qual documento devo informar para comprovar meia-entrada?",
        "resposta": "Estudantes - Lei Federal 12.933/13, Decreto Federal 8.537/15.\nO que levar: Carteira de Identificação Estudantil - CIE - documento emitido pela Associação Nacional de Pós-Graduandos (ANPG), pela União Nacional dos Estudantes (UNE), pela União Brasileira dos Estudantes Secundaristas (Ubes), pelas entidades estaduais e municipais filiadas àquelas, pelos Diretórios Centrais dos Estudantes (DCEs) e pelos Centros e Diretórios Acadêmicos que comprova a condição de estudante regularmente matriculado nos níveis e modalidades de educação e ensino previstos no Título V da Lei nº 9.394, de 1996, conforme modelo único nacionalmente padronizado, com certificação digital e que pode ter cinquenta por cento de características locais.\nNa carteira deverão constar os seguintes elementos: nome completo e data de nascimento do estudante; foto recente do estudante; nome da instituição de ensino na qual o estudante esteja matriculado; grau de escolaridade; e data de validade até o dia 31 de março do ano subsequente ao de sua expedição. Não serão aceitos em nenhuma hipótese boleto bancário ou comprovante de mensalidade. Veja modelo no link: www.documentodoestudante.com.br\n\nMaiores de 60 anos - Lei Federal 10.741/03 e Decreto Federal 8.537/15\nO que levar: Documento de identidade oficial com foto.\n\nJovens de Baixa Renda de 15 a 29 anos - Lei Federal 12.933/13 e Decreto Federal 8.537/15\nO que levar: Carteira de Identidade Jovem emitida pela Secretaria Nacional de Juventude, acompanhada de documento de identidade oficial com foto.\n\nPcD - Lei Federal 12.933/13 e Decreto Federal 8.537/15\nO que levar: Cartão de Benefício de Prestação Continuada da Assistência Social da Pessoa com Deficiência ou de documento emitido pelo Instituto Nacional do Seguro Social - INSS que ateste a aposentadoria de acordo com os critérios estabelecidos na Lei Complementar nº 142, de 8 de maio de 2013. No momento de apresentação, esses documentos deverão estar acompanhados de documento de identidade oficial com foto."
    },
    {
        "pergunta": "Como comprar ingresso de meia-entrada?",
        "resposta": "Após escolher a sessão, filme e o cinema, quando estiver na tela para selecionar o tipo de ingresso, escolha o tipo de ingresso “meia-entrada”:\n\nAo escolher esse ingresso, será necessário informar o número do documento de meia-entrada que será apresentada na entrada da sala de exibição. Exemplo: se você comprar ingressos para estudantes, deve informar o número da Carteira de Identificação Estudantil; se estiver comprando para um Idoso, informe o número do documento de identidade ou documento oficial com foto e assim por diante.\n\nPara cada ingresso meia-entrada comprado deverá ser preenchido e apresentado um comprovante para liberação do uso do ingresso."
    }
]

perguntas = [item["pergunta"] for item in faq]
faq_embeddings = model.encode(perguntas)


nova_pergunta = input("Digite sua pergunta: ")
nova_pergunta_embedding = model.encode([nova_pergunta])


similaridades = cosine_similarity([nova_pergunta_embedding[0]], faq_embeddings)[0]

indice_mais_similar = np.argmax(similaridades)


pergunta_similar = faq[indice_mais_similar]["pergunta"]
resposta_similar = faq[indice_mais_similar]["resposta"]


## print("\nPergunta mais similar encontrada no FAQ:")
print(f"Pergunta: {pergunta_similar}")
print(f"Resposta: {resposta_similar}")
