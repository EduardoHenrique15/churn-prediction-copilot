# Rascunho do post para o LinkedIn

Anexe o vídeo `demo.mp4` (enviado junto com os arquivos; no LinkedIn, vídeo nativo fica mais nítido que GIF) ou as imagens de `docs/screenshots/`, e troque os links depois do deploy.

---

Um modelo de churn que diz "este cliente tem 72% de chance de cancelar" parece pronto. Mas 72% de quê? E vale a pena gastar uma oferta com ele?

Construí o **Churn Radar** para responder essas duas perguntas — do dado bruto ao produto publicado.

**O que ele faz**
Prevê quais clientes de uma operadora vão cancelar, explica o porquê de cada previsão e calcula o valor esperado de contatar cada um. Tem API, interface web e um assistente de IA que consulta o modelo por function calling e a documentação por RAG.

**As decisões que mais me ensinaram**
- **Probabilidade calibrada antes de tudo.** A versão inicial reponderava as classes e previa 38% de risco médio para uma base com 27% de cancelamento. Sem a reponderação, o erro de calibração caiu de 11,7 para 3,1 pontos percentuais — e aí dá para fazer conta de dinheiro com o número.
- **Corte de decisão por valor, não 0,5.** Contatar compensa quando risco × chance de a oferta funcionar × valor do cliente supera o custo da oferta. Nos dados, isso dá 32%: a lista encontra 73% dos cancelamentos contatando 36% da base.
- **O modelo mais simples dentro do ruído.** O Gradient Boosting ganhou por 0,007 de average precision — menos que 1 erro-padrão. Fiquei com a regressão logística: explicação exata de cada previsão e uma API que roda com ~170 MB de RAM.
- **Achei e corrigi um bug de train/serve skew:** na API, um cliente sozinho perdia todas as categorias no one-hot e virava o cliente "padrão". Hoje há testes que travam isso.

**Stack:** Python · scikit-learn · FastAPI · Streamlit · LangChain · Gemini · ChromaDB · MLflow · Docker · GitHub Actions (263 testes no CI)

Teste ao vivo: [link da interface]
Código e decisões documentadas: [link do GitHub]

Se você trabalha com retenção, dados ou ML em produção, adoraria ouvir o que faria diferente.

#DataScience #MachineLearning #MLOps #IA #Python #Churn

---

**Dicas para publicar:** responda os comentários nas primeiras horas e marque a CESAR School se fizer sentido. Muita gente relata alcance menor em posts com link no corpo; se quiser testar, deixe os dois links no primeiro comentário.
