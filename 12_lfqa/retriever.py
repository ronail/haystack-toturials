from haystack.nodes import DensePassageRetriever
from document_store import document_store

from haystack.document_stores import FAISSDocumentStore

# retriever = DensePassageRetriever(
#     document_store=document_store.document_store,
#     query_embedding_model="vblagoje/dpr-question_encoder-single-lfqa-wiki",
#     passage_embedding_model="vblagoje/dpr-ctx_encoder-single-lfqa-wiki",
# )

from haystack.nodes import EmbeddingRetriever
import os
retriever = EmbeddingRetriever(
   document_store=document_store,
   batch_size=8,
   embedding_model="ada",
   api_key=os.environ.get('OPENAI_KEY'),
   max_seq_len=1024
)


import os.path
if not os.path.exists("index.faiss"):
  document_store.update_embeddings(retriever, update_existing_embeddings=False)
