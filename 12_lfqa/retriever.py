from haystack.nodes import DensePassageRetriever
import document_store

from haystack.document_stores import FAISSDocumentStore

retriever = DensePassageRetriever(
    document_store=document_store.document_store,
    query_embedding_model="vblagoje/dpr-question_encoder-single-lfqa-wiki",
    passage_embedding_model="vblagoje/dpr-ctx_encoder-single-lfqa-wiki",
)


import os.path
if not os.path.exists("index.faiss"):
  document_store.document_store.update_embeddings(retriever, update_existing_embeddings=False)
