import os
import time

from haystack.document_stores import OpenSearchDocumentStore
from haystack.nodes import BM25Retriever, EmbeddingRetriever, FARMReader
import docs

# Wait 30 seconds only to be sure Elasticsearch is ready before continuing

# Get the host where Elasticsearch is running, default to localhost
host = os.environ.get("ELASTICSEARCH_HOST", "localhost")

# Initialize DocumentStore and index documents
document_store = OpenSearchDocumentStore(host=host, port=1443, username='')
document_store.delete_documents()
document_store.write_documents(docs.got_docs)

# Initialize Sparse retriever
bm25_retriever = BM25Retriever(document_store=document_store)

# Initialize dense retriever
embedding_retriever = EmbeddingRetriever(
    document_store=document_store, embedding_model="sentence-transformers/multi-qa-mpnet-base-dot-v1"
)
document_store.update_embeddings(embedding_retriever, update_existing_embeddings=False)

# Initialize reader
reader = FARMReader(model_name_or_path="deepset/roberta-base-squad2")
