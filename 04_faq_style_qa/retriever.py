from haystack.nodes import EmbeddingRetriever
from document_store import document_store

retriever = EmbeddingRetriever(
    document_store=document_store,
    embedding_model="sentence-transformers/all-MiniLM-L6-v2",
    use_gpu=True,
    scale_score=False,
)
