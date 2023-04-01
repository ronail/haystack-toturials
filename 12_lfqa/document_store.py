from haystack.document_stores import FAISSDocumentStore
import os.path

if os.path.exists("index.faiss"):
  document_store = FAISSDocumentStore(faiss_index_path="index.faiss")
else:
  document_store = FAISSDocumentStore(embedding_dim=1024, faiss_index_factory_str="Flat")
