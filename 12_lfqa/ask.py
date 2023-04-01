from haystack.utils import (print_documents, print_answers)
# from haystack.pipelines import DocumentSearchPipeline
from document_store import document_store

import os.path
if not os.path.exists("index.faiss"):
# Now, let's write the dicts containing documents to our DB.
  from docs import docs
  document_store.write_documents(docs)


from retriever import retriever

import os.path
if not os.path.exists("index.faiss"):
  document_store.save(index_path="index.faiss")

# p_retrieval = DocumentSearchPipeline(retriever)
# res = p_retrieval.run(query="Tell me something about Arya Stark?", params={"Retriever": {"top_k": 10}})
# print_documents(res, max_text_len=512)

from haystack.nodes import Seq2SeqGenerator

generator = Seq2SeqGenerator(model_name_or_path="vblagoje/bart_lfqa")

from haystack.pipelines import GenerativeQAPipeline

pipe = GenerativeQAPipeline(generator, retriever)

query = input("What do you want to know?\n")
if not query:
    query = "Why is Arya Stark an unusual character?"
print(f"Asking '{query}'...")    
answer = pipe.run(query=query, params={"Retriever": {"top_k": 3}})
print_answers(answer, details="minimum")

