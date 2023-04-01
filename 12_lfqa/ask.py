from haystack.utils import (print_documents, print_answers)
# from haystack.pipelines import DocumentSearchPipeline
from docs import docs
from document_store import document_store

import os.path
if not os.path.exists("index.faiss"):
# Now, let's write the dicts containing documents to our DB.
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

answer = pipe.run(
    query="How did Arya Stark's character get portrayed in a television adaptation?", params={"Retriever": {"top_k": 3}}
)
print_answers(answer, details="minimum")
answer = pipe.run(query="Why is Arya Stark an unusual character?", params={"Retriever": {"top_k": 3}})
print_answers(answer, details="minimum")

