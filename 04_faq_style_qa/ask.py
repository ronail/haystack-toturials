from docs import docs
from document_store import document_store

document_store.write_documents(docs)

from haystack.pipelines import FAQPipeline
from retriever import retriever

pipe = FAQPipeline(retriever=retriever)

from haystack.utils import print_answers


if __name__ == "__main__":
  # Run any question and change top_k to see more or less answers
  prediction = pipe.run(query="How is the virus spreading?", params={"Retriever": {"top_k": 1}})
  print_answers(prediction, details="medium")

