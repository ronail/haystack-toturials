from haystack.utils import print_documents
from haystack.pipelines import DocumentSearchPipeline
import retriever

p_retrieval = DocumentSearchPipeline(retriever.retriever)
res = p_retrieval.run(query="Tell me something about Arya Stark?", params={"Retriever": {"top_k": 10}})
print_documents(res, max_text_len=512)

