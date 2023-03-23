import retriever

from haystack.nodes import Seq2SeqGenerator

generator = Seq2SeqGenerator(model_name_or_path="vblagoje/bart_lfqa")

from haystack.pipelines import GenerativeQAPipeline

pipe = GenerativeQAPipeline(generator, retriever.retriever)

answer = pipe.run(
    query="How did Arya Stark's character get portrayed in a television adaptation?", params={"Retriever": {"top_k": 3}}
)

from haystack.utils import (print_answers, print_documents)

print_answers(answer, details="medium")
