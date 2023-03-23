from haystack.document_stores import FAISSDocumentStore

# document_store = FAISSDocumentStore(embedding_dim=128, faiss_index_factory_str="Flat")

from haystack.utils import convert_files_to_docs, fetch_archive_from_http, clean_wiki_text


# Let's first get some files that we want to use
doc_dir = "data/tutorial12"
s3_url = "https://s3.eu-central-1.amazonaws.com/deepset.ai-farm-qa/datasets/documents/wiki_gameofthrones_txt12.zip"
fetch_archive_from_http(url=s3_url, output_dir=doc_dir)

# Convert files to dicts
docs = convert_files_to_docs(dir_path=doc_dir, clean_func=clean_wiki_text, split_paragraphs=True)

# Now, let's write the dicts containing documents to our DB.
# document_store.write_documents(docs)
import pathlib
document_store = FAISSDocumentStore.load(pathlib.Path('index').resolve())

from haystack.nodes import DensePassageRetriever

retriever = DensePassageRetriever(
    document_store=document_store,
    query_embedding_model="vblagoje/dpr-question_encoder-single-lfqa-wiki",
    passage_embedding_model="vblagoje/dpr-ctx_encoder-single-lfqa-wiki",
)

# document_store.update_embeddings(retriever=retriever, update_existing_embeddings=True)
# explicitly saving index
# import pathlib
# document_store.save(pathlib.Path('index').resolve())

