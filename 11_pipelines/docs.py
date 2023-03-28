from haystack.utils import fetch_archive_from_http, convert_files_to_docs, clean_wiki_text

# Download and prepare data - 517 Wikipedia articles for Game of Thrones
doc_dir = "data/tutorial11"
s3_url = "https://s3.eu-central-1.amazonaws.com/deepset.ai-farm-qa/datasets/documents/wiki_gameofthrones_txt11.zip"
fetch_archive_from_http(url=s3_url, output_dir=doc_dir)

# convert files to dicts containing documents that can be indexed to our datastore
got_docs = convert_files_to_docs(dir_path=doc_dir, clean_func=clean_wiki_text, split_paragraphs=True)
