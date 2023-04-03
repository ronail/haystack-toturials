from haystack.utils import convert_files_to_docs, fetch_archive_from_http, clean_wiki_text

# Let's first get some files that we want to use
# doc_dir = "data/tutorial12"
# s3_url = "https://s3.eu-central-1.amazonaws.com/deepset.ai-farm-qa/datasets/documents/wiki_gameofthrones_txt12.zip"
# fetch_archive_from_http(url=s3_url, output_dir=doc_dir)
# Convert files to dicts
# docs = convert_files_to_docs(dir_path=doc_dir, clean_func=clean_wiki_text, split_paragraphs=True)


from haystack.nodes import Crawler, PreProcessor
import os
doc_dir = f"data/{os.environ.get('CRAWL_HOST','faq')}"

from haystack import Pipeline
indexing_pipeline = Pipeline()

preprocessor = PreProcessor(
    clean_empty_lines=True,
    clean_whitespace=True,
    clean_header_footer=False,
    split_by="word",
    split_length=500,
    split_respect_sentence_boundary=True,
)

if not os.path.isdir(doc_dir):
  print(f"Crawling into {doc_dir}...") 
  # crawl Haystack docs, i.e. all pages that include haystack.deepset.ai/overview/
  crawler = Crawler(
      output_dir=doc_dir,
      urls=[f"https://{os.environ.get('CRAWL_HOST')}"],
      crawler_depth=1
  )

  indexing_pipeline.add_node(component=crawler, name="Crawler", inputs=['File'])
  # If SSL error is raised on MacOS, follow this https://stackoverflow.com/a/46167270
  indexing_pipeline.add_node(component=preprocessor, name="Preprocessore", inputs=["Crawler"])
  docs = indexing_pipeline.run_batch(params={"Crawler": {'return_documents': True}})["documents"]
else:
  print(f"Loading archive from {doc_dir}...") 
  # importing existings archive
  from haystack.nodes import JsonConverter
  json_converter = JsonConverter()
  indexing_pipeline.add_node(component=json_converter, name="JsonConverter", inputs=["File"])
  indexing_pipeline.add_node(component=preprocessor, name="Preprocessore", inputs=["JsonConverter"])

  import os
  files_to_index = [doc_dir + "/" + f for f in os.listdir(doc_dir)]
  docs = indexing_pipeline.run_batch(params={"JsonConverter": {'file_paths': files_to_index}})["documents"]
