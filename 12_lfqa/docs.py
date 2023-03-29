from haystack.utils import convert_files_to_docs, fetch_archive_from_http, clean_wiki_text

# Let's first get some files that we want to use
# doc_dir = "data/tutorial12"
# s3_url = "https://s3.eu-central-1.amazonaws.com/deepset.ai-farm-qa/datasets/documents/wiki_gameofthrones_txt12.zip"
# fetch_archive_from_http(url=s3_url, output_dir=doc_dir)

from haystack.nodes import Crawler, PreProcessor

doc_dir = "faq"
host = "faq-domain"
# crawl Haystack docs, i.e. all pages that include haystack.deepset.ai/overview/
crawler = Crawler(
    output_dir=doc_dir,
    urls=[f"https://{host}"],
  crawler_depth=1
  )


preprocessor = PreProcessor(
    clean_empty_lines=True,
    clean_whitespace=True,
    clean_header_footer=False,
    split_by="word",
    split_length=500,
    split_respect_sentence_boundary=True,
)



from haystack import Pipeline
indexing_pipeline = Pipeline()
# indexing_pipeline.add_node(component=crawler, name="Crawler", inputs=['File'])
from haystack.nodes import JsonConverter
json_converter = JsonConverter()
indexing_pipeline.add_node(component=json_converter, name="JsonConverter", inputs=["File"])
# If SSL error is raised on MacOS, follow this https://stackoverflow.com/a/46167270
# indexing_pipeline.add_node(component=preprocessor, name="Preprocessore", inputs=["Crawler"])
indexing_pipeline.add_node(component=preprocessor, name="Preprocessore", inputs=["JsonConverter"])
# docs = indexing_pipeline.run_batch(params={"Crawler": {'return_documents': True}})["documents"]

import os
files_to_index = [doc_dir + "/" + f for f in os.listdir(doc_dir)]
docs = indexing_pipeline.run_batch(file_paths=files_to_index)["documents"]
