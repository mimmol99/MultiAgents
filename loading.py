from langchain_core.output_parsers.json import SimpleJsonOutputParser
from concurrent.futures import ThreadPoolExecutor, as_completed
from langchain_community.document_loaders import DirectoryLoader
from langchain_community.document_loaders import WebBaseLoader
import validators 
import re
from typing import Union, List

class Loader:

    def __init__(self, file_directory: Union[str, List[str]]):
        self.file_directory = file_directory if isinstance(file_directory, list) else [file_directory]
        self.all_docs = []

    def normalize_doc(self, doc):
        s = doc.page_content
        s = re.sub(r'\s+', ' ', s).strip()
        s = re.sub(r". ,", "", s)
        s = s.replace("..", ".")
        s = s.replace(". .", ".")
        s = s.replace("\n", "")
        s = s.strip()
        doc.page_content = s.lower()
        return doc       

    def load_documents(self, accepted_files: List[str] = None):
        # Use a list of patterns to match specific file types
        patterns = ["**/*." + f for f in accepted_files] if accepted_files else ["**/*"]
        all_docs = []

        for directory in self.file_directory:
            for pattern in patterns:
                loader = DirectoryLoader(directory, glob=pattern, use_multithreading=True, show_progress=False)
                current_docs = loader.load()
                current_docs = [self.normalize_doc(doc) for doc in current_docs]
                all_docs.extend(current_docs)

        return all_docs

    def is_string_an_url(self, url_string: str) -> bool:
        result = validators.url(url_string)
        return result is True

    def load_urls(self, urls: List[str]):
        urls = [url for url in urls if self.is_string_an_url(url) is True]
        loader = WebBaseLoader(urls)
        docs = loader.load()
        docs = [self.normalize_doc(doc) for doc in docs]
        return docs
    
    





  
       

    
      
