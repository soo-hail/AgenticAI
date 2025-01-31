# AGENTIC-PART: WE PROVIDE LLMS THE TOOLS THAT AUTOMATES THE WORK.
# HERE WE ARE PROVING TWO TOOLS, PDF-READER AND WEB-READER TO AUTOMATE THE SEARCH.

import os
from typing import Type
# 'BaseTool' SUPER-CLASS TO CREATE TOOLS.
from crewai.tools import BaseTool
# 'pydantic' TOOL FOR DATA-VALIDATION.
from pydantic import BaseModel, Field, ConfigDict
# BASEMODEL - FOR DATA VALIDATION, WE EXTEND THIS CLASS.
# FIELD - TO PROVIDE ADDITIONAL METADATA, CONSTRAINS.

from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter # TO CHUNK THE TEXT
from langchain.embeddings import OllamaEmbeddings
from langchain.vectorstores import FAISS

import streamlit as st

class DocumentSearchToolInput(BaseModel):
    # DEFINE VARIABLES WITH DATA-VALIDATION.
    query: str = Field(..., description="Query to search the document.")
    # QUERY SHOULD BE STRING, OTHERWISE AN VALIDATION ERROR IS RAISED.

class DocumentSearchTool(BaseTool):
    
    name: str = "DocumentSearchTool"
    description: str = "Search the document for the given query."
    args_schema: Type[BaseModel] = DocumentSearchToolInput
    
    model_config = ConfigDict(extra="allow")
    def __init__(self, pdf_file):
        '''  
            Initialize Search With a PDF-File.
        '''
        super().__init__() # CALL SUPER-CLASS CONTRACTOR.
        self.pdf_file = pdf_file
        self.vector_store = self._process_document()
        
    # FUNCTION-NAME STARTING WITH AN '_'(UNDERSCORE) IS A CONVENTION THAT INDICATES THE FUNCTION IS INTENDED TO BE PRIVATE. 
    def _extract_text(self) -> str:
        ''' EXTRACT RAW TEXT FROM PDF USING MARKITDOWN.'''
        text = ""
            
        # ENSURE 'PDF_file' IS ITERABLE (SINGLE FILE CASE).
        pdf_files = self.pdf_file if isinstance(self.pdf_file, list) else [self.pdf_file]

        for pdf in pdf_files:
            pdf_reader = PdfReader(pdf)
            for page in pdf_reader.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text  # Append extracted text
                        
        return text
            
        
    def _create_chunks(self, raw_text: str) -> list:
        ''' CREATE SEMATIC CHUNKS FROM RAW TEXT.'''
        text_splitter = CharacterTextSplitter(separator='\n', chunk_size=1000, chunk_overlap=200, length_function=len)
        chunks = text_splitter.split_text(raw_text)
            
        return chunks
            
    def _process_document(self):
        '''PROCESS THE DOMENT AND ADD CHUNKS(EMBEDDINGS) TO QDRANT DATABASE.'''
        raw_text = self._extract_text()
        text_chunks = self._create_chunks(raw_text)
        
        embeddings = OllamaEmbeddings()
        
        chunk_embeddings = embeddings.embed_documents(text_chunks)
    
        text_embedding_pairs = list(zip(text_chunks, chunk_embeddings))
    
        # Now, create a FAISS index with the text-embedding pairs
        vector_store = FAISS.from_embeddings(text_embedding_pairs, embeddings)

        return vector_store
    
    def _run(self, query: str) -> list:
        '''Search Document With a Query'''
        result = self.vector_store.similarity_search(query)

        return [res.page_content for res in result]


# TEST THE IMPLMENTATION.
def test_document_searcher():
    pdf_file = st.file_uploader('Upload PDF', accept_multiple_files=True)
    
    if pdf_file:
        
        searcher = DocumentSearchTool(pdf_file=pdf_file)
    
        result = searcher._run('the house was dark and musty, with dust covering every surface')
        print(result)

if __name__ == "__main__":
    test_document_searcher()
        