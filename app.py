import streamlit as st
import pinecone
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from PyPDF2 import PdfReader
from langchain.vectorstores import Pinecone


# Initialize Pinecone
PINECONE_API_KEY = st.secrets['PINECONE_API_KEY']
PINECONE_API_ENV = "gcp-starter"
index_name = "testing"

pinecone.init(api_key=PINECONE_API_KEY, environment=PINECONE_API_ENV)

# Initilize OpenAI
OPENAI_API_KEY = st.secrets['OPENAI_API_KEY']
embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)


def get_pdf_text(pdf_docs):
    text = ""
    pdf_reader = PdfReader(pdf_docs)
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text


def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=2000,
        chunk_overlap=200,
    )
    chunks = text_splitter.split_text(text)
    return chunks


def get_vectorstore(text_chunks, pdf_name):
    embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
    meta = [{'filename' : pdf_name} for _ in range(len(text_chunks))]
    vectorstore = Pinecone.from_texts(text_chunks, embeddings, index_name=index_name, metadatas=meta)
    return vectorstore


def main():
    st.set_page_config(page_title="Upload Files", page_icon=":outbox_tray:")
    st.header("Upload Files :outbox_tray:")
    # st.subheader("Your documents")
    pdf_docs = st.file_uploader(
        "Upload your PDFs here and click on 'Process'", accept_multiple_files=False, type='pdf')

    if st.button("Process"):
        with st.spinner("Processing"):
            # get pdf text
            raw_text = get_pdf_text(pdf_docs)

            # get the text chunks
            text_chunks = get_text_chunks(raw_text)

            # create vector store
            vectorstore = get_vectorstore(text_chunks, pdf_docs.name)

        st.write('Upload complete.')
    

if __name__ == '__main__':
    main()
