import streamlit as st
import os
from io import BytesIO
from docx import Document 
import numpy as np
from langchain_community.document_loaders import WebBaseLoader
import pdfplumber 
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS 
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq 
from bs4 import BeautifulSoup
import requests
from dotenv import load_dotenv   
from langchain_text_splitters import RecursiveCharacterTextSplitter

 

load_dotenv() 
groq_api_key = os.getenv("groq_key") 

def process_input(input_type, input_data):
    loader = None
    if input_type == "Link":  
        urls = input_data if isinstance(input_data, list) else [input_data]
        urls = [url for url in urls if url]  
        all_text = ""
        for url in urls: 
            response = requests.get(url) 
            soup = BeautifulSoup(response.content, "html.parser")  
            for tag in soup(["nav", "footer", "script", "style"]): 
                tag.decompose() 
            main = soup.find("main") or soup.find("article") or soup.find("body")
            all_text += main.get_text(separator="\n") if main else soup.get_text(separator="\n")
        documents = all_text 
    elif input_type == "PDF":  
        if input_data is None:  
            raise ValueError("No PDF uploaded")   
        pdf_bytes = input_data.getvalue() 
        text = "" 
        with pdfplumber.open(BytesIO(pdf_bytes)) as pdf: 
            for page in pdf.pages: 
                page_text = page.extract_text() 
                if page_text: 
                    text += page_text + "\n" 
            if not text.strip():   
                raise ValueError("Error: Could not extract text from PDF. File may be image-based or corrupted.") 
        documents = text
    elif input_type == "Text":
        if isinstance(input_data, str):
            documents = input_data  # Input is already a text string
        else:
            raise ValueError("Expected a string for 'Text' input type.")
    elif input_type == "DOCX": 
        if input_data is None: 
            raise ValueError("No DOCX uploaded.")  
        doc_bytes = input_data.getvalue() 
        doc = Document(BytesIO(doc_bytes))
        text = "\n".join([para.text for para in doc.paragraphs])
        documents = text
    else:
        raise ValueError("Unsupported input type")

    text_splitter = CharacterTextSplitter(chunk_size=1500, chunk_overlap=500)
    texts = text_splitter.split_text(documents)  
    
    print("="*50)  
    print(f"TOTAL CHUNKS: {len(texts)}")  
    print("="*50)  
    for i, chunk in enumerate(texts):  
        print(f"\n--- CHUNK {i} ({len(chunk)} chars) ---")  
        print(chunk)  
        print("-"*30)  
        print("="*50)

    model_name = "sentence-transformers/all-MiniLM-L6-v2"
    model_kwargs = {'device': 'cpu'}
    encode_kwargs = {'normalize_embeddings': False}

    hf_embeddings = HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs=model_kwargs,
        encode_kwargs=encode_kwargs
    )

    vector_store = FAISS.from_texts(texts, embedding=hf_embeddings) #langchain
    return vector_store

def answer_question(vectorstore, query):  
    llm = ChatGroq(
        model="llama-3.1-8b-instant",
        api_key=groq_api_key,
        temperature=0.6
    ) 
    system_prompt = (
        "Use the given context to answer the question. "
        "If you don't know the answer, say you don't know. "
        "Keep the answer concise. "
        "Context: {context}"
    )
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{input}"),
    ])
    
    question_answer_chain = create_stuff_documents_chain(llm, prompt)
    chain = create_retrieval_chain(vectorstore.as_retriever(search_kwargs={"k": 10}), question_answer_chain) 
    try: 
        result = chain.invoke({"input": query})
        return result["answer"]
    except Exception as e:
        # Check if its a token limit error
        if "rate_limit_exceeded" in str(e) or "Request too large" in str(e):
            # Extract token numbers if possible
            error_msg = str(e)
            if "Requested" in error_msg:
                # Try to extract the requested token count
                try:
                    requested = error_msg.split("Requested ")[1].split(",")[0]
                    return f"Error: Document too large. Exceeded token limit (requested {requested} tokens). Please try a shorter document or ask a more specific question."
                except:
                    return "Error: Document too large. Exceeded token limit. Please try a shorter document or ask a more specific question."
            else:
                return "Error: Document too large. Exceeded token limit. Please try a shorter document."
        else:
            # Other errors
            return f"Error: {str(e)}"

def main():  
    st.title('Askora') 
    input_type = st.selectbox("Input Type: ", ["Link", "PDF", "Text", "DOCX"])   
    if input_type == "Link":     
        number_input = st.number_input(min_value=1, max_value=10, step=1,label="Enter the number of Links") 
        input_data = []
        for i in range(number_input):
            url = st.sidebar.text_input(f"URL {i+1}")
            input_data.append(url)  
    
    if input_type == "Text": 
        input_data = st.text_input("Enter the text") 
        
    if input_type == "PDF": 
        input_data = st.file_uploader("Upload a PDF file", type=["pdf"]) 
    
    if input_type == "DOCX": 
        input_data = st.file_uploader("Upload a DOCX file", type=["docx", "doc"])  
    clicked = st.button("Continue")  #renders and return boolean
    if (clicked):     
        vectorstore = process_input(input_type, input_data)
        st.session_state["vectorstore"] = vectorstore    #memery box for a session 
        
    if "vectorstore" in st.session_state:
        query = st.text_input("Ask your question")
        if st.button("Submit"): 
            answer = answer_question(st.session_state["vectorstore"], query) 
            st.write(answer)
    
if __name__ == '__main__': 
    main()   
    
    
    
    
    