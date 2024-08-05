from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, ServiceContext
from llama_index.llms.huggingface import HuggingFaceLLM
from llama_index.core.prompts.prompts import SimpleInputPrompt
import os
import streamlit as st

HF_TOKEN = os.getenv("HF_TOKEN")
if not HF_TOKEN:
    st.error("Hugging Face token not found. Please set the HF_TOKEN environment variable.")
else:
    st.write("Hugging Face token loaded successfully.")
documents = SimpleDirectoryReader("./data").load_data()
system_prompt = """
You are a Q&A assistant. Your goal is to answer questions as accurately as
possible based on the instructions and context provided.
"""
query_wrapper_prompt = SimpleInputPrompt("<|USER|>{query_str}<|ASSISTANT|>")

import torch

llm = HuggingFaceLLM(
    context_window=4096,
    max_new_tokens=256,
    generate_kwargs={"temperature": 0.0, "do_sample": False},
    system_prompt=system_prompt,
    query_wrapper_prompt=query_wrapper_prompt,
    tokenizer_name="meta-llama/Llama-2-7b-chat-hf",
    model_name="meta-llama/Llama-2-7b-chat-hf",
    tokenizer_kwargs={"token": HF_TOKEN},
    model_kwargs={"token": HF_TOKEN, "torch_dtype": torch.float16, "load_in_8bit":True},
    device_map="auto",
    # uncomment this if using CUDA to reduce memory usage

)

from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from llama_index.core import ServiceContext
from llama_index.embeddings.langchain import LangchainEmbedding

embed_model = LangchainEmbedding(
  HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
)

service_context = ServiceContext.from_defaults(
    chunk_size=1024,
    llm=llm,
    embed_model=embed_model
)

index = VectorStoreIndex.from_documents(documents, service_context=service_context)
query_engine = index.as_query_engine()

# response = query_engine.query("tell me different types of synapses ")
# print(response)

# Streamlit app
st.title("RAG (Retrieval-Augmented Generation) Application")

# User input
user_input = st.text_area("Enter your query:", "")

if st.button("Generate Response"):
    if user_input:
        with st.spinner("Generating response..."):
            try:
                st.write("Sending query to the engine...")
                response = query_engine.query(user_input)  # Ensure the correct method is used
                st.write("Response received.")
                st.success(response)
            except Exception as e:
                st.error(f"Error generating response: {e}")
    else:
        st.warning("Please enter a query to generate a response.")
