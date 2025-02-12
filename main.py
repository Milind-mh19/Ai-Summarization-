import streamlit as st
import validators
from langchain.prompts import PromptTemplate
from langchain_groq import ChatGroq
from langchain.chains.summarize import load_summarize_chain
from langchain_community.document_loaders import YoutubeLoader, UnstructuredURLLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Streamlit App Configuration
st.set_page_config(page_title="Langchain: Summarize Text From YT or Website", page_icon="🦜")
st.title("🦜Langchain: Summarize Text From YT or Website")
st.subheader('Summarize URL')

# Sidebar for API Key Input
with st.sidebar:
    groq_api_key = st.text_input("Groq API Key", value="", type="password")

generic_url = st.text_input("URL", label_visibility="collapsed")

# Initialize Groq Chat Model with a text generation model
llm = ChatGroq(
    model_name="mixtral-8x7b-32768",  # Corrected model for text generation
    groq_api_key=groq_api_key
)

# Prompt Template for Summarization
prompt_template = """
Provide a concise summary of the following content in approximately 300 words. Focus on the main points and key details:
Content: {text}
"""
prompt = PromptTemplate(template=prompt_template, input_variables=["text"])

# Text Splitter to handle large inputs
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=4000,  # Adjust chunk size to fit within token limits
    chunk_overlap=500,  # Overlap to maintain context between chunks
    length_function=len
)

if st.button("Summarize the Content from YT or Website"):
    # Validate Inputs
    if not groq_api_key.strip() or not generic_url.strip():
        st.error("Please provide both a Groq API key and a URL to proceed.")
    elif not validators.url(generic_url):
        st.error("Please enter a valid URL (YouTube or website).")
    else:
        try:
            with st.spinner("Processing..."):
                # Load Content based on URL type
                if "youtube.com" in generic_url:
                    loader = YoutubeLoader.from_youtube_url(
                        generic_url,
                        add_video_info=False  # Only transcript needed for summarization
                    )
                else:
                    loader = UnstructuredURLLoader(
                        urls=[generic_url],
                        ssl_verify=True,
                        headers={"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.114 Safari/537.36"}
                    )
                docs = loader.load()

                # Split the text into smaller chunks
                texts = text_splitter.split_documents(docs)

                # Generate Summary for each chunk
                chain = load_summarize_chain(
                    llm,
                    chain_type="map_reduce",  # Use map_reduce for large documents
                    map_prompt=prompt,
                    combine_prompt=prompt,
                    verbose=False
                )
                output_summary = chain.run(texts)
                
                st.success("Summary Generated Successfully!")
                st.write(output_summary)
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")