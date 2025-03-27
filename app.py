import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from InstructorEmbedding import INSTRUCTOR
from langchain.chat_models import ChatOpenAI
from langchain.embeddings import HuggingFaceInstructEmbeddings
from langchain.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from htmlTemplates import css, user_template, bot_template
from langchain.llms import HuggingFaceHub
import google.generativeai as genai
from langchain_google_genai import ChatGoogleGenerativeAI
import os
from textblob import TextBlob  # for sentiment analysis
import spacy # entity recognition
import speech_recognition as sr # speech recognition through audio to support multimodal



# load .env file
load_dotenv()

# Get Api Token
HUGGINGFACE_API_TOKEN = os.getenv("HUGGINGFACE_API_TOKEN")


# Extract texts from each pdf and each page
def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text() or "" #  "" = adding empty str instead None
    return text


# Split the texts into chunks (mini parts)
def get_text_chunks(text):
    text_splitter = CharacterTextSplitter(separator='\n',
                                          chunk_size=1000,
                                          chunk_overlap=200,
                                          length_function=len)
    chunks = text_splitter.split_text(text)
    return chunks


# After converting perform embeddings on the chunks
def get_vectorStore(text_chunks):
    # embeddings = OpenAIEmbeddings()  # It costs more
    # Instructor is free one
    embeddings = HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-xl", model_kwargs={"device": "cuda"})
    vectorStore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    return vectorStore


# for conversation
def get_conversation_chain(vectorStore):
    # llm = ChatOpenAI()
    llm = HuggingFaceHub(repo_id="google/flan-t5-large",
                         model_kwargs={"temperature": 0.5, "max_length": 512},
                         huggingfacehub_api_token=HUGGINGFACE_API_TOKEN)
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    conversationChain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorStore.as_retriever(),
        memory=memory
    )
    return conversationChain


# Handle user questions
def handle_userInput(user_question):
    response = st.session_state.conversation({"question": user_question})
    st.session_state.chat_history = response["chat_history"]

    for i, message in enumerate(st.session_state.chat_history):
        if i % 2 != 0:  # Bot's answer
            st.write(bot_template.replace("{{MSG}}", message.content), unsafe_allow_html=True)
        else:  # Human's question
            st.write(user_template.replace("{{MSG}}", message.content), unsafe_allow_html=True)



# entity recognition ( to extract address, org , location etc .... from the user question
# Load the English language model

nlp = spacy.load("en_core_web_sm")

# Process the text
text = 'user_question'
doc = nlp(text)

# Print the entities
for ent in doc.ents:
    print(ent.text, ent.label_)




# loading Textblob for sentiment analysis 

def analyze_sentiment(user_question):
    senti=TextBlob(user_question).sentiment.polarity
    doc = senti(user_question)
    
    # Calculating the sentiment scores
    sentiment_score = doc._.sentiment
    
    # sentiment scores
    if sentiment_score > 0.05:
        return 'positive'
    elif sentiment_score < -0.05:
        return 'negative'
    else:
        return 'neutral'

# Testing the func

user_question= 'I like this product!'
print(analyze_sentiment(user_question))  


def main():

    # Initialize the conversation
    if "conversation" not in st.session_state:
        st.session_state.conversation = None

    # Initialize the chat_history
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    # set the page layout and title
    st.set_page_config(page_title="chatPDF", page_icon=":books:")

    st.write(css, unsafe_allow_html=True)

    st.header("Chat with MULTIPLE PDFs :books:")

 # Get input from users
    user_question = st.text_input("Ask a question about your documents")

 # speech recognition 
 
    st.subheader(" Use your Voice :")
    if st.button("Record"):
        r=sr.Recognizer()
        with sr.Microphone() as source:
            st.write("Listening")
            try:
                audio=r.listen(source)
                user_question=r.recognize_google(audio)
                st.write("you said:",user_question)
            except sr.UnknownValueError:
                st.write("sorry i could not recognize your speech ,try again.")
            except sr.RequestError:
                st.write("sorry for the inconvience, try again")

# process use_question (if provided)                
    if user_question:
        handle_userInput(user_question)

    # sentiment analyse
        sentiment=analyze_sentiment(user_question)

    # displaying the sentiment
        st.write(f"Sentiment:{sentiment}")

    # for uploading docs (pdfs)
    with st.sidebar:
        st.subheader("Your Documents :books:")
        pdf_docs = st.file_uploader("Upload your PDFs", accept_multiple_files=True)

        # process the files
        if st.button("Process"):
            with st.spinner("processing"):
                # stage1 - get pdf text
                raw_text = get_pdf_text(pdf_docs)

                # stage2 - get the chunks text
                text_chunks = get_text_chunks(raw_text)

                # stage3 - create the vector store
                vectorStore = get_vectorStore(text_chunks)

                # stage 4 - Create conversation chain
                st.session_state.conversation = get_conversation_chain(vectorStore)

                st.success("PDF processed successfully!")

                # stage 5 -analysing the sentiment of user_question

                


if __name__ == "__main__":
    main()

                

                


