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
pip install transformers 
from transformers import pipeline


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

# context-aware emotional intelligence
# load the sentiment(emotion) detector model
# for analysing variable = emotion is used
# for genertaing response variable = emotionsb is used
# model="j-hartmann/emotion-english-distilroberta-base" = for english language #

sentiment_analyser=pipeline("text-classification",model="j-hartmann/emotion-english-distilroberta-base", return_all_scores=True)
def analyse_sentiment(user_question):
    try:
        emotion=sentiment_analyser(user_question)[0] # getting the emotions scores
        emotion_highscore=max(emotion,key=lambda x:x['score'])
        return emotion_highscore['label'],emotion_highscore['score']
    except Exception as e:
        # changed to neutral if any error comes
        return "neutral",0.0



# Handle user questions
def handle_userInput(user_question):
    emotions,emotions_score=analyse_sentiment(user_question)

# adding the extracted emotions to session state(conersation history)

    st.session_state.emotions.append({
        "Question":user_question,
        "Emotion":emotions,
        "Emotion_score":emotions_score
    })
    
    

# getting response fromn conversion chain

    response = st.session_state.conversation({"question": user_question})
    st.session_state.chat_history = response["chat_history"]

    for i, message in enumerate(st.session_state.chat_history):
        if i % 2 != 0:  # Bot's answer
            st.write(bot_template.replace("{{MSG}}", message.content), unsafe_allow_html=True)
        else:  # Human's question
            emotion_data=st.session_state.emotions[i//2]
            st.write(user_template.replace("{{MSG}}", message.content), unsafe_allow_html=True)

# providing response for the emotions obtain from user questions

def generate_contextual_response(user_question):
    emotions,_=analyse_sentiment(user_question)
    response=st.session_state.conversation({"question":user_question})['answer']

# aading emotions to response

    emotion_responses = {
        "anger": "I'm sorry to hear that you're upset. Let me try to help: ",
        "joy": "That's great! Here's some more information: ",
        "sadness": "I'm here to help. Let me assist you: ",
        "fear": "Don't worry, I'm here to guide you: ",
        "neutral": "Here's the information you asked for: ",
        "disgust": "Sorry to hear that. Let's sort it out: ",
        "surprise": "Wow, that's interesting! ",
    }
    return emotion_responses.get(emotions, "I'm not sure how to respond to that: ") + response

# Entity recognition ( to extract address, org , location etc .... from the user question
# Load the English language model thorugh spacy

entity_model=spacy.load("en_core_web_sm")
def extract_entites(user_question):
    doc=entity_model(user_question)
    return[(ent.text,ent.label_) for ent in doc.ents]


# loading Textblob for sentiment analysis 

def analyze_sentiment(user_question):
    sentiment=TextBlob(user_question).sentiment.polarity
    
 # sentiment scores
    if sentiment > 0.05:
        return 'positive'
    elif sentiment < -0.05:
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

    # Initialize the chat_history and emotions
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = st.session_state.get("chat_history",[])
    if "emotions" not in st.session_state:
        st.session_state.emotions=st.session_state.get("emotions",[])

    # set the page layout and title
    st.set_page_config(page_title="chatPDF", page_icon=":books:")

    st.write(css, unsafe_allow_html=True)

    st.header("Chat with MULTIPLE PDFs :books:")

 # Get input from users
    user_question = st.text_input("Ask a question about your documents")

# auto compleletion of users question
    if user_question:
        if st.button("Auto complete"):
            model=pipeline("text-generation",model="gpt2")
            completion=model(user_question,max_length=50)[0]['generated_text']
            st.write("Auto completion:",completion)

# [0] is used to acces the first dict in list
            
# speech recognition 
 
    st.subheader("record your voice")
    if st.button("Record"):
        r=sr.Recognizer()
        with sr.Microphone() as source:
            st.write("Listening......")
            try:
                audio=r.listen(source)
                user_question=r.recognize_google(audio)
                st.write("you said:",user_question)
            except sr.UnknownValueError:
                st.write("sorry i could not recognize your speech ,try again.")
                user_question=""
            except sr.RequestError:
                st.write("sorry for the inconvience, try again")
                user_question=""

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

                

                


