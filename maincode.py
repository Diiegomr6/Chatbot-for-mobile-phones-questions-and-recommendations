# main.py

# main.py

import streamlit as st
import logging
from langchain_community.vectorstores import SupabaseVectorStore
from langchain_ollama import OllamaLLM
from langchain_community.embeddings import HuggingFaceEmbeddings
import supabase
import pandas as pd
from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings

from intent_classifier import detect_intent
from qa_module import answer_question
#from comparison_module import compare_phones
from recommendation_hybrid_module import hybrid_recommend_phones
#from opinion_module import give_opinion

# Define Variables
SUPABASE_URL = "https://lwfgrbldlcdmvapbqzlr.supabase.co"  
SUPABASE_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6Imx3ZmdyYmxkbGNkbXZhcGJxemxyIiwicm9sZSI6InNlcnZpY2Vfcm9sZSIsImlhdCI6MTczNzcxNTcxNSwiZXhwIjoyMDUzMjkxNzE1fQ.8aOe2yDreNGHMtezz0ji-agfBRI6rsgEP8AiW9fVQvE"

@st.cache_resource
def initialize_all_vector_stores():
    supabase_client = supabase.create_client(SUPABASE_URL, SUPABASE_KEY)

    embeddings_model = OpenAIEmbeddings(model="text-embedding-3-small")  # O "text-embedding-3-large"


    vector_db_qa = SupabaseVectorStore(
        client=supabase_client,
        table_name="qa_ingles",
        query_name="match_documents_qa_ingles",
        embedding=embeddings_model,
    )

    vector_db_recommend = SupabaseVectorStore(
    client=supabase_client,
    table_name="recommendation_ingles",
    query_name="match_documents_recommendation_ingles",
    embedding=embeddings_model,
    )

    db_recommendation = supabase_client.table("recommendation_ingles").select("*").execute().data
    df_recommendation = pd.DataFrame(db_recommendation)


    return vector_db_qa, df_recommendation, vector_db_recommend, embeddings_model

def get_conversation_context(history, max_turns=5):
    history = history[-(max_turns * 2):]  # Últimos 5 turnos
    context_lines = []
    for i in range(0, len(history), 2):
        user_msg = history[i][1]
        bot_msg = history[i + 1][1] if i + 1 < len(history) else ""
        context_lines.append(f"User: {user_msg}\nAssistant: {bot_msg}")
    return "\n".join(context_lines)

vector_db_qa, df_recommendation, vector_db_recommend, embeddings = initialize_all_vector_stores()
llm = ChatOpenAI(
    model="gpt-4o-mini",  
    temperature=0
)
st.title("Mobile Phone Expert")
st.markdown("### Hi, I am your expert in mobile phones! Ask me anything you want.")

if "history" not in st.session_state:
    st.session_state.history = []

user_input = st.text_input("Enter your question:")

if user_input:
    with st.spinner("Thinking..."):
        try:
            intent = detect_intent(user_input, llm)

            conversation_context = get_conversation_context(st.session_state.history)

            if intent == "qa":
                response = answer_question(user_input, vector_db_qa, llm, conversation_context)
            elif intent == "recommendation":
                response = hybrid_recommend_phones(user_input, df_recommendation, vector_db_recommend, llm, conversation_context)
            else:
                response = "I'm sorry, I didn't quite understand your request."

            st.session_state.history.append(("You", user_input))
            st.session_state.history.append(("Assistant", response))

            for speaker, message in reversed(st.session_state.history):
                st.markdown(f"**{speaker}:** {message}")

        except Exception as e:
            st.error(f"An error occurred: {str(e)}")
