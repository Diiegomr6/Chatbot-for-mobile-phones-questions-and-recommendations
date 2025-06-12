import re
import pandas as pd
import numpy as np
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_openai import OpenAIEmbeddings
import supabase
import pandas as pd
from langchain_ollama import OllamaLLM
from langchain.prompts import PromptTemplate
from langchain.prompts import (
    ChatPromptTemplate,
    FewShotChatMessagePromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
    AIMessagePromptTemplate,
    MessagesPlaceholder,
)
from langchain.schema import HumanMessage, AIMessage

from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

from pandasql import sqldf
import re


def generate_sql_query(question, llm, conversation_context):
    system_message=("""
    You are an AI SQL generator for a single table named phones.  
    You will read the user's request and output exactly one SQL query.

    STRICT MODE RULES:
    - The chat history shows the conversation you had with the client. Use it ONLY and ONLY if the client omits the phone model in the current question but it's clearly inferable from the previous turn or mention something about the past conversation. Otherwise, ignore it. NEVER use it to guess or complete technical details.
    - Return ONLY the raw SQL, WITH NO explanation, no markdown, no backticks.
    - Use ONLY the table "phones" with the schema given below. Do not invent columns or tables. 

    Table “phones” schema (description, allowed values and strict types):
    - model= full name of the phone (text).
    - brand= brand, either "Samsung" or "Apple" (text).
    - releaseyear= the year the phone was released (integer).
    - price= the price in euros (float).
    - platform= the platform, either "Android" or "iOS" (text).
    - battery= capacity of the battery in mAh (integer).
    - camera= camera resolution in MP (integer).
    - screensize= screen size in inches (float).
    - memorysize= memory size in GB, from 256 to 1024 (integer).
    - ram= RAM size in GB, from 4 to 12 (integer).
    - processor= full name of the processor (text).
    - weight= weight in grams (integer).
    - globalscore= integer, global score from 0 to 150.
    - scorequalityprice= quality/price score from 0 to 100 (integer).
    - camerascore= camera score from 0 to 150 (integer).
    - audioscore= audio score from 0 to 150 (integer).
    - displayscore= display score from 0 to 150 (integer).
    - batteryscore= battery score from 0 to 150 (integer). 
    
    Rules:
    1- Always select model, description.
    2- Apply only a `WHERE` when the user gives explicit numeric or text criteria with a boundary or exact match.
    3- UNDER NO CIRCUMSTANCES use a `WHERE` clause with any of the following score columns: batteryscore, displayscore, audioscore, camerascore, globalscore, scorequalityprice. Instead, order by these columns.
    4- If the user uses a qualitative or subjective term for any numeric field without an explicit value or range, NEVER invent or assume bounds. NEVER generate a `WHERE` on that field. Instead, use it to an `ORDER BY`.
    5- When the explicit numeric or text criteria is given, use `WHERE`. Text columns use = 'value' (must match allowed values), and numeric columns use =, <, >, <=, >= or `BETWEEN` for ranges.
    6- Only use one `ORDER BY` clause, by default `ORDER BY globalscore DESC`.
    7- If the user uses a qualitative or subjective term for any field without an explicit value or range, NEVER invent or assume bounds. NEVER generate a `WHERE` on that field. Instead, map it to an `ORDER BY`. For example: if the user says "long/strong/big battery", use `ORDER BY batteryscore DESC`, if the user says "great/excellent/good display", use `ORDER BY displayscore DESC`, if the user says "great/excellent/good camera", use `ORDER BY camerascore DESC`, if the user says "loud/good/excellent audio", use `ORDER BY audioscore DESC`, if the user says "small/huge/big weight", use `ORDER BY weight ASC/DESC`, if the user says "big/huge/good memory size", use `ORDER BY memorysize DESC`.
    8- Explicit sort requests on any other numeric field (price, releaseyear, globalscore, memorysize, ram, battery, camera, screensize, weight, scorequalityprice) map to `ORDER BY that_field [ASC|DESC]`.
    9- If user asks for “top N” or “N best”, add `LIMIT N`. Otherwise add `LIMIT 5`.
    10- There are no NULLs—no `IS NOT NULL` checks needed."""
    )
    examples = [
        {
            "input": "List top 3 Apple phones with the best camera.",
            "query": "SELECT model, description FROM phones WHERE brand = 'Apple' ORDER BY camerascore DESC LIMIT 3;"
        },
        {
            "input": "Show me 5 Android phones with more than 4000 mAh battery.",
            "query": "SELECT model, description FROM phones WHERE battery > 4000 AND platform = 'Android' ORDER BY globalscore DESC LIMIT 5;"
        },
        {
            "input": "What are the cheapest phones from Samsung?",
            "query": "SELECT model, description FROM phones WHERE brand = 'Samsung' ORDER BY price ASC LIMIT 5;"
        },
        {
            "input": "Phones with great display and more than 8GB RAM.",
            "query": "SELECT model, description FROM phones WHERE ram > 8 ORDER BY displayscore DESC LIMIT 5;"
        },
        {
            "input": "List top 5 phones with the best quality/price ratio.",
            "query": "SELECT model, description FROM phones ORDER BY scorequalityprice DESC LIMIT 5;"
        },  
        {
            "input": "Suggest 2 lightweight phones with more than 40 MP in the camera.",
            "query": "SELECT model, description FROM phones WHERE camera > 40 ORDER BY weight ASC LIMIT 2;"
        },
        {
            "input": "List 7 iOS phones ordered by weight",
            "query": "SELECT model, description FROM phones WHERE platform = 'iOS' ORDER BY weight DESC LIMIT 7;"
        },
        {
            "input": "Top 3 phones with best audio.",
            "query": "SELECT model, description FROM phones ORDER BY audioscore DESC LIMIT 3;"
        },
        {
            "input": "Phones with screens larger than 6.7 inches.",
            "query": "SELECT model, description FROM phones WHERE screensize > 6.7 ORDER BY globalscore DESC LIMIT 5;"
        },
        {
            "input": "Give me phones with more than 256GB memory and excellent display.",
            "query": "SELECT model, description FROM phones WHERE memorysize > 256 ORDER BY displayscore DESC LIMIT 5;"
        },
        {
            "input": "List phones released after 2022.",
            "query": "SELECT model, description FROM phones WHERE releaseyear > 2022 ORDER BY globalscore DESC LIMIT 5;"
        },
        {
            "input": "Which are the most expensive phones?",
            "query": "SELECT model, description FROM phones ORDER BY price DESC LIMIT 5;"
        },
        {
            "input": "Top phones with best battery performance.",
            "query": "SELECT model, description FROM phones ORDER BY batteryscore DESC LIMIT 5;"
        },
        {
            "input": "Show me the best iPhone.",
            "query": "SELECT model, description FROM phones WHERE brand = 'Apple' ORDER BY globalscore DESC LIMIT 1;"
        },
        {
            "input": "What are the best phones available?",
            "query": "SELECT model, description FROM phones ORDER BY globalscore DESC LIMIT 5;"
        }
    ]

    few_shot = FewShotChatMessagePromptTemplate(
        example_prompt=ChatPromptTemplate.from_messages([
            HumanMessagePromptTemplate.from_template("User request: {input}"),
            AIMessagePromptTemplate.from_template("{query}")
        ]),
        examples=examples
    )

    prompt = ChatPromptTemplate.from_messages([
        SystemMessagePromptTemplate.from_template(system_message),
        few_shot,
        MessagesPlaceholder(variable_name="chat_history"),
        HumanMessagePromptTemplate.from_template("User request: {question}")
    ])

    if not conversation_context:
        history = []
    else:
        history = []
        for turn in conversation_context:
            if "user" in turn:
                history.append(HumanMessage(content=turn["user"]))
            if "assistant" in turn:
                history.append(AIMessage(content=turn["assistant"]))

    chain = prompt | llm | StrOutputParser()


    return chain.invoke({
        "question": question,
        "chat_history": history
    })



def execute_sql_on_dataframe(sql_query, df):
    try:
        result = sqldf(sql_query, {"phones": df})
        return result
    except Exception as e:
        return pd.DataFrame([{"error": f"SQL execution failed: {e}"}])


def format_docs(docs):
    return "\n\n".join(doc.page_content.replace("\n", " ") for doc in docs)

def detect_phone_model(user_query, known_models):
    """
    Detecta de manera segura si un modelo de móvil conocido está mencionado en la consulta del usuario.
    Prioriza modelos más largos para evitar falsos positivos.
    """
    query_lower = user_query.lower()
    known_models_sorted = sorted(known_models, key=lambda x: -len(x))  # Primero los modelos más largos

    for model in known_models_sorted:
        model_clean = model.lower().strip()

        # Usar regex para buscar coincidencia exacta de palabras, ignorando mayúsculas
        pattern = r'\b' + re.escape(model_clean) + r'\b'  # \b significa "limite de palabra"
        if re.search(pattern, query_lower):
            return model

    return None



# ⚡ Combinar dos embeddings (media simple)
def combine_embeddings(embedding1, embedding2):
    return ((np.array(embedding1) + np.array(embedding2)) / 2).tolist()

# 🚀 Función principal
def hybrid_recommend_phones(user_input, db_recommendation, vectorstore, llm, conversation_context):
    df = pd.DataFrame(db_recommendation)
    embedding_model = OpenAIEmbeddings(model="text-embedding-3-small")  # O "text-embedding-3-large"

    # Paso 1: Generar SQL tradicional
    sql_query = generate_sql_query(user_input, llm, conversation_context)
    print(f"[🧠 SQL generada]:\n{sql_query}\n")
    
    sql_result_df = execute_sql_on_dataframe(sql_query, df)
    if "error" in sql_result_df.columns:
        sql_result_text = "No structured results found."
    else:
        sql_result_text = sql_result_df.to_string(index=False)
    
    sql_result_text = "\n\n".join(
        f"{row['model']}: {row['description'].strip()}"
        for _, row in sql_result_df.iterrows()
    )
    print(f"[📊 Structured context]:\n{sql_result_text}\n")  # <-- Add this line


    # Paso 2: Retrieval semántico avanzado
    retriever = vectorstore.as_retriever(search_kwargs={"k": 10})

    # Detectar si el usuario menciona un modelo concreto
    known_models = df['model'].dropna().unique().tolist()
    rewritten_query = rewrite_question_with_context(user_input, conversation_context, llm)

    detected_model = detect_phone_model(rewritten_query, known_models)

    if detected_model:
        print(f"🔎 Modelo detectado en la consulta: {detected_model}")

        # Obtener el embedding del modelo detectado
        try:
            docs = vectorstore.similarity_search(detected_model, k=1)
            model_doc = docs[0]
            model_embedding = embedding_model.embed_query(model_doc.page_content)
        except Exception as e:
            print(f"❌ Error al recuperar embedding del modelo: {e}")
            model_embedding = None

        if model_embedding:
            # Obtener el embedding de la consulta
            embeddings_model = embedding_model
            query_embedding = embeddings_model.embed_query(user_input)

            # Combinar embeddings
            combined_embedding = combine_embeddings(query_embedding, model_embedding)

            # Retrieval usando el embedding combinado (asumiendo vectorstore soporta búsqueda por embedding)
            semantic_docs = vectorstore.similarity_search_by_vector(combined_embedding, k=10)
        else:
            # Si no conseguimos el embedding del modelo, retrieval normal
            semantic_docs = retriever.invoke(rewritten_query)

    else:
        # No se detectó modelo concreto ➔ retrieval normal
        semantic_docs = retriever.invoke(rewritten_query)

    semantic_context = format_docs(semantic_docs) if semantic_docs else "No semantic results found."
    
    # Paso 3: Preparar el prompt combinado
    system_prompt = """
You are a professional assistant for a mobile phone website, helping users choose the best phones based on their needs.

You have access to two sources of information:
- Structured SQL results: models filtered by explicit technical criteria, given between triple backticks below.
- Semantic results: models retrieved by similarity to the customer's needs, given between triple single quotes below.

Your task:
1. Choose the best mobile phones to recommend based on the user query and both inputs. ONLY recommend smarthphone models given in any of both context, since we are an online shop you only can recommend models available.  
2. ONLY if the client asks for models similar to one given, use the semantic suggestions.
3. If the user does not ask for similar models, ONLY use the structured results. When using structured results, strictly follow the list as provided. Do not add or complete with semantic models, even if there are few structured results. Return exactly the models present in the structured results, keeping the same order.
4. Give a structured answer, using a numbered list and mentioning briefly key characteristics about the models given the user query in maximum 2 sentences. Do not mention every characteristic provided, just 2 or 3 that are relevant for the query or can help the user to choose a mobile. 
5. Use a friendly and professional tone. Be concise.
6. Do not repeat models, just mention one time the ones you choose.
7. Never mention "SQL" or "semantic" or how you retrieved the data.
8. If no models match well, politely say that no recommendations are available.
9. The number of models to return depends on the source: When using structured results, always return exactly the number of models provided in the structured results between triple backticks. When using semantic results, return 5 models by default, unless the user requests a different number. If the user asks for the best, provide only one.

Structured results:
```{structured_context}```

Semantic results:
'''{semantic_context}'''

Conversation history:
[{chat_history}]
"""


    human_prompt = "---{question}---"

    chat_prompt = ChatPromptTemplate.from_messages([
        SystemMessagePromptTemplate.from_template(system_prompt),
        HumanMessagePromptTemplate.from_template(human_prompt)
    ])

    # Paso 4: Construir y ejecutar la chain
    chain = (
        {
            "structured_context": RunnablePassthrough(),
            "semantic_context": RunnablePassthrough(),
            "chat_history": RunnablePassthrough(),
            "question": RunnablePassthrough(),
        }
        | chat_prompt
        | llm
        | StrOutputParser()
    )

    # Paso 5: Lanzar la generación
    response = chain.invoke({
        "structured_context": sql_result_text,
        "semantic_context": semantic_context,
        "chat_history": str(conversation_context) if conversation_context else "",
        "question": user_input,
    })

    return response


def rewrite_question_with_context(question, context, llm):
    prompt = f"""
    You are a rewriting assistant.

    Your task is to transform the following user question into a fully self-contained version by incorporating any relevant missing information from the recent conversation. Do not change the meaning, tone, or intent of the original question. Only add what is necessary for it to make sense on its own.

    Conversation:
    {context}

    User question:
    {question}

    Rewritten standalone question:
    """
    result = llm.invoke(prompt)
    return result.content.strip() if hasattr(result, "content") else str(result).strip()