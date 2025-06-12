from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

def format_docs(docs):
    return "\n\n".join(doc.page_content.replace("\n", " ") for doc in docs)

def create_retriever(vector_db):
    return vector_db.as_retriever(search_kwargs={"k": 10})

def answer_question(query, vector_db, llm, chat_context):
    
    rewritten_query = rewrite_question_with_context(query, chat_context, llm)

    retriever = create_retriever(vector_db)

    prompt = ChatPromptTemplate([
        ("system", """
You are a helpful chatbot that answers customer questions on a mobile phone website.\
These questions are about different mobile phone models, their characteristics, and technical specifications.\

Please answer the following question using only the information provided in the context (shown between triple backticks). 
Also, the conversation history is given (shown between squared brackets).\
Keep your response clear. Respond ONLY on the mobile phone model mentioned in the question, and include all asked details from it.\

Before answering, follow these reasoning steps:
1. Identify which model the question asks for.\
2. If no specific model is mentioned but a general category or brand, do not attempt to answer. Instead, JUST politely ask the 
customer to clarify which specific model they want to know more about.\
3. If a specific phone model is mentioned in the question, check if it is mentioned in the given context.\
4. If the model is not in the context, apologize and say that the model is not available — nothing else.\
5. If a model is included but the specific information requested is missing, apologize and say that we don't have that information.\
6. If everything is available, answer the question directly, in a natural tone and understandable way — do not mention that the 
answer comes from the context.\
7. If the user asks for general information, briefly mention the most important characteristics.
8. Please preferably do not use bullet points or lists filled with technical terms. Instead, explain the characteristics in 
clear, natural language that someone without technical knowledge can understand.\
              
Do not ever mention the words "context," "source," "provided information," "chat history" or anything similar, neither from 
where you got the knowledge. \
Do not mention that you have information on the given question, nor any introduction to the answer. If you have it, just 
answer the question.\

You may also refer to the recent conversation history ONLY IF the current question omits the phone model but clearly follows up on 
a previous one.\
Never guess or infer technical details from the conversation history. Use the history strictly to identify the phone model or any 
detail being discussed, when it is missing or referenced in the question.\
         
Context:
```{context}```

Conversation history:
[{chat_history}]
"""),
        ("human", "---{question}---")
    ])

    docs = retriever.invoke(rewritten_query)
    formatted_context = format_docs(docs)

    chain = (
        {
            "context": RunnablePassthrough(),
            "chat_history": RunnablePassthrough(),
            "question": RunnablePassthrough(),
        }
        | prompt
        | llm
        | StrOutputParser()
    )

    return chain.invoke({
        "context": formatted_context,
        "chat_history": str(chat_context),
        "question": query,
    })


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