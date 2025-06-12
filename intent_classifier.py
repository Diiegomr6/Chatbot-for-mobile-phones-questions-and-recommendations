# intent_classifier.py

from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

# 🔹 Prompt para clasificar la intención del usuario
intent_prompt = PromptTemplate.from_template("""
You are an intent classifier for a chatbot that answers user questions about mobile phones.

Your task is to classify a user question into one of the following categories:

- **qa** → Factual, technical questions about phone specifications. These questions typically mention a specific phone model and inquire about objective, measurable details (e.g., battery capacity, screen size, RAM, etc.).

- **recommendation** → Questions where the user seeks suggestions or advice based on certain criteria. These questions generally do **not** mention specific phone models, but instead ask for one or more suggestions that meet objective or subjective requirements (e.g., best phone under 500€, phone with good battery life, etc.).

Classify based on the **intention**, not just keywords.
Only respond with one of these labels: qa, recommendation.

--- Examples ---

"How much RAM does the Galaxy S24 have?" → qa  
"Which phone is better for gaming?" → recommendation  
"Which phone has the best camera for recording videos?" → recommendation  
"How big is the battery on the iPhone 13 Mini?" → qa  

--- Now classify this ---

User question: {query}
Intent:
""")

def detect_intent(query, llm):
    """Clasifica la intención del usuario usando LLM."""
    chain = intent_prompt | llm | StrOutputParser()
    intent = chain.invoke({"query": query}).strip().lower()
    return intent
