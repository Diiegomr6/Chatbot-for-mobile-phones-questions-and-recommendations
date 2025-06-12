# Chatbot for Mobile Phone Questions and Recommendations

This project implements a modular conversational assistant built in Python. It is a chatbot capable of answering technical questions about mobile phones and giving recommendation. This chatbot was developed for my Bachelor Tesis “Development of a chatbot based on Large Language Models and Retrieval Augmented Generation” for Universidad Carlos III de Madrid.

## Technologies Used

- **Python**: Core programming language used to build all modules.
- **OpenAI GPT models**: Used for natural language understanding, intent classification, and answer generation.
- **Supabase**: Acts as the backend database for storing user data, content, and embeddings. Also used for vector similarity search.
- **Streamlit**: Provides a lightweight web interface for user interaction with the chatbot.

## System Functionality

The chatbot works by processing each user message in the following pipeline:

1. **Intent Detection**: The input message is sent to OpenAI's API to determine the user's intent (e.g., ask a question, request a recommendation).
2. **Routing**: Based on the intent, the message is routed to either:
   - the question answering module, or
   - the recommendation engine.
3. **Question Answering**: If a question is detected:
   - The system performs a similarity search on Supabase using vector embeddings to retrieve relevant context.
   - The relevant information and original question are sent to the LLM for answer generation.
4. **Recommendation**: If a recommendation is requested:
   - The system retrieves user and item data from Supabase.
   - A hybrid recommendation is generated using content-based and collaborative signals.
5. **Response Generation**: The final output is displayed in the Streamlit interface.

## File Descriptions

- `maincode.py`: The main script that ties all components together. It handles the user interface (via Streamlit), receives input, routes it to the appropriate module, and displays the output.
  
- `intent_classifier.py`: This module takes a user message and sends it to OpenAI to classify the intent. It returns a tag or label used to determine which module should handle the input.

- `qa_module.py`: Handles question answering. It queries Supabase for related documents using vector search, composes a prompt with the retrieved context, and calls OpenAI to generate a final answer.

- `recommendation_hybrid_module.py`: Generates personalized recommendations by combining content-based filtering (matching user interests with item features) and collaborative filtering (based on similar user behavior). It interacts with Supabase to fetch and process user and item data.

