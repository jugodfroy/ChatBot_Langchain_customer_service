# Project: Free Customer Service Chatbot

## Overview

This project develops a PoC of a customer service chatbot specifically designed for the telecommunications operator Free. It leverages a dataset consisting of articles from the "customer service" section of Free's website, providing the knowledge base for answering user queries. The chatbot utilizes technologies such as LangChain (python) for natural language processing, ChromaDB for vectorization of documents, and the choice between local open-source models (via Ollama) and online open-source models available through the Mistral API for generating responses. This chatbot is its ability to conduct complete conversations with users, keeping track of the conversation history to provide context-aware responses.

## Features

- **SQLite Database Integration:** Uses an SQLite database to store customer service articles, enabling efficient data retrieval.
- **ChromaDB Vectorization:** Employs ChromaDB for vectorizing service documents, facilitating the efficient search for documents similar to the user's query.
- **Flexible LLM Usage:** Offers the flexibility to use local open-source models with Ollama or access a variety of models through the Mistral API for response generation.
- **Conversation History:** Capable of handling complete discussions, maintaining a history of the conversation to provide contextually relevant responses.
- **Streamlit Web Interface:** Features a user-friendly web interface built with Streamlit, allowing easy interaction with the chatbot.

## How It Works

### Data Preparation

- **Function:** `prepare_df(file)`
- **Purpose:** Processes the SQLite database containing customer service articles, preparing the data for vectorization and retrieval.
- **Details:** Removes duplicates, calculates content length, and combines 'title' and 'content' for a comprehensive representation of each article.

### Vectorized Data Creation

- **Function:** `load_vectorized_data(df, model="all-MiniLM-L12-v2", recreate=False)`
- **Purpose:** Vectorizes the prepared data using ChromaDB and the specified embedding model to facilitate similarity searches.
- **Details:** Can either generate new vectorized data or load previously created vectorized data from disk.

### Similarity Search

- **Function:** `search_similar_documents(db, query, k=2)`
- **Purpose:** Searches for customer service articles similar to the user's query within the vectorized data, aiming to find the most relevant articles to base the response on. If the highest similarity is below a certain threshold, the article is not returned.
- **Details:** Utilizes the vectorized data to find top-k similar documents, enhancing the chatbot's ability to provide accurate information.

### Chatbot Interaction

- **Functions:** Include `query(question, similar_documents, previous_doc, conversation, model='mistral-openorca:7b-q5_K_S', gpu=False, API_KEY="")` and `main(question, conversation='', previous_doc=["Vide"], llm='mistral-openorca:7b-q5_K_M', gpu=False, API_KEY="")`
- **Purpose:** Manages the interaction process, generating responses to user queries based on similar documents found and maintaining a conversation history.
- **Flexibility:** Supports using local open-source models or online models via the Mistral API, catering to different operational environments and requirements.

### Web Interface

- **Implementation:** The web interface, developed with Streamlit, offers a straightforward way for users to interact with the chatbot, including features like LLM selection, API key input for Mistral API, and conversation history management.

## Installation

1. Clone the project repository in a folder and enter the project directory.
2. Install the necessary dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. (Optional) If yo plan to use local Ollama models, make sure you download Ollama [here](https://ollama.com/download) and pull the wanted models (gemma, mistral, mistral-openorca, etc.) using the following command:
   ```bash
   ollama pull <model_name>
   ```
4. Launch the Streamlit application:
   ```bash
   streamlit run main.py
   ```

## Configuration Options (in the UI)

- **LLM Selection:** Choose between leveraging local open-source models with Ollama or using the Mistral API for online models, depending on availability and performance preferences.
- **GPU Acceleration:** Option to utilize GPU acceleration for local model inference, enhancing performance where applicable.
  ![Online_ctrl_panel](https://github.com/jugodfroy/ChatBot_Langchain_customer_service/assets/79590825/47e00261-b474-4ad4-804d-3e3631c52b17)![Offline_ctrl_pannel](https://github.com/jugodfroy/ChatBot_Langchain_customer_service/assets/79590825/969b1908-9cde-4f23-a7b5-aaaf22232156)

## Usage

Upon starting the Streamlit application and navigating to the provided URL, users can engage with the chatbot by typing their queries into the text area and clicking "Send". The chatbot, drawing on its database of customer service articles, generates responses that are both relevant and contextually aware, thanks to its underlying technologies and conversation history feature.

## Conclusion

The Free Customer Service Chatbot showcases the integration of state-of-the-art technologies in natural language processing and document vectorization to create a sophisticated tool capable of supporting comprehensive customer service interactions. Its ability to conduct full conversations with historical context makes it a valuable asset for improving customer experience and efficiency in handling inquiries.

## USEFUL LINKS

- [Langchain documentation](https://python.langchain.com/docs)
- [MistralAPI documentation](https://docs.mistral.ai/)
- [Chromadb documentation](https://docs.trychroma.com/)
- [Ollama available local LLMs](https://ollama.com/library)
- [Streamlit documentation](https://docs.streamlit.io/)
