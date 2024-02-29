import streamlit as st
from streamlit_chat import message
import langchain_helper as lch
from langchain.schema import (SystemMessage, HumanMessage, AIMessage, messages)


def main():
    st.set_page_config(
        page_title="Iliad technical assessment",
        page_icon="🤖",
    )
    st.header("ChatBot Free Assistance")
    st.write("by [Julien GODFROY](https://github.com/jugodfroy)", )

    if "messages" not in st.session_state:
        st.session_state.messages = [
            # SystemMessage(content="En tant que ChatBot du service client de FREE, ton objectif est de fournir des réponses structurée, factuelles, utiles et concises aux questions des clients. Tu dois répondre en Markdown, uniquement en Français. Utilise les informations extraites des documents du service client pour répondre. Si la réponse à la question n'est pas disponible dans ta base de données, indique clairement que tu ne sais pas, sans inventer de réponse. Après avoir répondu, recommande une ou plusieurs URL pertinentes parmi celles fournies."),
        ]

    ##########################################
    #                SIDEBAR                 #
    ##########################################

    with st.sidebar:
        img = st.image("img/Logo_iliad.png", width=50)
        title = st.title("Iliad technical assessment")
        mistral = st.selectbox(
            "Utiliser l'API Mistral (online) ? :", ['No, run locally', 'Yes (key needed)'])
        with st.form("my_form"):
            if mistral == 'No, run locally':
                llm = st.selectbox("Choisissez un LLM offline :", [
                    "vigostral", "mistral-openorca:7b-q5_K_S", "mistral-openorca:7b-q5_K_M", "gemma", "mistral:instruct", "mistral:7b-instruct-q5_K_M", "mixtral"])
                st.write(
                    "Make sur the selected model is installed : ollama pull <modelname>")
                gpu = st.checkbox("Utiliser le GPU (CUDA) (pas testé)", False)
            else:
                llm = st.selectbox("Choisissez un LLM online:", [
                    "open-mistral-7b", "open-mixtral-8x7b"])
                API_KEY = st.text_input(
                    "Entrez votre clé API Mistral :", type="password")
            user_input = st.text_area(
                "Posez votre question ici :",
                max_chars=150,
                help="Keep your question clear and concise for the best results.",
                placeholder="Comment obtenir le numéro RIO de ma ligne mobile ?"
            )
            submit_btn = st.form_submit_button("Envoyer")

        reset_btn = st.button("Reset press 2 times")

    ##########################################
    #                MAIN CORE               #
    ##########################################
    previous_doc = []
    message("Bonjour, je suis l'agent conversationnel de Free. Comment puis-je vous aider ?", is_user=False)

    # If the user has submitted a question
    if submit_btn and user_input != "":
        with st.spinner("Je réflechis..."):
            if mistral == 'No, run locally':    # run with local LLM
                response, doc = lch.main(
                    user_input, st.session_state.messages, previous_doc, llm, gpu)
            else:
                response, doc = lch.main(       # run with Mistral API
                    user_input, st.session_state.messages, previous_doc=previous_doc, llm=llm, API_KEY=API_KEY)
            st.session_state.messages.append(HumanMessage(content=user_input))

            # to deal with different response types depending on the type of LLM (local, or api)
            if mistral == 'No, run locally':
                st.session_state.messages.append(
                    AIMessage(content=response))
            else:
                st.session_state.messages.append(
                    AIMessage(content=response.content))
        previous_doc = doc  # keep track of the previous doc for the next query

    # Refresh the chat area
    messages = st.session_state.get('messages', [])
    for i, msg in enumerate(messages):
        if i % 2 == 0:  # user msg
            message(msg.content, is_user=True, key="user_"+str(i))
        else:        # bot msg
            message(msg.content, is_user=False, key="bot_"+str(i))

    if reset_btn:
        st.session_state.messages.clear()
        previous_doc = []
        user_input = ""


if __name__ == "__main__":
    main()
