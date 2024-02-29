# step1 libs
import pandas as pd
import sqlite3
import pandas as pd

# step2 and step3 libs
from langchain_community.document_loaders import DataFrameLoader
from langchain_community.vectorstores import Chroma
from langchain.embeddings.sentence_transformer import SentenceTransformerEmbeddings

# step4 libs
from langchain_community.llms import Ollama
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
# from langchain_core.prompts import PromptTemplate
from langchain_mistralai.chat_models import ChatMistralAI


def prepare_df(file):

    # Create raw dataframe
    con = sqlite3.connect(file)
    cur = con.cursor()
    cur.execute("SELECT * FROM assistance")
    df = pd.read_sql_query("SELECT * FROM assistance", con)
    con.close()

    df = df.drop_duplicates(subset=['title'])  # 131 duplicates

    # Content lenght + title lenght
    df['total_size'] = df.apply(lambda row: len(
        row['url']) + len(row['title']) + len(row['content']), axis=1)

    # Concatenation of title and content
    df['full_content'] = df.apply(lambda row: 'Titre de la page : "' +
                                  row['title'] + '". Contenu de la page : "' + row['content']+'"', axis=1)
    df.drop(['title', 'content'], axis=1)

    return df


# STEP 2 : Create vectorized data (chromadb)

# recreate=True to fully recreate the db
def load_vectorized_data(df, model="all-MiniLM-L12-v2", recreate=False):
    """Load the vectorized data from the disk if it exists, otherwise create it from the dataframe and save it on disk."""
    embedding_function = SentenceTransformerEmbeddings(
        model_name=model)  # check other models here : https://www.sbert.net/docs/pretrained_models.html
    if recreate:
        df_loader = DataFrameLoader(df, page_content_column="full_content")
        df_document = df_loader.load()

        # Create the vectorized data
        print("\nCreating the vectorized data, can take 1 minute...\n")
        db = Chroma.from_documents(df_document, embedding_function)

        # Save it on disk
        Chroma.from_documents(
            df_document, embedding_function, persist_directory="./data/vectorized_db")
        return db
    else:  # if no need to recreate the db, just load it from disk
        return Chroma(persist_directory="./data/vectorized_db",
                      embedding_function=embedding_function)


# STEP 3 : Search simalirar documents
def search_similar_documents(db, query, k=2):
    """Search for similar documents in the vectorized data. Only return the k most similar documents, with a score above 1.1."""
    similar_documents = db.similarity_search_with_score(
        query, 2)
    print("\nRAW similar documents : \n", similar_documents)
    output = []
    for i, doc in enumerate(similar_documents):
        if doc[1] > 1.2:
            output.append("Vide")
        else:
            output.append(doc[0].metadata)
    # print(output)
    return output


# STEP 4 : Chatting with the LLM
def query(question, similar_documents, previous_doc, conversation, model='mistral-openorca:7b-q5_K_S', gpu=False, API_KEY=""):
    """Query the LLM with the question and the similar documents, and return the response. If the conversation has already started, also provide the previous document and the conversation history."""
    # Load the LLM locally
    if API_KEY == "":
        if gpu:
            llm = Ollama(model=model,
                         callback_manager=CallbackManager([StreamingStdOutCallbackHandler()]))
        else:
            llm = Ollama(model=model,
                         callback_manager=CallbackManager([StreamingStdOutCallbackHandler()]))
    else:  # load LLM with Mistral API (online version)
        llm = ChatMistralAI(
            model=model, mistral_api_key=API_KEY)

    # define the template for the first prompt of the discussion
    init_prompt_template = """
    En tant que ChatBot du service client de Free, ton objectif est de fournir des réponses structurées, factuelles, utiles et concises aux questions des clients, tout en maintenant une interaction humaine et profesionnelle. Tu dois répondre en Markdown, uniquement en français. Pour les questions spécifiques nécessitant une recherche dans la base de données, utilise les informations extraites des documents du service client. Si la réponse à la question n'est pas disponible dans ta base de données, indique clairement que tu ne sais pas, sans inventer de réponse. Pour les salutations ou questions générales, offre une réponse cordiale et engageante. Après avoir répondu à une question spécifique, tu peux recommender une URL pertinente. Si ta base de données est vide, c'est que la question est d'ordre général et tu peux répondre de manière générale.

    Si la question est une salutation ou une interaction générale (comme "Bonjour"), réponds de manière chaleureuse et engageante avant de poursuivre avec des informations spécifiques si nécessaire.

    Question du client : "{question}"

    Base de données pour cette question : "{docs}"
                                              
    Réponse :
    """
    # define the prompt when the discussion has already started
    base_prompt_template = """A partir de maintenant, exprime toi uniquement en Francais.                                              
    Tu es toujours le ChatBot du service client de Free, tu dois continuer la discussion entamée avec le client ton objectif est toujours de fournir des réponses structurées, factuelles, utiles et concises aux questions des clients, tout en maintenant une interaction humaine et profesionnelle. Tu dois répondre en Markdown, uniquement en français. Pour les questions spécifiques nécessitant une recherche dans la base de données, utilise les informations extraites des documents du service client ou de l'historique de la conversation. Si la réponse à la question n'est pas disponible dans ta base de données, indique clairement que tu ne sais pas, sans inventer de réponse. Pour les salutations ou questions générales, offre une réponse cordiale. Après avoir répondu à une question spécifique, tu peux recommender une URL pertinente. Si ta base de données est vide, c'est que la question est d'ordre général et tu peux répondre de manière générale. Inutile de dire Bonjour et Bonne journée à chaque fois.
    
    Question du client : "{question}"

    Base de données pour cette question : "{docs}"
                                              
    Base de données pour la question précédente : "{previous_doc}"

    Historique de la conversation : "{conversation}"

    Réponse :
    """

    # Create the prompt from a defined template (different if it's the first question or not and if the model in local or API)

    if API_KEY != "":  # local LLM
        if conversation == "":
            prompt = PromptTemplate.from_template(init_prompt_template)
            prompt = prompt.format(
                question=question, docs=similar_documents)
        else:
            prompt = PromptTemplate.from_template(base_prompt_template)
            prompt = prompt.format(question=question,
                                   docs=similar_documents, previous_doc=previous_doc, conversation=conversation)
    else:  # Mistral API
        if conversation == "":
            prompt = PromptTemplate(
                input_variables=["question", "docs"], template=init_prompt_template)
        else:
            prompt = PromptTemplate(
                input_variables=["question", "docs", "previous_doc", "conversation"], template=base_prompt_template)

    print("\n\n\n############## SETTINGS ##############\n")
    print("--Question : ", question)
    print("--Model llm : ", model)
    print("--GPU : ", gpu)
    print("--API_KEY : ", API_KEY)
    print("\n--Full prompt : \n", prompt)
    print("\n--Docs : \n", similar_documents)
    print("\n--Doc history : \n", previous_doc)
    print("\n--Conversation : \n", conversation)
    print("\nStart generating the response...\n")

    # Generate the response : 1 case for local LLM and 1 case for Mistral API
    if API_KEY == "":
        chain = LLMChain(llm=llm, prompt=prompt)
        if conversation == "":
            response = chain.run(question=question, docs=similar_documents)
        else:
            response = chain.run(
                question=question, docs=similar_documents, previous_doc=previous_doc, conversation=conversation)
    else:
        response = llm.invoke(prompt)

    print("Response : \n", response)
    return response


def flatten_conversation(messages):
    """Flatten the conversation from document type to string."""
    # Convert the messages to strings
    messages_str = [str(message) for message in messages]

    # Join the messages into a single string
    messages_combined = "\n".join(messages_str)
    # print("Conversation : \n", messages_combined)
    return messages_combined


def main(question, conversation='', previous_doc=["Vide"], llm='mistral-openorca:7b-q5_K_M', gpu=False, API_KEY=""):
    df = prepare_df("data/assistance.sqlite3")

    db = load_vectorized_data(df, recreate=False)

    docs = search_similar_documents(
        db, question)
    response = query(question, docs, previous_doc, flatten_conversation(conversation),
                     model=llm, gpu=gpu, API_KEY=API_KEY)
    return response, docs[0]


if __name__ == "__main__":
    print(main("Comment obtenir le numéro RIO de ma ligne mobile ?")[0])
