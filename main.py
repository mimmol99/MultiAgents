
import os
from autogen import AssistantAgent, UserProxyAgent
from autogen.coding import DockerCommandLineCodeExecutor
from pathlib import Path
from dotenv import load_dotenv
from qdrant_client import QdrantClient
import autogen
from autogen.agentchat.contrib.qdrant_retrieve_user_proxy_agent import QdrantRetrieveUserProxyAgent,RetrieveUserProxyAgent
from autogen.agentchat.contrib.retrieve_assistant_agent import AssistantAgent,RetrieveAssistantAgent
from qdrant_client import QdrantClient
from langchain_qdrant import Qdrant 
from loading import Loader
from langchain_experimental.text_splitter import SemanticChunker
from langchain_openai import OpenAIEmbeddings

#from autogen.agentchat.contrib.self_evaluation_agent import SelfEvaluationAgent

# Accepted file formats for that can be stored in
# a vector database instance
from autogen.retrieve_utils import TEXT_FORMATS

load_dotenv(Path("../api_key.env"))

def main():
      
    config_list=[{
        "model": "gpt-3.5-turbo",
        "api_key": os.environ.get("OPENAI_API_KEY"),
    }]

    llm_config={
            "timeout": 600,
            "cache_seed": 42,
            "config_list": config_list,
        }
    



    assistant = RetrieveAssistantAgent(
        name="assistente",
        system_message="Sei un assistente di una compagnia di assicurazione.rispondi alle domande in base al contesto passato",
        llm_config=llm_config
    )

    EVAL_CUSTOM_PROMPT =  """Sei un valutatore di qualità della risposta di un chatbot per una compagnia di assicurazioni.
    In base alla domanda ed al contesto genera una label tra le seguenti: [CORRETTO,ERRATO].

    La domanda dell'utente è: {input_question}.

    Il contesto è: {input_context}.

    La risposta è: {input_answer}.
    """

    llm_eval_config={
            "timeout": 600,
            "cache_seed": 42,
            "config_list": config_list,
            "max_consecutive_auto_reply":0
            #"customized_prompt":EVAL_CUSTOM_PROMPT
        }
    
    # Set up the self-evaluation agent
    evaluator = UserProxyAgent(
        name="valutatore",
        human_input_mode="NEVER",
        system_message="Devi valutare le risposte dell'assistente.",
        llm_config=llm_eval_config
    )

    evaluator = UserProxyAgent(
        name="valutatore",
        human_input_mode="NEVER",
        #system_message=" ",
        llm_config=llm_eval_config
    )

    assistant_evaluator = RetrieveAssistantAgent(
        name = "assistant_evaluator",
        human_input_mode="NEVER",
        system_message="Devi valutare le risposte dell'assistente.",
        llm_config=llm_eval_config
    )
     
    files_path = "/home/utente/Desktop/Projects/CHATBOT_YOLO/ALL_FILES/COMPANY/SUB"
    
    #loader = Loader(files_path)
    #docs = loader.load_documents()

    #embedding_function = OpenAIEmbeddings()
    #base_splitter = SemanticChunker(embedding_function)
    #chunks = base_splitter.split_documents(docs)
    #vectorstore = Qdrant.from_documents(documents=chunks, embedding=embedding_function ,location=":memory:" )
    
    #client = QdrantClient(":memory:") 
    #client.add(collection_name="coll",documents=[doc.page_content for doc in docs])

    PROMPT_QA = """You're a retrieve augmented chatbot. You answer user's questions based on your own knowledge and the
    context provided by the user.
    If you can't answer the question with or without the current context, you should reply exactly `UPDATE CONTEXT`.
    You must give as short an answer as possible.

    User's question is: {input_question}

    Context is: {input_context}
    """

    RETRIEVER_CUSTOM_PROMPT = """Sei un chatbot per una compagnia di assicurazioni. Rispondi alla domanda in base
    al solo contesto fornito.

    La domanda dell'utente è: {input_question}

    Il contesto è: {input_context}
    """

    ragproxyagent = QdrantRetrieveUserProxyAgent(
        name="qdrantagent",
        human_input_mode="NEVER",
        max_consecutive_auto_reply=10,
        retrieve_config={
            "docs_path":files_path,
            "task": "default",
            "chunk_token_size": 2000,
            #"client": client
            "vector_db": "chroma",
            "model": config_list[0]["model"],#chroma,pgvector
            "embedding_model": "BAAI/bge-small-en-v1.5",
            "customized_prompt" : RETRIEVER_CUSTOM_PROMPT,
        },
        code_execution_config=False,
    )
 
    assistant.reset()

    input_text  = ""

    #gli indennizzi sono cumulabili?
    while input_text != "exit":
        input_text = input("insert input (""exit"" to esc):")
        if input_text == "exit": break
        chat_result = ragproxyagent.initiate_chat(assistant, message=ragproxyagent.message_generator, silent=False, problem=input_text)
        #print(response,type(response),response.chat_history)
        print("RESPONSE",chat_result.chat_history[-1]['content'])
        response_content = chat_result.chat_history[-1]['content']
        retrieved_doc_contents = ragproxyagent._get_context(ragproxyagent._results)
        
        evaluation_prompt = EVAL_CUSTOM_PROMPT.format(
            input_question=input_text,
            input_context=retrieved_doc_contents,
            input_answer=response_content
        )

        eval_chat_result = assistant_evaluator.initiate_chat(assistant_evaluator, problem=evaluation_prompt)
        eval_response_content = eval_chat_result.chat_history[-1]['content']
        print("EVAL",eval_response_content)
    



if __name__ == "__main__":
    main()
