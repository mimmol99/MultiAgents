
import os
from pathlib import Path
from dotenv import load_dotenv
import autogen
from autogen.agentchat.contrib.qdrant_retrieve_user_proxy_agent import QdrantRetrieveUserProxyAgent,RetrieveUserProxyAgent
from autogen.agentchat.contrib.retrieve_assistant_agent import AssistantAgent,RetrieveAssistantAgent
from loading import Loader
from langchain_experimental.text_splitter import SemanticChunker
from typing_extensions import Annotated

load_dotenv(Path("../api_key.env"))

def termination_msg(x):
    return isinstance(x, dict) and "TERMINATE" == str(x.get("content", ""))[-9:].upper()

def main():

    
    config_list=[{
        "model": "gpt-3.5-turbo",
        "api_key": os.environ.get("OPENAI_API_KEY"),
    }]

    llm_config={
            "timeout": 600,
            "cache_seed": 42,
            "config_list": config_list
            #"cache_root_path": "./custom_cache_path"
        }
    

    EVAL_CUSTOM_PROMPT =  """Sei un valutatore di qualità della risposta di un chatbot per una compagnia di assicurazioni.
    In base alla domanda ed al contesto genera una label tra le seguenti: [CORRETTO,ERRATO].

    Rispondi nel seguente formato: " --LABEL-- --spiega qui la scelta della label-- --se erratta,riscrivi qui la risposta correttamente-- ".

    La domanda dell'utente è: ""{input_question}"".

    Il contesto è: ""{input_context}"".

    La risposta è: ""{input_answer}"".
    """


     
    files_path = "/home/utente/Desktop/Projects/CHATBOT_YOLO/ALL_FILES/COMPANY/SUB"
    

    RETRIEVER_CUSTOM_PROMPT = (
    "Sei un assistente chatbot per una compagnia di assicurazioni (Yolo-insurance). "
    "Usa i seguenti pezzi di contesto recuperato per rispondere "
    "alla domanda. Quando citi la nostra compagnia di assicurazione parla in "
    "prima personale plurale (e.g contatta la nostra assistenza clienti)."
    "Non citare mai che hai usato il contesto"
    " o documenti nella risposta "
    " e non citare mai riferimenti ad altre compagnie assicurative."
    "Se non conosci la risposta o non trovi "
    "la risposta nel contesto,scusati e rispondi semplicemnte di contattare l'assistenza"
    "\n"
    "La domanda dell'utente è :""{input_question}"" \n"
    "Contesto: ""{input_context}"" "
    )



    boss = autogen.UserProxyAgent(
        name="Boss",
        is_termination_msg=termination_msg,
        human_input_mode="NEVER",
        code_execution_config=False,  # we don't want to execute code in this case.
        default_auto_reply="Reply `TERMINATE` if the task is done.",
        description="Il boss che fa domande e fornisce i task.",
    )

    ragproxyagent = QdrantRetrieveUserProxyAgent(

        name="admin",
        is_termination_msg=termination_msg,
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
        default_auto_reply="Reply `TERMINATE` if the task is done.",
        description = "assistente che genera il contesto per rispondere al task"
    
    )
 

    assistant = autogen.AssistantAgent(
        name="Assistente",
        is_termination_msg=termination_msg,
        system_message="""
                        Risponde alla domanda in base al contesto.
                        """,
        llm_config=llm_config,
    )

    evaluator = autogen.AssistantAgent(
        name="Valutatore",
        is_termination_msg=termination_msg,
        system_message="""
                        Valutatore. Ricontrolla che la risposta 
                        sia corretta e coerente con il contesto fornito, 
                        fornisci un feedback e se negativo passa il tutto
                        al correttore. Altrimenti se è corretto,restituisci la risposta
                        all'admin
                        """,
        llm_config=llm_config,
    )

    corrector = autogen.AssistantAgent(
        name="Correttore",
        is_termination_msg=termination_msg,
        system_message="""
            Correttore. Correggi la risposta 
            utilizzando il feedback 
            e il contesto,restituisci la risposta corretta all'admin""",
        llm_config=llm_config,
        default_auto_reply="Rispondi `TERMINATE` se la risposta è corretta rispetto al contesto.",
    )

        
    groupchat = autogen.GroupChat(
    agents=[boss,ragproxyagent,assistant,evaluator,corrector], messages=[], max_round=50
    )
    def reset_agents(agents):
        for agent in agents:
            agent.reset()

    reset_agents(groupchat.agents)

    manager = autogen.GroupChatManager(groupchat=groupchat, llm_config=llm_config)

    """
    user_message = input("Insert prompt:""(exit to esc)"" ")

    while user_message!="exit":
        boss.initiate_chat(
            manager,
            message=user_message,
        )
        user_message = input("Insert prompt:""(exit to esc)"" ")
    """

    def call_rag_chat(user_input):


        # In this case, we will have multiple user proxy agents and we don't initiate the chat
        # with RAG user proxy agent.
        # In order to use RAG user proxy agent, we need to wrap RAG agents in a function and call
        # it from other agents.
        def retrieve_content(
            message: Annotated[
                str,
                "Refined message which keeps the original meaning and can be used to retrieve content for code generation and question answering.",
            ],
            n_results: Annotated[int, "number of results"] = 3,
        ) -> str:
            ragproxyagent.n_results = n_results  # Set the number of results to be retrieved.
            # Check if we need to update the context.
            update_context_case1, update_context_case2 = ragproxyagent._check_update_context(message)
            if (update_context_case1 or update_context_case2) and ragproxyagent.update_context:
                ragproxyagent.problem = message if not hasattr(ragproxyagent, "problem") else ragproxyagent.problem
                _, ret_msg = ragproxyagent._generate_retrieve_user_reply(message)
            else:
                _context = {"problem": message, "n_results": n_results}
                ret_msg = ragproxyagent.message_generator(ragproxyagent, None, _context)
            return ret_msg if ret_msg else message

        ragproxyagent.human_input_mode = "NEVER"  # Disable human input for boss_aid since it only retrieves content.

        for caller in [assistant,evaluator,corrector]:
            d_retrieve_content = caller.register_for_llm(
                description="retrieve content for question answering.", api_style="function"
            )(retrieve_content)

        for executor in [boss,assistant,evaluator,corrector]:
            executor.register_for_execution()(d_retrieve_content)

        groupchat = autogen.GroupChat(
            agents=[boss,ragproxyagent,assistant,evaluator,corrector],
            messages=[],
            max_round=12,
            speaker_selection_method="round_robin",
            allow_repeat_speaker=False,
        )

        reset_agents(groupchat.agents)

        manager = autogen.GroupChatManager(groupchat=groupchat, llm_config=llm_config)

        # Start chatting with the boss as this is the user proxy agent.
        boss.initiate_chat(
            manager,
            message=user_input,
        )

    user_message = input("Insert prompt:""(exit to esc)"" ")

    while user_message!="exit":
        call_rag_chat(user_message)
        user_message = input("Insert prompt:""(exit to esc)"" ")

if __name__ == "__main__":
    main()

