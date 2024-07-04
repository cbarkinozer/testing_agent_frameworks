from autogen import ConversableAgent
import os

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
prompt_price_per_1k=0
completion_token_price_per_1k=0
config_list= [{"model":"llama3-70b-8192", "api_key":GROQ_API_KEY, "base_url": "https://api.groq.com/openai/v1", "price": [prompt_price_per_1k, completion_token_price_per_1k],"frequency_penalty": 0.5,"max_tokens": 2048,"presence_penalty": 0.2,"temperature": 0.5,"top_p": 0.2}]
llm_config = {"config_list":config_list}

        
cathy = ConversableAgent(
    name="cathy",
    system_message=
    "Your name is Cathy and you are a stand-up comedian.",
    llm_config=llm_config,
    human_input_mode="NEVER",
)

joe = ConversableAgent(
    name="joe",
    system_message=
    "Your name is Joe and you are a stand-up comedian. "
    "Start your next joke using the punchline of the previous one.",
    llm_config=llm_config,
    human_input_mode="NEVER",
)

# agent conversation definition
chat_result = joe.initiate_chat(
    recipient=cathy, 
    message="I'm Joe. Cathy, let's keep the jokes rolling.",
    max_turns=2,
)

import pprint

pprint.pprint(chat_result.chat_history)
pprint.pprint(chat_result.cost)
pprint.pprint(chat_result.summary)