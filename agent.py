import os
from deepagents import create_deep_agent
# from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI
from tools import search_documentation, backend
from prompts import  CHUNK_ANALYST_INSTRUCTIONS, RAG_WORKFLOW_INSTRUCTIONS, SUBAGENT_DELEGATION_INSTRUCTIONS
from dotenv import load_dotenv
from langchain.messages import HumanMessage

load_dotenv()

max_concurrent_analysts= 3

INSTRUCTIONS = (
    RAG_WORKFLOW_INSTRUCTIONS 
    + "\n\n"
    + "="*80
    + "\n\n"
    + SUBAGENT_DELEGATION_INSTRUCTIONS.format(
        max_concurrent_analysts= max_concurrent_analysts
    )
)

chunk_analyst_subagent= {
    "name": "phan tich chunk", 
    "description": (
        "analyse one retrieved doc chunk file"
        "Pass the use question and a single file path under /retrieved/"
    ),
    "system_prompt": CHUNK_ANALYST_INSTRUCTIONS
}

model = ChatOpenAI(model="my_models", base_url="http://localhost:20128/v1"
                            #    api_key= os.environ["GOOGLE_API_KEY"],
                            #    vertexai= False, 
                            #    max_retries= 2 #cho phep thu 2 lan, neu goi loi qua 2 lan thi huy
                               )

agent = create_deep_agent(
    model=model,
    tools=[search_documentation],
    backend=backend,
    system_prompt=INSTRUCTIONS,
    subagents=[chunk_analyst_subagent],
)

# ex_prompt= "Hay cho toi biet thong tin ve chu de so 2 co trong bao cao mau"
# for token, metadata in agent.stream(
#         {
#             "messages": [HumanMessage(content=ex_prompt)]
#         },
#         stream_mode="messages",
#     ):
#     if token.text:
#         print(token.text, end="", flush=True)

