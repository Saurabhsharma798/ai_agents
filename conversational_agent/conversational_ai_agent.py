from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.runnables.history import RunnableWithMessageHistory
# from langchain.memory import ChatMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate,MessagesPlaceholder
from dotenv import load_dotenv

load_dotenv()


llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash")

store = {}


def get_chat_history(session_id: str):
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]


prompt = ChatPromptTemplate.from_messages([
    ("system","You are a helpful AI assistent"),
    MessagesPlaceholder(variable_name="history"),
    ("human","{input}")
])


chain = prompt | llm

chain_with_history = RunnableWithMessageHistory(
    chain,
    get_chat_history,
    input_messages_key="input",
    history_messages_key="history"
)


session_id = "user_123"


response1 = chain_with_history.invoke(
    {"input" : "How are you?"},
    config={"configurable":{"session_id": session_id}}
)


print("AI:",response1.content)


response2 = chain_with_history.invoke(
    {"input":"What was my previous message?"},
    config={"configurable":{"session_id": session_id}}
)

print("AI:",response2.content)

print("---------------------------------------------------\n")

for message in store[session_id].messages:
    print(f"{message.type}:{message.content}")