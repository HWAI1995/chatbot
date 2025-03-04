import os
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from pymongo.mongo_client import MongoClient
from pymongo.server_api import ServerApi
from langchain_mongodb import MongoDBAtlasVectorSearch
from langgraph.graph import MessagesState, StateGraph
from langchain_core.tools import tool
from langchain_core.messages import SystemMessage
from langgraph.prebuilt import ToolNode
from langgraph.graph import END
from langgraph.prebuilt import tools_condition
from langgraph.checkpoint.memory import MemorySaver
from dotenv import load_dotenv

load_dotenv()
os.environ["LANGCHAIN_TRACING_V2"] = os.getenv("LANGCHAIN_TRACING_V2")
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
uri = os.environ["MONGODB_URI"]
    
llm = ChatOpenAI(model="gpt-4o",temperature=0)
embeddings = OpenAIEmbeddings(model="text-embedding-3-large")

client = MongoClient(uri, server_api=ServerApi('1'))

try:
    client.admin.command('ping')
    print("Pinged your deployment. You successfully connected to MongoDB!")
except Exception as e:
    print(e)

db = client["Chatbot"]
collection = db["fakturierung_data"]

vector_store = MongoDBAtlasVectorSearch(
    embedding=embeddings,
    collection=collection,
    index_name="default",
    relevance_score_fn="cosine",
)

graph_builder = StateGraph(MessagesState)

@tool(response_format="content_and_artifact")
def retrieve(query: str):
    """Retrieve information related to a query."""
    retrieved_docs = vector_store.similarity_search(query, k=20)
    serialized = "\n\n".join(
        (f"Source: {doc.metadata}\n" f"Content: {doc.page_content}")
        for doc in retrieved_docs
    )
    return serialized, retrieved_docs

def query_or_respond(state: MessagesState):
    """Generate tool call for retrieval or respond."""
    llm_with_tools = llm.bind_tools([retrieve],tool_choice="required")    
    response = llm_with_tools.invoke([{"role": "system", "content": "Die Fragen beziehen sich auf die Handwerkersoftware: Powerbird®"}, *state["messages"]])

    return {"messages": [response]}

# Step 2: Execute the retrieval.
tools = ToolNode([retrieve])

def generate(state: MessagesState):
    """Generate answer."""
    recent_tool_messages = []
    for message in reversed(state["messages"]):
        if message.type == "tool":
            recent_tool_messages.append(message)
        else:
            break
    tool_messages = recent_tool_messages[::-1]

    docs_content = "\n\n".join(doc.content for doc in tool_messages)
    system_message_content = (
        "Du bist ein kompetenter und professioneller Assistent für Powerbird®, eine vollständig integrierte Handwerkersoftware für Elektro- und Haustechnikbetriebe. Deine Aufgabe ist es, Kunden und Interessenten umfassend zu unterstützen, indem du Fragen zur Software beantwortest, Funktionen erklärst und Hilfestellung bei der Anpassung und Nutzung der Software bietest.\n"
        "Du unterstützt die Kunden bei Fragen zu Powerbird® und befolgst folgende Anweisungen Schritt für Schritt:\n"
        "### Anweisungen:\n"
        "- Berücksichtige dabei nur die gegebenen Daten unten!\n"
        "- Beantworte die Frage ausführlich und strukturiert!\n"
        "- Sei präzise und erfinde Nichts dazu!\n"
        "- Erwähne relevante URLs!\n"
        "- Gib alle Quellen an!\n"
        "- Gib alle Seitenzahlen an!\n\n"
        f"{docs_content}"
    )

    conversation_messages = [
        message
        for message in state["messages"]
        if message.type in ("human", "system")
        or (message.type == "ai" and not message.tool_calls)
    ]
    prompt = [SystemMessage(system_message_content)] + conversation_messages

    response = llm.invoke(prompt)
    return {"messages": [response]}

graph_builder.add_node(query_or_respond)
graph_builder.add_node(tools)
graph_builder.add_node(generate)

graph_builder.set_entry_point("query_or_respond")
graph_builder.add_conditional_edges(
    "query_or_respond",
    tools_condition,
    {END: END, "tools": "tools"},
)
graph_builder.add_edge("tools", "generate")
graph_builder.add_edge("generate", END)

graph = graph_builder.compile()

memory = MemorySaver()
graph = graph_builder.compile(checkpointer=memory)