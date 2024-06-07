from dotenv import load_dotenv
from pathlib import Path
from llama_index.core import VectorStoreIndex
from llama_index.core.objects import ObjectIndex
from llama_index.core.agent import FunctionCallingAgentWorker
from llama_index.core.agent import AgentRunner
from llama_index.llms.gemini import Gemini
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex, SummaryIndex
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.tools import FunctionTool, QueryEngineTool
from llama_index.core.vector_stores import MetadataFilters, FilterCondition
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.groq import Groq
from llama_index.core import Settings

Settings.embed_model = HuggingFaceEmbedding(
    model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
)
from typing import List, Optional
import os


def load_llm(model_name:str = "gemini"):
    is_loaded = load_dotenv(".env")
    if is_loaded == False:
        print("Env variables are not loaded.")
        exit()
    
    if model_name == "openai":
        #TODO
        pass
    elif model_name == "llama3":
        GROQ_API_KEY = os.getenv("GROQ_API_KEY")
        llm = Groq(api_key=GROQ_API_KEY, model="llama3-70b-8192")
        pass
    else:
        GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
        os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY
        llm = Gemini(model="models/gemini-pro")
    
    return llm

def get_object_index(docs: list, llm):
    doc_to_tools_dict = {}
    for doc in docs:
        print(f"Getting tools for document: {doc}")
        vector_tool, summary_tool = get_doc_tools(doc, Path(doc).stem, llm)
        doc_to_tools_dict[doc] = [vector_tool, summary_tool]
    all_tools = [t for doc in docs for t in doc_to_tools_dict[doc]]
    obj_index = ObjectIndex.from_objects(
        all_tools,
        index_cls=VectorStoreIndex,
    )
    obj_retriever = obj_index.as_retriever(similarity_top_k=3)
    return obj_retriever

def get_doc_tools(file_path: str,name: str, llm) -> tuple:
    """Get vector query and summary query tools from a document."""

    # load documents
    documents = SimpleDirectoryReader(input_files=[file_path]).load_data()
    splitter = SentenceSplitter(chunk_size=1024)
    nodes = splitter.get_nodes_from_documents(documents)
    
    vector_query_tool = FunctionTool.from_defaults(
        name=f"vector_tool_{name}",
        fn=vector_query
    )
    
    summary_index = SummaryIndex(nodes)
    summary_query_engine = summary_index.as_query_engine(
        llm=llm,
        response_mode="tree_summarize",
        use_async=True,
    )
    summary_tool = QueryEngineTool.from_defaults(
        name=f"summary_tool_{name}",
        query_engine=summary_query_engine,
        description=(
            f"Useful for summarization questions related to {name}"
        ),
    )
    return vector_query_tool, summary_tool

def vector_query(file_path: str, query: str, page_numbers: Optional[List[str]] = None) -> str:
        """Use to answer questions over a given document.
    
        Useful if you have specific questions over the document.
        Always leave page_numbers as None UNLESS there is a specific page you want to search for.
    
        Args:
            query (str): the string query to be embedded.
            page_numbers (Optional[List[str]]): Filter by set of pages. Leave as NONE 
                if we want to perform a vector search
                over all pages. Otherwise, filter by the set of specified pages.
        
        """
        documents = SimpleDirectoryReader(input_files=[file_path]).load_data()
        splitter = SentenceSplitter(chunk_size=1024)
        nodes = splitter.get_nodes_from_documents(documents)
        vector_index = VectorStoreIndex(nodes, embed_model='local')

        page_numbers = page_numbers or []
        metadata_dicts = [
            {"key": "page_label", "value": p} for p in page_numbers
        ]
        
        query_engine = vector_index.as_query_engine(
            similarity_top_k=2,
            filters=MetadataFilters.from_dicts(
                metadata_dicts,
                condition=FilterCondition.OR
            )
        )
        response = query_engine.query(query)
        return response

def agentic_rag(obj_retriever, question:str, llm) -> str:
    agent_worker = FunctionCallingAgentWorker.from_tools(
        tool_retriever=obj_retriever,
        llm=llm, 
        system_prompt=""" \
        You are an agent designed to answer queries over a set of given documents.
        Please use the tools provided to answer a question. Do not rely on prior knowledge.\
        """,
        verbose=True
    )
    
    agent = AgentRunner(agent_worker)

    response = agent.query(
        question
    )
    
    answer = str(response)
    
    return answer

if __name__ == "__main__":

    question = ""

    docs = [
        "pdfs/metagpt.pdf",
        "pdfs/longlora.pdf",
        "pdfs/loftq.pdf",
        "pdfs/swebench.pdf",
        "pdfs/selfrag.pdf",
        "pdfs/zipformer.pdf",
        "pdfs/values.pdf",
        "pdfs/finetune_fair_diffusion.pdf",
        "pdfs/knowledge_card.pdf",
        "pdfs/metra.pdf",
        "pdfs/vr_mcl.pdf"
    ]

    llm = load_llm(model_name="gemini")

    object_retriever = get_object_index(docs=docs, llm=llm)

    agentic_rag(obj_retriever=object_retriever, llm=llm, question=question)

    