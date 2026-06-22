import os
from dotenv import load_dotenv
from openai import OpenAI
from gitsource import GithubRepositoryDataReader, chunk_documents
from minsearch import Index
from rag_helper import RAGBase

from toyaikit.llm import OpenAIClient
from toyaikit.tools import Tools
from toyaikit.chat import IPythonChatInterface
from toyaikit.chat.runners import OpenAIResponsesRunner, DisplayingRunnerCallback

search_call_count = 0

def build_index(documents):
    index = Index(
        text_fields=["content"],
        keyword_fields=["filename"]
    )
    index.fit(documents)
    return index

def search(chunk_index, query):
    return chunk_index.search(query)

def main():
    reader = GithubRepositoryDataReader(
        repo_owner="DataTalksClub",
        repo_name="llm-zoomcamp",
        commit_id="8c1834d",
        allowed_extensions={"md"},
        filename_filter=lambda path: "/lessons/" in path,
    )

    files = reader.read()
    documents = []

    for file in files:
        doc = file.parse()
        documents.append(doc)
    print(f"Total documents: {len(documents)}")
    
    load_dotenv()
    openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    
    # 1. RAG without chunking
    indexes = build_index(documents)
    assistant = RAGBase(index=indexes, llm_client=openai_client)
    answer = assistant.rag("How does the agentic loop keep calling the model until it stops?")
    print(f"Without chunking: {answer[1]}")
    
    # 2. RAG with chunking
    chunks = chunk_documents(documents, size=2000, step=1000)
    print(f"Total chunks: {len(chunks)}")
    indexes = build_index(chunks)
    assistant = RAGBase(index=indexes, llm_client=openai_client)
    answer = assistant.rag("How does the agentic loop keep calling the model until it stops?")
    print(f"With chunking: {answer[1]}")

    # 3. Agent Setup
    instructions = (
        "You're a course teaching assistant. Answer the student's question using the "
        "search tool. Make multiple searches with different keywords before answering."
    )

    

    # Define tool properly so toyaikit can parse its schema
    def search_course_materials(query: str) -> list:
        """
        Search the course documents and lesson pages for relevant information based on a keyword query.
        """
        global search_call_count
        search_call_count += 1
        return indexes.search(query,  num_results=5)

    agent_tools = Tools()
    agent_tools.add_tool(search_course_materials)
    
    llm_client = OpenAIClient(
        model="gpt-5.4-mini", 
        client=openai_client, 
        extra_kwargs={"max_output_tokens": 1024},
    )
    
    runner = OpenAIResponsesRunner(
        tools=agent_tools,
        developer_prompt=instructions,
        llm_client=llm_client
    )

    agent_question = "How does the agentic loop work, and how is it different from plain RAG?"

    result = runner.loop(prompt=agent_question)
    print(result.last_message)
    print("search calls:", search_call_count)

if __name__ == "__main__":
    main()