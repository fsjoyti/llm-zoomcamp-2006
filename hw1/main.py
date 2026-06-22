"""Homework runner for the Agentic RAG assignment.

This script builds a local document index from the course materials and
can evaluate the homework questions using both raw documents and chunked
documents. It also supports an agentic search flow when an OpenAI API key
is available.
"""

import argparse
import os
import sys

import tiktoken
from dotenv import load_dotenv
from gitsource import GithubRepositoryDataReader, chunk_documents
from minsearch import Index
from openai import OpenAI

sys.path.insert(0, os.path.dirname(__file__))
from rag_helper import RAGBase
from toyaikit.chat import IPythonChatInterface
from toyaikit.chat.runners import DisplayingRunnerCallback, OpenAIResponsesRunner
from toyaikit.llm import OpenAIClient
from toyaikit.tools import Tools

search_call_count = 0


def build_index(documents):
    index = Index(text_fields=["content"], keyword_fields=["filename"])
    index.fit(documents)
    return index


def search(chunk_index, query):
    """Search the provided chunk index for the given query."""
    return chunk_index.search(query)


def count_tokens(text: str) -> int:
    """Count prompt tokens using the cl100k_base tokenizer."""
    encoding = tiktoken.get_encoding("cl100k_base")
    return len(encoding.encode(text))


def main():
    reader = GithubRepositoryDataReader(
        repo_owner="DataTalksClub",
        repo_name="llm-zoomcamp",
        commit_id="8c1834d",
        allowed_extensions={"md"},
        filename_filter=lambda path: "/lessons/" in path,
    )

    parser = argparse.ArgumentParser(description="Run the Agentic RAG homework script.")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run the local document workflow without calling OpenAI.",
    )
    args = parser.parse_args()

    files = reader.read()
    documents = [file.parse() for file in files]
    print(f"Total documents: {len(documents)}")

    if args.dry_run:
        query = "How does the agentic loop keep calling the model until it stops?"
        indexes = build_index(documents)
        results = indexes.search(query)
        print(f"First search result filename: {results[0]['filename']}")

        chunks = chunk_documents(documents, size=2000, step=1000)
        print(f"Total chunks: {len(chunks)}")

        context_lines = []
        for doc in results[:5]:
            context_lines.extend(["C: " + doc["content"], "Q: " + doc["filename"], ""])
        prompt = f"QUESTION: {query}\n\nCONTEXT:\n" + "\n".join(context_lines).strip()
        print(f"Estimated input tokens (raw docs): {count_tokens(prompt)}")

        chunk_results = build_index(chunks).search(query)
        context_lines = []
        for doc in chunk_results[:5]:
            context_lines.extend(["C: " + doc["content"], "Q: " + doc["filename"], ""])
        prompt2 = f"QUESTION: {query}\n\nCONTEXT:\n" + "\n".join(context_lines).strip()
        print(f"Estimated input tokens (chunked docs): {count_tokens(prompt2)}")
        return

    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError(
            "OPENAI_API_KEY is required to run this homework script. "
            "Set it in your environment or in a .env file."
        )

    openai_client = OpenAI(api_key=api_key)

    # 1. RAG without chunking
    indexes = build_index(documents)
    assistant = RAGBase(index=indexes, llm_client=openai_client)
    result = assistant.rag(
        "How does the agentic loop keep calling the model until it stops?"
    )
    print(f"Without chunking usage: {result.usage.input_tokens}")
    print(f"Without chunking answer: {result.answer}")

    # 2. RAG with chunking
    chunks = chunk_documents(documents, size=2000, step=1000)
    print(f"Total chunks: {len(chunks)}")
    indexes = build_index(chunks)
    assistant = RAGBase(index=indexes, llm_client=openai_client)
    result = assistant.rag(
        "How does the agentic loop keep calling the model until it stops?"
    )
    print(f"With chunking usage: {result.usage.input_tokens}")
    print(f"With chunking answer: {result.answer}")

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
        return indexes.search(query, num_results=5)

    agent_tools = Tools()
    agent_tools.add_tool(search_course_materials)

    llm_client = OpenAIClient(
        model="gpt-5.4-mini",
        client=openai_client,
        extra_kwargs={"max_output_tokens": 1024},
    )

    runner = OpenAIResponsesRunner(
        tools=agent_tools, developer_prompt=instructions, llm_client=llm_client
    )

    agent_question = (
        "How does the agentic loop work, and how is it different from plain RAG?"
    )

    result = runner.loop(prompt=agent_question)
    print(result.last_message.encode("utf-8", errors="replace").decode("utf-8"))
    print("search calls:", search_call_count)


if __name__ == "__main__":
    main()
