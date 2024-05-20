import os
from dotenv import load_dotenv
from fastapi import FastAPI
from pydantic import BaseModel
from llama_index.core import VectorStoreIndex
from llama_index.core.objects import ObjectIndex
from llama_index.core.agent import FunctionCallingAgentWorker, AgentRunner
from llama_index.llms.openai import OpenAI
from utils import get_doc_tools
from pathlib import Path
from utils import papers, get_doc_tools  # Updated import
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import asyncio
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import subprocess
import sys
from utils import papers, urls
import requests
from helper import get_openai_api_key
from playwright.async_api import async_playwright

OPENAI_API_KEY = get_openai_api_key()
print(OPENAI_API_KEY)


app = FastAPI()

# Verify that the API key is loaded
print("OpenAI API Key:", os.getenv("OPENAI_API_KEY"))

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

os.makedirs("papers", exist_ok=True)


# Function to download and parse HTML content using Playwright
async def download_and_parse_html(url, paper):
    async with async_playwright() as p:
        browser = await p.chromium.launch()
        page = await browser.new_page()
        await page.goto(url)
        content = await page.content()
        await browser.close()

    with open(paper, "w", encoding="utf-8") as file:
        file.write(content)

    return content


async def download_papers(urls, papers):
    for url, paper in zip(urls, papers):
        file_path = Path(paper)
        if not file_path.exists():
            print(f"Downloading {url} to {paper}")
            response = requests.get(url, verify=False)
            # if link doesn't end in .pdf, then it's an html page
            if not url.endswith(".pdf"):
                # download and parse html page
                await download_and_parse_html(url, paper)
            else:
                with open(paper, "wb") as file:
                    file.write(response.content)


# Load the papers and create the tools (modify this based on your existing code)
paper_to_tools_dict = {}
for paper in papers:
    print(f"Getting tools for paper: {paper}")
    if not os.path.exists(paper):
        print(f"Paper {paper} not found. Downloading...")
        download_papers([url for url, p in zip(urls, papers) if p == paper], [paper])
    vector_tool, summary_tool = get_doc_tools(paper, Path(paper).stem)
    paper_to_tools_dict[paper] = [vector_tool, summary_tool]

# Load the papers and create the tools (modify this based on your existing code)
# paper_to_tools_dict = {}
# for paper in papers:
#     print("getting tools for ", paper)
#     vector_tool, summary_tool = get_doc_tools(paper, Path(paper).stem)
#     paper_to_tools_dict[paper] = [vector_tool, summary_tool]
#     if not paper_to_tools_dict:
#         print("No papers found in the papers list")
#         for url, paper in zip(urls, papers):
#             subprocess.run(["wget", url, "-O", paper], check=True)
#         sys.exit()

# # If there are no papers in the paper_to_tools_dict, then there are no papers in the papers list
# if not paper_to_tools_dict:
#     print("No papers found in the papers list")
#     for url, paper in zip(urls, papers):
#         subprocess.run(["wget", url, "-O", paper], check=True)
#     sys.exit()


all_tools = [t for paper in papers for t in paper_to_tools_dict[paper]]

# Create the object index and retriever
obj_index = ObjectIndex.from_objects(all_tools, index_cls=VectorStoreIndex)
obj_retriever = obj_index.as_retriever(similarity_top_k=3)

# Create the agent worker and runner
llm = OpenAI(model="gpt-3.5-turbo")
agent_worker = FunctionCallingAgentWorker.from_tools(
    tool_retriever=obj_retriever,
    llm=llm,
    system_prompt="You are an agent designed to answer queries over a set of given papers. Please always use the tools provided to answer a question. Do not rely on prior knowledge.",
    verbose=True,
)
agent = AgentRunner(agent_worker)


class UserInput(BaseModel):
    input: str


async def query_agent(input_text: str):
    retries = 5
    for i in range(retries):
        try:
            response = agent.query(input_text)  # Removed await
            return response
        except Exception as e:
            if "rate_limit_exceeded" in str(e):
                wait_time = 2**i  # Exponential backoff
                print(f"Rate limit exceeded. Retrying in {wait_time} seconds...")
                await asyncio.sleep(wait_time)
            else:
                raise e
    raise HTTPException(
        status_code=429, detail="Rate limit exceeded. Please try again later."
    )


@app.post("/generate_response")
async def generate_response(user_input: UserInput):
    response = await query_agent(user_input.input)
    return {"response": str(response)}


# Run the app
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)

# from llama_index.core.tools import FunctionTool
# from llama_index.llms.openai import OpenAI
# from llama_index.core import SimpleDirectoryReader
# from llama_index.core.node_parser import SentenceSplitter
# from llama_index.core.vector_stores import MetadataFilters
# from typing import List
# from llama_index.core.vector_stores import FilterCondition
# from llama_index.core import SummaryIndex
# from llama_index.core.tools import QueryEngineTool
# from llama_index.core.agent import FunctionCallingAgentWorker
# from llama_index.core.agent import AgentRunner
# from llama_index.core import VectorStoreIndex, StorageContext
# from llama_index.core.objects import ObjectIndex
# from llama_index.vector_stores.chroma import ChromaVectorStore

# from llama_index.core import Settings
# from llama_index.llms.openai import OpenAI
# from llama_index.embeddings.openai import OpenAIEmbedding


# from pathlib import Path
# import requests
# import asyncio
# from playwright.async_api import async_playwright
# import chromadb
# from chromadb import EmbeddingFunction, Documents, Embeddings


# # import nest_asyncio
# from fastapi import FastAPI, Request, HTTPException
# from pydantic import BaseModel
# import uvicorn

# # endpoint
# from io import BytesIO
# from typing import IO
# import uuid
# from fastapi import FastAPI, Request, BackgroundTasks, HTTPException
# from fastapi.responses import JSONResponse, StreamingResponse
# from dotenv import load_dotenv
# from fastapi.middleware.cors import CORSMiddleware
# import uvicorn
# from slowapi import Limiter, _rate_limit_exceeded_handler
# from slowapi.util import get_remote_address
# from slowapi.errors import RateLimitExceeded
# from pydantic import BaseModel
# import chromadb


# import os
# from dotenv import load_dotenv

# load_dotenv(dotenv_path=".env.local")

# OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")


# def get_openai_api_key():
#     print(os.environ.get("OPENAI_API_KEY"))
#     return os.environ.get("OPENAI_API_KEY")


# OPENAI_API_KEY = get_openai_api_key()
# # llm = OpenAI("gpt-4o", temperature=0)
# # llm = OpenAI("gpt-3.5-turbo", temperature=0, max_token=100)
# # print("using model: ", llm)
# Settings.llm = OpenAI(model="gpt-3.5-turbo")
# Settings.embed_model = OpenAIEmbedding(model="text-embedding-ada-002")

# chroma_client = chromadb.EphemeralClient()
# db = chromadb.PersistentClient(path="./chroma_db")
# chroma_collection = db.get_or_create_collection("db_collection")
# # chroma_collection = chroma_client.create_collection("start")


# # nest_asyncio.apply()


# urls = [
#     "https://nlp.seas.harvard.edu/annotated-transformer/",
#     "https://scottaaronson.blog/?p=762",
#     "https://karpathy.github.io/2015/05/21/rnn-effectiveness/",
#     "https://colah.github.io/posts/2015-08-Understanding-LSTMs/",
#     "https://cs231n.github.io/",
#     # "https://arxiv.org/pdf/1409.2329.pdf",
#     # "https://www.cs.toronto.edu/~hinton/absps/colt93.pdf",
#     # "https://arxiv.org/pdf/1506.03134.pdf",
#     # "https://proceedings.neurips.cc/paper_files/paper/2012/file/c399862d3b9d6b76c8436e924a68c45b-Paper.pdf",
#     # "https://arxiv.org/pdf/1511.06391.pdf",
#     # "https://arxiv.org/pdf/1811.06965.pdf",
#     # "https://arxiv.org/pdf/1512.03385.pdf",
#     # "https://arxiv.org/pdf/1511.07122.pdf",
#     # "https://arxiv.org/pdf/1704.01212.pdf",
#     # "https://arxiv.org/pdf/1706.03762.pdf",
#     # "https://arxiv.org/pdf/1409.0473.pdf",
#     # "https://arxiv.org/pdf/1603.05027.pdf",
#     # "https://arxiv.org/pdf/1706.01427.pdf",
#     # "https://arxiv.org/pdf/1611.02731.pdf",
#     # "https://arxiv.org/pdf/1806.01822.pdf",
#     # "https://arxiv.org/pdf/1405.6903.pdf",
#     # "https://arxiv.org/pdf/1410.5401.pdf",
#     # "https://arxiv.org/pdf/1512.02595.pdf",
#     # "https://arxiv.org/pdf/2001.08361.pdf",
#     # "https://arxiv.org/pdf/math/0406077.pdf",
#     # "https://www.vetta.org/documents/Machine_Super_Intelligence.pdf",
#     # "https://www.lirmm.fr/~ashen/kolmbook-eng-scan.pdf",
# ]


# papers = [
#     "annotated_transformer.pdf",
#     "first_law_of_thermodynamics.html",
#     "karpathy_rnn_effectiveness.html",
#     "colah_understanding_lstms.html",
#     "cs231n_2015_rnn_regularization.pdf",
#     # "keeping_nn_simple_hinton.pdf",
#     # "pointer_networks.pdf",
#     # "imagenet_classification.pdf",
#     # "order_matters_s_to_s.pdf",
#     # "GPipe.pdf",
#     # "deep_residual_learning_image_recognition.pdf",
#     # "multi_scale_context_aggregation.pdf",
#     # "nerual_quantum_chemistry.pdf",
#     # "attention_is_all_you_need.pdf",
#     # "neural_machine_translation.pdf",
#     # "identity_mappings_deep_residual_networks.pdf",
#     # "simple_nn_module.pdf",
#     # "variational_lossy_autoencoder.pdf",
#     # "relational_rnns.pdf",
#     # "quantifying_rise_fall_complexity.pdf",
#     # "neural_turing_machines.pdf",
#     # "deep_speech_2.pdf" "scaling_laws_for_neural_llms.pdf",
#     # "intro_minimum_description_length_principle.pdf",
#     # "machine_super_intelligence.pdf",
#     # "kolmogorov_complexity.pdf",
#     # "cnns_for_visual_recognition.html",
# ]


# # Ensure the 'papers' directory exists
# os.makedirs("papers", exist_ok=True)

# # Update the papers list to include the 'papers' directory
# papers = [f"papers/{paper}" for paper in papers]


# # Function to download PDF content
# def download_pdf(url, paper):
#     response = requests.get(url, verify=False)
#     # Check if the content type is PDF
#     if "application/pdf" in response.headers.get("Content-Type", ""):
#         with open(paper, "wb") as file:
#             file.write(response.content)
#     else:
#         print(f"Warning: The URL {url} did not return a PDF. Skipping this file.")


# # Function to download and parse HTML content using Playwright
# async def download_and_parse_html(url, paper):
#     async with async_playwright() as p:
#         browser = await p.chromium.launch()
#         page = await browser.new_page()
#         await page.goto(url)
#         content = await page.content()
#         await browser.close()

#     with open(paper, "w", encoding="utf-8") as file:
#         file.write(content)

#     return content


# # download and parse documents
# # Function to handle the asynchronous tasks
# async def download_documents(urls, papers):
#     tasks = []
#     for url, paper in zip(urls, papers):
#         if url.endswith(".pdf"):
#             download_pdf(url, paper)
#         else:
#             tasks.append(download_and_parse_html(url, paper))
#     if tasks:
#         print("tasks: ", tasks)
#         await asyncio.gather(*tasks)
#     if not os.path.exists("papers"):
#         os.makedirs("papers", exist_ok=True)


# # Load documents
# print("loading documents")
# documents = SimpleDirectoryReader(input_dir="papers").load_data()

# # # splitter = SentenceSplitter(chunk_size=1024)
# # print("splitting documents and creating nodes")
# # splitter = SentenceSplitter(chunk_size=1024)
# # nodes = splitter.get_nodes_from_documents(documents)
# # print(f"Number of nodes created: {len(nodes)}")


# # # Create vector store using ChromaDB
# # print("creating vector store using chromadb")
# # vector_store = ChromaVectorStore(chroma_collection=chroma_collection)

# # # Embed nodes and store in ChromaDB
# # print("embedding nodes and adding to storage in chromadb")
# # for i, node in enumerate(nodes):
# #     embedding = Settings.embed_model.get_text_embedding(node.get_content())
# #     unique_id = f"{node.ref_doc_id}_{i}"  # Generate a unique ID by combining ref_doc_id and index
# #     chroma_collection.add(
# #         documents=[node.get_content()],
# #         metadatas=[{"ref_doc_id": node.ref_doc_id}],
# #         ids=[unique_id],
# #         embeddings=[embedding],
# #     )

# # # Embed nodes and store in ChromaDB
# # print("embedding nodes and adding to storage in chromadb")
# # for node in nodes:
# #     embedding = Settings.embed_model.get_text_embedding(node.get_content())
# #     chroma_collection.add(
# #         documents=[node.get_content()],
# #         metadatas=[{"id": node.id}],  # Use node.id instead of node.get_id()
# #         ids=[str(node.id)],  # Convert node.id to string
# #         embeddings=[embedding],
# #     )


# print("creating storage context, will be used to create the vector index")
# storage_context = StorageContext.from_defaults(vector_store=vector_store)
# # vector store index is for vector search
# #  It uses the VectorStoreIndex to manage the embeddings in memory, but this does not persist the embeddings to disk or a database.
# print("creating vector index")
# vector_index = VectorStoreIndex(
#     nodes, show_progress=True, storage_context=storage_context
# )


# # def vector_query(query: str, page_numbers: List[str]) -> str:
# #     """Perform a vector search over an index.

# #     query (str): the string query to be embedded.
# #     page_numbers (List[str]): Filter by set of pages. Leave BLANK if we want to perform a vector search
# #         over all pages. Otherwise, filter by the set of specified pages.

# #     """
# #     metadata_dicts = [{"key": "page_label", "value": p} for p in page_numbers]

# #     query_engine = vector_index.as_query_engine(
# #         # llm=llm,
# #         similarity_top_k=2,
# #         filters=MetadataFilters.from_dicts(
# #             metadata_dicts, condition=FilterCondition.OR
# #         ),
# #     )
# #     response = query_engine.query(query)
# #     return response


# def vector_query(query: str, page_numbers: List[str]) -> str:
#     embed_model = Settings.embed_model
#     # Embed the query
#     query_embedding = embed_model.embed(query)

#     # Query ChromaDB
#     results = chroma_collection.query(query_embeddings=[query_embedding], n_results=2)

#     # Process results
#     response = ""
#     for result in results:
#         response += f"ID: {result['id']}, Content: {result['document']}\n"

#     return response


# vector_query_tool = FunctionTool.from_defaults(name="vector_tool", fn=vector_query)


# summary_index = SummaryIndex(nodes)
# summary_query_engine = summary_index.as_query_engine(
#     # llm=llm,
#     response_mode="tree_summarize",
#     use_async=True,
# )
# summary_tool = QueryEngineTool.from_defaults(
#     name="summary_tool",
#     query_engine=summary_query_engine,
#     description=("Useful if you want to get a summary of MetaGPT"),
# )

# ############################################################################################


# # function that gets the vector and summary tools for a given document
# def get_doc_tools(doc_path: str, doc_name: str):  # optional types
#     return [vector_query_tool, summary_tool]


# # agent worker that uses the vector and summary tools, and the LLM to answer a question
# # agent_worker = FunctionCallingAgentWorker.from_tools(
# #     [vector_query_tool, summary_tool], verbose=True
# # )

# # agent = AgentRunner(agent_worker)

# paper_to_tools_dict = {}
# for paper in papers:
#     print(f"Getting tools for paper: {paper}")
#     vector_tool, summary_tool = get_doc_tools(paper, Path(paper).stem)
#     paper_to_tools_dict[paper] = [vector_tool, summary_tool]

# # all tools referring to papers in papers list
# all_tools = [t for paper in papers for t in paper_to_tools_dict[paper]]


# # define an "object" index and retriever over these tools
# obj_index = ObjectIndex.from_objects(
#     all_tools,
#     index_cls=VectorStoreIndex,
# )

# obj_retriever = obj_index.as_retriever(similarity_top_k=3)

# agent_worker = FunctionCallingAgentWorker.from_tools(
#     tool_retriever=obj_retriever,
#     system_prompt=""" \
# You are an agent designed to answer queries over a set of given papers.
# Please always use the tools provided to answer a question. Do not rely on prior knowledge.\

# """,
#     verbose=True,
# )

# agent = AgentRunner(agent_worker)

# # response = agent.query(
# #     # "Compare and contrast the attention is all you need paper and the imagenet classification paper. "
# #     # "Analyze the approach in each paper first."
# #     # "What are the high-level results of Attention is all you need as described on page 2?"
# #     "What are the high-level results of karpathy's RNN effectiveness paper?"
# #     # "what is in the gpipe paper and how does it compare to rnns?"
# # )

# # for n in response.source_nodes:
# #     # print(n.get_content(metadata_mode="all"))
# #     # print("\n")
# #     print(n.metadata)


# # print(str(response))


# ############################################################################################
# # Define a Pydantic model for the request body
# class QueryRequest(BaseModel):
#     query: str


# limiter = Limiter(key_func=get_remote_address)

# app = FastAPI()

# # origins is the list of origins that the API will allow (the frontend)
# origins = [
#     "http://localhost:3000",
#     # add more origins here
# ]

# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=origins,
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )


# @app.get("/healthcheck")
# def healthcheck():
#     return {"status": "ok"}

#     # Define the endpoint to generate a response


# @app.post("/generate_response")
# async def generate_response(request: QueryRequest):
#     try:
#         # Extract the query from the request
#         query = request.query

#         # Use the pre-initialized agent runner to process the query
#         response = await agent.query(query)

#         # Process the source nodes asynchronously
#         async def process_source_nodes(nodes):
#             for n in nodes:
#                 print(n.metadata)

#         await process_source_nodes(response.source_nodes)

#         # Return the response
#         print(str(response))
#         return {"response": str(response)}
#     except Exception as e:
#         # Handle any errors that occur
#         raise HTTPException(status_code=500, detail=str(e))


# # Run the app
# if __name__ == "__main__":
#     uvicorn.run(app, host="0.0.0.0", port=8000)
