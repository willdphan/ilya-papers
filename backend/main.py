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
