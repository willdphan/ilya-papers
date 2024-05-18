from llama_index.core.tools import FunctionTool
from llama_index.llms.openai import OpenAI
from llama_index.core import SimpleDirectoryReader
from llama_index.core import VectorStoreIndex
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.vector_stores import MetadataFilters
from typing import List
from llama_index.core.vector_stores import FilterCondition
from llama_index.core import SummaryIndex
from llama_index.core.tools import QueryEngineTool
from llama_index.core.agent import FunctionCallingAgentWorker
from llama_index.core.agent import AgentRunner
from llama_index.core import VectorStoreIndex
from llama_index.core.objects import ObjectIndex
from llama_index.core.agent import FunctionCallingAgentWorker
from llama_index.core.agent import AgentRunner

from pathlib import Path
import requests
import asyncio
from playwright.async_api import async_playwright
import nest_asyncio


import os
from dotenv import load_dotenv

load_dotenv(dotenv_path=".env.local")

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")


def get_openai_api_key():
    print(os.environ.get("OPENAI_API_KEY"))
    return os.environ.get("OPENAI_API_KEY")


OPENAI_API_KEY = get_openai_api_key()


nest_asyncio.apply()


urls = [
    "https://nlp.seas.harvard.edu/annotated-transformer/",
    "https://scottaaronson.blog/?p=762",
    "https://karpathy.github.io/2015/05/21/rnn-effectiveness/",
    "https://colah.github.io/posts/2015-08-Understanding-LSTMs/",
    "https://cs231n.github.io/",
    "https://arxiv.org/pdf/1409.2329.pdf",
    "https://www.cs.toronto.edu/~hinton/absps/colt93.pdf",
    "https://arxiv.org/pdf/1506.03134.pdf",
    "https://proceedings.neurips.cc/paper_files/paper/2012/file/c399862d3b9d6b76c8436e924a68c45b-Paper.pdf",
    "https://arxiv.org/pdf/1511.06391.pdf",
    "https://arxiv.org/pdf/1811.06965.pdf",
    "https://arxiv.org/pdf/1512.03385.pdf",
    "https://arxiv.org/pdf/1511.07122.pdf",
    "https://arxiv.org/pdf/1704.01212.pdf",
    "https://arxiv.org/pdf/1706.03762.pdf",
    "https://arxiv.org/pdf/1409.0473.pdf",
    "https://arxiv.org/pdf/1603.05027.pdf",
    "https://arxiv.org/pdf/1706.01427.pdf",
    "https://arxiv.org/pdf/1611.02731.pdf",
    "https://arxiv.org/pdf/1806.01822.pdf",
    "https://arxiv.org/pdf/1405.6903.pdf",
    "https://arxiv.org/pdf/1410.5401.pdf",
    "https://arxiv.org/pdf/1512.02595.pdf",
    "https://arxiv.org/pdf/2001.08361.pdf",
    "https://arxiv.org/pdf/math/0406077.pdf",
    "https://www.vetta.org/documents/Machine_Super_Intelligence.pdf",
    "https://www.lirmm.fr/~ashen/kolmbook-eng-scan.pdf",
]


papers = [
    "annotated_transformer.pdf",
    "first_law_of_thermodynamics.html",
    "karpathy_rnn_effectiveness.html",
    "colah_understanding_lstms.html",
    "cs231n_2015_rnn_regularization.pdf",
    "keeping_nn_simple_hinton.pdf",
    "pointer_networks.pdf",
    "imagenet_classification.pdf",
    "order_matters_s_to_s.pdf",
    "GPipe.pdf",
    "deep_residual_learning_image_recognition.pdf",
    "multi_scale_context_aggregation.pdf",
    "nerual_quantum_chemistry.pdf",
    "attention_is_all_you_need.pdf",
    "neural_machine_translation.pdf",
    "identity_mappings_deep_residual_networks.pdf",
    "simple_nn_module.pdf",
    "variational_lossy_autoencoder.pdf",
    "relational_rnns.pdf",
    "quantifying_rise_fall_complexity.pdf",
    "neural_turing_machines.pdf",
    "deep_speech_2.pdf" "scaling_laws_for_neural_llms.pdf",
    "intro_minimum_description_length_principle.pdf",
    "machine_super_intelligence.pdf",
    "kolmogorov_complexity.pdf",
    "cnns_for_visual_recognition.html",
]


# Ensure the 'papers' directory exists
os.makedirs("papers", exist_ok=True)

# Update the papers list to include the 'papers' directory
papers = [f"papers/{paper}" for paper in papers]

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


# Function to download PDF content
def download_pdf(url, paper):
    response = requests.get(url, verify=False)
    with open(paper, "wb") as file:
        file.write(response.content)


# Function to handle the asynchronous tasks
# download and parse documents
async def download_documents(urls, papers):
    tasks = []
    for url, paper in zip(urls, papers):
        if url.endswith(".pdf"):
            download_pdf(url, paper)
        else:
            tasks.append(download_and_parse_html(url, paper))
    else:
        print("No documents to download")
    if tasks:
        await asyncio.gather(*tasks)
    if not os.path.exists("papers"):
        os.makedirs("papers", exist_ok=True)


# Download and parse documents
asyncio.run(download_documents(urls, papers))

# Load documents
documents = SimpleDirectoryReader(input_dir="papers").load_data()


splitter = SentenceSplitter(chunk_size=1024)
nodes = splitter.get_nodes_from_documents(documents)


# print(nodes[0].get_content(metadata_mode="all"))

vector_index = VectorStoreIndex(nodes)
query_engine = vector_index.as_query_engine(similarity_top_k=2)

# query engine is for the page number
query_engine = vector_index.as_query_engine(
    similarity_top_k=2,
    filters=MetadataFilters.from_dicts([{"key": "page_label", "value": "2"}]),
)


# response = query_engine.query(
#     "What are some high-level results of MetaGPT?",
# )

# print(str(response))

# for n in response.source_nodes:
    # print(n.metadata)


def vector_query(query: str, page_numbers: List[str]) -> str:
    """Perform a vector search over an index.

    query (str): the string query to be embedded.
    page_numbers (List[str]): Filter by set of pages. Leave BLANK if we want to perform a vector search
        over all pages. Otherwise, filter by the set of specified pages.

    """

    # metadata_dicts = [{"key": "page_label", "value": p} for p in page_numbers]

    # query_engine = vector_index.as_query_engine(
    #     similarity_top_k=2,
    #     filters=MetadataFilters.from_dicts(
    #         metadata_dicts, condition=FilterCondition.OR
    #     ),
    # )
    # response = query_engine.query(query)
    # return response


vector_query_tool = FunctionTool.from_defaults(name="vector_tool", fn=vector_query)

llm = OpenAI("gpt-4o", temperature=0)
# llm = OpenAI(model="gpt-3.5-turbo", temperature=0)
# response = llm.predict_and_call(
#     [vector_query_tool],
#     "What are the high-level results of MetaGPT as described on page 2?",
#     verbose=True,
# )

# for n in response.source_nodes:
#     print(n.metadata)


summary_index = SummaryIndex(nodes)
summary_query_engine = summary_index.as_query_engine(
    response_mode="tree_summarize",
    use_async=True,
)
summary_tool = QueryEngineTool.from_defaults(
    name="summary_tool",
    query_engine=summary_query_engine,
    description=("Useful if you want to get a summary of MetaGPT"),
)

# response = llm.predict_and_call(
#     [vector_query_tool, summary_tool],
#     "What are the MetaGPT comparisons with ChatDev described on page 8?",
#     verbose=True,
# )

# for n in response.source_nodes:
#     print(n.metadata)

# response = llm.predict_and_call(
#     [vector_query_tool, summary_tool], "What is a summary of the paper?", verbose=True
# )
############################################################################################


def get_doc_tools(doc_path: str, doc_name: str):  # optional types
    return [vector_query_tool, summary_tool]


# vector_tool, summary_tool = get_doc_tools("metagpt.pdf", "metagpt")


agent_worker = FunctionCallingAgentWorker.from_tools(
    [vector_query_tool, summary_tool], llm=llm, verbose=True
)

agent_runner = AgentRunner(agent_worker)


agent = AgentRunner(agent_worker)

# print(response.source_nodes[0].get_content(metadata_mode="all"))

## Lower-Level: Debuggability and Control

# agent_worker = FunctionCallingAgentWorker.from_tools(
#     [vector_tool, summary_tool], llm=llm, verbose=True
# )
# agent = AgentRunner(agent_worker)

# task = agent.create_task(
#     "Tell me about the agent roles in MetaGPT, "
#     "and then how they communicate with each other."
# )

# step_output = agent.run_step(task.task_id)

# completed_steps = agent.get_completed_steps(task.task_id)
# print(f"Num completed for task {task.task_id}: {len(completed_steps)}")
# print(completed_steps[0].output.sources[0].raw_output)

# upcoming_steps = agent.get_upcoming_steps(task.task_id)
# print(f"Num upcoming steps for task {task.task_id}: {len(upcoming_steps)}")
# upcoming_steps[0]

# step_output = agent.run_step(
#     task.task_id, input="What about how agents share information?"
# )

# step_output = agent.run_step(task.task_id)
# print(step_output.is_last)

# response = agent.finalize_response(task.task_id)

# print(str(response))

# # Lesson 4: Building a Multi-Document Agent

paper_to_tools_dict = {}
for paper in papers:
    print(f"Getting tools for paper: {paper}")
    vector_tool, summary_tool = get_doc_tools(paper, Path(paper).stem)
    paper_to_tools_dict[paper] = [vector_tool, summary_tool]

# all tools referring to papers in papers list
all_tools = [t for paper in papers for t in paper_to_tools_dict[paper]]


# define an "object" index and retriever over these tools
obj_index = ObjectIndex.from_objects(
    all_tools,
    index_cls=VectorStoreIndex,
)

obj_retriever = obj_index.as_retriever(similarity_top_k=3)

# tools = obj_retriever.retrieve(
#     "Tell me about the eval dataset used in MetaGPT and SWE-Bench"
# )

# # seeing what tools are used in answering the question
# tools[2].metadata

agent_worker = FunctionCallingAgentWorker.from_tools(
    tool_retriever=obj_retriever,
    llm=llm,
    system_prompt=""" \
You are an agent designed to answer queries over a set of given papers.
Please always use the tools provided to answer a question. Do not rely on prior knowledge.\

""",
    verbose=True,
)
agent = AgentRunner(agent_worker)

# response = agent.query(
#     "Explain the annotated transformer for me"
# )
# print(str(response))

response = agent.query(
    # "Compare and contrast the attention is all you need paper and the imagenet classification paper. "
    # "Analyze the approach in each paper first."
    # "What are the high-level results of Attention is all you need as described on page 2?"
    "What are the high-level results of karpathy's RNN effectiveness paper?"
)

for n in response.source_nodes:
    print(n.get_content(metadata_mode="all"))

print(str(response))
