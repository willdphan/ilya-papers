from llama_index.core import SimpleDirectoryReader
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core import Settings
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.core import SummaryIndex, VectorStoreIndex
from llama_index.core.tools import QueryEngineTool
from llama_index.core.query_engine.router_query_engine import RouterQueryEngine
from llama_index.core.selectors import LLMSingleSelector


from llama_index.core import SimpleDirectoryReader, VectorStoreIndex, SummaryIndex
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.tools import FunctionTool, QueryEngineTool
from llama_index.core.vector_stores import MetadataFilters, FilterCondition
from typing import List, Optional

urls = [
    "https://nlp.seas.harvard.edu/annotated-transformer/",
    "https://scottaaronson.blog/?p=762",
    "https://karpathy.github.io/2015/05/21/rnn-effectiveness/",
    "https://colah.github.io/posts/2015-08-Understanding-LSTMs/",
    # "https://cs231n.github.io/",
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

# in a subdirectory called papers
papers = [
    "papers/annotated_transformer.pdf",
    "papers/first_law_of_thermodynamics.html",
    "papers/karpathy_rnn_effectiveness.html",
    "papers/colah_understanding_lstms.html",
    "papers/cs231n_2015_rnn_regularization.pdf",
    "papers/keeping_nn_simple_hinton.pdf",
    "papers/pointer_networks.pdf",
    "papers/imagenet_classification.pdf",
    "papers/order_matters_s_to_s.pdf",
    "papers/GPipe.pdf",
    "papers/deep_residual_learning_image_recognition.pdf",
    "papers/multi_scale_context_aggregation.pdf",
    "papers/nerual_quantum_chemistry.pdf",
    "papers/attention_is_all_you_need.pdf",
    "papers/neural_machine_translation.pdf",
    "papers/identity_mappings_deep_residual_networks.pdf",
    "papers/simple_nn_module.pdf",
    "papers/variational_lossy_autoencoder.pdf",
    "papers/relational_rnns.pdf",
    "papers/quantifying_rise_fall_complexity.pdf",
    "papers/neural_turing_machines.pdf",
    "papers/deep_speech_2.pdf",
    "papers/scaling_laws_for_neural_llms.pdf",
    "papers/intro_minimum_description_length_principle.pdf",
    "papers/machine_super_intelligence.pdf",
    "papers/kolmogorov_complexity.pdf",
    # "papers/cnns_for_visual_recognition.html",
]


def get_router_query_engine(file_path: str, llm=None, embed_model=None):
    """Get router query engine."""
    llm = llm or OpenAI(model="gpt-3.5-turbo")
    embed_model = embed_model or OpenAIEmbedding(model="text-embedding-ada-002")

    # load documents
    documents = SimpleDirectoryReader(input_files=[file_path]).load_data()

    splitter = SentenceSplitter(chunk_size=1024)
    nodes = splitter.get_nodes_from_documents(documents)

    summary_index = SummaryIndex(nodes)
    vector_index = VectorStoreIndex(nodes, embed_model=embed_model)

    summary_query_engine = summary_index.as_query_engine(
        response_mode="tree_summarize", use_async=True, llm=llm
    )
    vector_query_engine = vector_index.as_query_engine(llm=llm)

    summary_tool = QueryEngineTool.from_defaults(
        query_engine=summary_query_engine,
        description=("Useful for summarization questions related to MetaGPT"),
    )

    vector_tool = QueryEngineTool.from_defaults(
        query_engine=vector_query_engine,
        description=("Useful for retrieving specific context from the MetaGPT paper."),
    )

    query_engine = RouterQueryEngine(
        selector=LLMSingleSelector.from_defaults(),
        query_engine_tools=[
            summary_tool,
            vector_tool,
        ],
        verbose=True,
    )
    return query_engine


# TODO: abstract all of this into a function that takes in a PDF file name

from llama_index.core import SimpleDirectoryReader, VectorStoreIndex, SummaryIndex
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.tools import FunctionTool, QueryEngineTool
from llama_index.core.vector_stores import MetadataFilters, FilterCondition
from typing import List, Optional


# def get_doc_tools(
#     file_path: str,
#     name: str,
# ) -> str:
#     """Get vector query and summary query tools from a document."""

#     # load documents
#     documents = SimpleDirectoryReader(input_files=[file_path]).load_data()
#     splitter = SentenceSplitter(chunk_size=1024)
#     nodes = splitter.get_nodes_from_documents(documents)
#     vector_index = VectorStoreIndex(nodes)

#     def vector_query(
#         query: str,
#         filter_key_list: List[str],
#         filter_value_list: List[str]
#     ) -> str:
#         """Perform a vector search over an index.

#         query (str): the string query to be embedded.
#         filter_key_list (List[str]): A list of metadata filter field names
#             Must specify ['page_label'] or empty list. Please leave empty
#             if there are no explicit filters to specify.
#         filter_value_list (List[str]): List of metadata filter field values
#             (corresponding to names specified in filter_key_list)

#         """
#         metadata_dicts = [
#             {"key": k, "value": v} for k, v in zip(filter_key_list, filter_value_list)
#         ]

#         query_engine = vector_index.as_query_engine(
#             similarity_top_k=2,
#             filters=MetadataFilters.from_dicts(metadata_dicts)
#         )
#         response = query_engine.query(query)
#         return response

#     vector_query_tool = FunctionTool.from_defaults(
#         fn=vector_query,
#         name=f"vector_query_{name}"
#     )

#     summary_index = SummaryIndex(nodes)
#     summary_query_engine = summary_index.as_query_engine(
#         response_mode="tree_summarize",
#         use_async=True,
#     )
#     summary_tool = QueryEngineTool.from_defaults(
#         query_engine=summary_query_engine,
#         name=f"summary_query_{name}",
#         description=(
#             f"Useful for summarization questions related to {name}"
#         ),
#     )
#     return vector_query_tool, summary_tool


def get_doc_tools(
    file_path: str,
    name: str,
) -> str:
    """Get vector query and summary query tools from a document."""

    # load documents
    documents = SimpleDirectoryReader(input_files=[file_path]).load_data()
    splitter = SentenceSplitter(chunk_size=1024)
    nodes = splitter.get_nodes_from_documents(documents)
    vector_index = VectorStoreIndex(nodes)

    def vector_query(query: str, page_numbers: Optional[List[str]] = None) -> str:
        """Use to answer questions over the MetaGPT paper.

        Useful if you have specific questions over the MetaGPT paper.
        Always leave page_numbers as None UNLESS there is a specific page you want to search for.

        Args:
            query (str): the string query to be embedded.
            page_numbers (Optional[List[str]]): Filter by set of pages. Leave as NONE
                if we want to perform a vector search
                over all pages. Otherwise, filter by the set of specified pages.

        """

        page_numbers = page_numbers or []
        metadata_dicts = [{"key": "page_label", "value": p} for p in page_numbers]

        query_engine = vector_index.as_query_engine(
            similarity_top_k=2,
            filters=MetadataFilters.from_dicts(
                metadata_dicts, condition=FilterCondition.OR
            ),
        )
        response = query_engine.query(query)
        return response

    vector_query_tool = FunctionTool.from_defaults(
        name=f"vector_tool_{name}", fn=vector_query
    )

    summary_index = SummaryIndex(nodes)
    summary_query_engine = summary_index.as_query_engine(
        response_mode="tree_summarize",
        use_async=True,
    )
    summary_tool = QueryEngineTool.from_defaults(
        name=f"summary_tool_{name}",
        query_engine=summary_query_engine,
        description=(
            "Use ONLY IF you want to get a holistic summary of MetaGPT. "
            "Do NOT use if you have specific questions over MetaGPT."
        ),
    )

    return vector_query_tool, summary_tool
