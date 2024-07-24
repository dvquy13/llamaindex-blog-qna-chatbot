from llama_index.core.query_engine.retriever_query_engine import (
    RetrieverQueryEngine,
)
from llama_index.core.base.base_retriever import BaseRetriever
from llama_index.core.response_synthesizers.base import BaseSynthesizer
from llama_index.core.schema import QueryType
from llama_index.core.base.response.schema import RESPONSE_TYPE


class ManualAppendReferenceQueryEngine(RetrieverQueryEngine):
    """RAG Query Engine."""

    retriever: BaseRetriever
    response_synthesizer: BaseSynthesizer

    def query(self, str_or_query_bundle: QueryType) -> RESPONSE_TYPE:
        response_obj = super().query(str_or_query_bundle)
        sources_fmt = self._compile_sources(response_obj)
        if sources_fmt:
            response_fmt = f"""
{response_obj.response}

{sources_fmt}
"""
            response_obj.response = response_fmt
        return response_obj

    async def aquery(self, str_or_query_bundle: QueryType) -> RESPONSE_TYPE:
        response_obj = await super().aquery(str_or_query_bundle)
        sources_fmt = self._compile_sources(response_obj)
        if sources_fmt:
            response_fmt = f"""
{response_obj.response}

{sources_fmt}
"""
            response_obj.response = response_fmt
        return response_obj

    def _compile_sources(self, response: RESPONSE_TYPE) -> str:
        sources = dict()
        for node in response.source_nodes:
            url = node.metadata.get("url")
            title = node.metadata.get("title")
            if url not in sources:
                sources[url] = title
        if sources:
            refs = [f"[{title}]({url})" for url, title in sources.items()]
            refs_fmt = "\n- ".join(refs)
            if len(refs) == 1:
                refs_fmt = f"- {refs_fmt}"
            output = f"""
Sources:
{refs_fmt}
"""
            return output
