"""Vector search tools — only interface to LanceDB.

LangChain @tool wrappers for use with bind_tools() in the Search Fanout agent.
"""

from langchain_core.tools import tool

from src.vectordb.embeddings import embed
from src.vectordb.client import get_async_db


@tool
async def vector_search(
    query: str,
    collection: str,
    top_k: int = 5,
) -> dict:
    """Search a LanceDB collection by vector similarity.

    Use this tool to find relevant text chunks from a specific document collection.
    Choose the right collection based on the Planner's route.

    Args:
        query: The search query text (will be embedded).
        collection: Name of the LanceDB table/collection to search.
        top_k: Number of top results to return (default 5).

    Returns:
        Dict with keys: collection, query, chunks (list of text), scores (list of distances).
    """
    db = await get_async_db()
    try:
        table = await db.open_table(collection)
    except Exception:
        return {
            "collection": collection,
            "query": query,
            "chunks": [],
            "scores": [],
            "error": f"Collection '{collection}' not found",
        }

    query_embedding = await embed(query)
    results = await table.search(query_embedding).limit(top_k).to_list()

    return {
        "collection": collection,
        "query": query,
        "chunks": [r.get("text", "") for r in results],
        "scores": [r.get("_distance", 0.0) for r in results],
    }


@tool
async def list_collections() -> list[str]:
    """List all available LanceDB collections (tables).

    Use this to discover what document sources are available before searching.
    The Planner should call this first to decide where to route queries.

    Returns:
        List of collection/table names.
    """
    db = await get_async_db()
    return await db.table_names()
