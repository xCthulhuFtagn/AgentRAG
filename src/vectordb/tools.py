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
    db_path: str | None = None,
) -> dict:
    """Search a LanceDB collection by vector similarity.

    Use this tool to find relevant text chunks from a specific document collection.
    Choose the right collection based on the Planner's route.

    Args:
        query: The search query text (will be embedded).
        collection: Name of the LanceDB table/collection to search.
        top_k: Number of top results to return (default 5).
        db_path: Optional LanceDB path to scope the search (per-project isolation).
            None → global LANCE_DB_PATH.

    Returns:
        Dict with keys: collection, query, chunks (list of text), scores (list of distances).
    """
    db = await get_async_db(db_path)
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
    # Async LanceDB: search() is a coroutine — await it before chaining.
    search_query = await table.search(query_embedding)
    results = await search_query.limit(top_k).to_list()

    return {
        "collection": collection,
        "query": query,
        "chunks": [r.get("text", "") for r in results],
        "scores": [r.get("_distance", 0.0) for r in results],
    }


@tool
async def list_collections(db_path: str | None = None) -> list[str]:
    """List all available LanceDB collections (tables).

    Use this to discover what document sources are available before searching.
    The Planner should call this first to decide where to route queries.

    Args:
        db_path: Optional LanceDB path to scope the listing (per-project isolation).
            None → global LANCE_DB_PATH.

    Returns:
        List of collection/table names.
    """
    db = await get_async_db(db_path)
    # Async LanceDB: list_tables() (table_names() is deprecated) returns a
    # paginated ListTablesResponse — walk page_token to collect every name.
    names: list[str] = []
    page_token = None
    while True:
        resp = await db.list_tables(page_token=page_token)
        names.extend(resp.tables)
        page_token = resp.page_token
        if not page_token:
            break
    return names
