"""Vectordb module — embeddings, LanceDB client, search tools, and document indexer.

Public API:
    from src.vectordb import embed, embed_batch          # embeddings
    from src.vectordb import get_async_db, get_sync_db    # LanceDB client
    from src.vectordb import vector_search, list_collections  # LangChain tools
    from src.vectordb.indexer import index_documents, main   # CLI indexer
"""

from src.vectordb.embeddings import embed, embed_batch
from src.vectordb.client import get_async_db, get_sync_db
from src.vectordb.tools import vector_search, list_collections
