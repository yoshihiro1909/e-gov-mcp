import sys, inspect
sys.path.insert(0, '.')
from mcp_server import search_laws, search_laws_by_keyword, find_law_article, follow_law_chain
for name, obj in [("search_laws", search_laws), ("search_laws_by_keyword", search_laws_by_keyword), ("find_law_article", find_law_article), ("follow_law_chain", follow_law_chain)]:
    fn = obj.fn if hasattr(obj, 'fn') else obj
    print(f"{name}: {inspect.signature(fn)}")
