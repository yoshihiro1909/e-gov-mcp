import ast
with open('src/mcp_server.py', encoding='utf-8') as f:
    ast.parse(f.read())
print('OK')
