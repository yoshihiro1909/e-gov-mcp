#!/usr/bin/env python3
"""
Self-contained entry point for e-Gov Law MCP Server
Handles dependency issues on Windows/FastMCP environments
"""
import sys
import os
from pathlib import Path

# Add src directory to Python path
src_dir = Path(__file__).parent / "src"
sys.path.insert(0, str(src_dir))

def check_dependencies():
    """Check and warn about missing optional dependencies"""
    missing_deps = []
    
    # Check for required dependencies
    try:
        import fastmcp
    except ImportError:
        missing_deps.append("fastmcp")
    
    try:
        import httpx
    except ImportError:
        missing_deps.append("httpx")
    
    try:
        import yaml
    except ImportError:
        missing_deps.append("PyYAML")
    
    # Check for optional dependencies
    try:
        import psutil
    except ImportError:
        print("Warning: psutil not available - memory monitoring disabled", file=sys.stderr)
        print("To enable performance monitoring: pip install psutil", file=sys.stderr)
    
    if missing_deps:
        print(f"Error: Missing required dependencies: {', '.join(missing_deps)}", file=sys.stderr)
        print("Please install with: pip install " + " ".join(missing_deps), file=sys.stderr)
        sys.exit(1)

def main():
    """Main entry point with dependency checking"""
    check_dependencies()
    
    # Import and run the MCP server
    try:
        from mcp_server import main as server_main
        server_main()
    except Exception as e:
        print(f"Error starting MCP server: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()