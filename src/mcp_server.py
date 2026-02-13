#!/usr/bin/env python3
"""
e-Gov Law MCP Server v2 - Ultra Smart & Efficient

A highly optimized Model Context Protocol server for Japanese e-Gov Law API.
Drastically simplified from 1000+ lines to <500 lines while adding more functionality.

Key Improvements:
- Direct mapping for 16+ major laws (六法 + key legislation)
- Smart Base64/XML text extraction
- Efficient article search with intelligent pattern matching
- Minimal API calls with maximum accuracy
- Clean, maintainable code architecture
"""

import argparse
import base64
import json
import logging
import os
import re
import threading
import time
import xml.etree.ElementTree as ET
from collections import OrderedDict
from pathlib import Path
from typing import Any, Optional

import httpx
import yaml
from fastmcp import FastMCP, Context
from fastmcp.exceptions import ToolError, ResourceError

# Optional import for performance monitoring
try:
    import psutil
    PERFORMANCE_MONITORING_AVAILABLE = True
except ImportError:
    PERFORMANCE_MONITORING_AVAILABLE = False

try:
    from .prompt_loader import PromptLoader
except ImportError:
    from prompt_loader import PromptLoader

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Log psutil availability
if not PERFORMANCE_MONITORING_AVAILABLE:
    logger.warning("psutil not available - memory monitoring disabled. Install with: pip install psutil")

# Performance optimization classes
class LRUCache:
    """Thread-safe LRU cache implementation with TTL support"""

    def __init__(self, max_size: int = 100, ttl: int = 3600):
        self.max_size = max_size
        self.ttl = ttl
        self.cache = OrderedDict()
        self.timestamps = {}
        self.lock = threading.RLock()

    def get(self, key: str) -> Optional[Any]:
        with self.lock:
            if key not in self.cache:
                return None

            # Check TTL
            if time.time() - self.timestamps[key] > self.ttl:
                del self.cache[key]
                del self.timestamps[key]
                return None

            # Move to end (most recently used)
            self.cache.move_to_end(key)
            return self.cache[key]

    def put(self, key: str, value: Any) -> None:
        with self.lock:
            if key in self.cache:
                # Update existing key
                self.cache[key] = value
                self.timestamps[key] = time.time()
                self.cache.move_to_end(key)
            else:
                # Add new key
                if len(self.cache) >= self.max_size:
                    # Remove least recently used
                    oldest_key = next(iter(self.cache))
                    del self.cache[oldest_key]
                    del self.timestamps[oldest_key]

                self.cache[key] = value
                self.timestamps[key] = time.time()

    def clear(self) -> None:
        with self.lock:
            self.cache.clear()
            self.timestamps.clear()

    def size(self) -> int:
        with self.lock:
            return len(self.cache)

    def cleanup_expired(self) -> None:
        """Remove expired entries"""
        with self.lock:
            current_time = time.time()
            expired_keys = [
                key for key, timestamp in self.timestamps.items()
                if current_time - timestamp > self.ttl
            ]
            for key in expired_keys:
                del self.cache[key]
                del self.timestamps[key]

class MemoryMonitor:
    """Memory usage monitoring for cache management"""

    def __init__(self, max_memory_mb: int = 512):
        self.max_memory_mb = max_memory_mb
        if PERFORMANCE_MONITORING_AVAILABLE:
            self.process = psutil.Process()
        else:
            self.process = None

    def get_memory_usage_mb(self) -> float:
        """Get current memory usage in MB"""
        if PERFORMANCE_MONITORING_AVAILABLE and self.process:
            return self.process.memory_info().rss / 1024 / 1024
        return 0.0  # Return 0 if psutil not available

    def is_memory_limit_exceeded(self) -> bool:
        """Check if memory limit is exceeded"""
        if not PERFORMANCE_MONITORING_AVAILABLE:
            return False  # Never exceed limit if monitoring disabled
        return self.get_memory_usage_mb() > self.max_memory_mb

class DiskCache:
    """Disk-based cache for e-Gov API responses to avoid repeated downloads"""

    def __init__(self, cache_dir: str = None, ttl: int = 86400 * 30):
        if cache_dir is None:
            cache_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".egov_cache")
        self.cache_dir = os.path.abspath(cache_dir)
        self.ttl = ttl  # 30 days default
        os.makedirs(self.cache_dir, exist_ok=True)

    def _path(self, key: str) -> str:
        import hashlib
        h = hashlib.sha256(key.encode()).hexdigest()[:20]
        return os.path.join(self.cache_dir, f"{h}.json")

    def get(self, key: str) -> Optional[Any]:
        path = self._path(key)
        if not os.path.exists(path):
            return None
        try:
            age = time.time() - os.path.getmtime(path)
            if age > self.ttl:
                os.remove(path)
                return None
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return None

    def put(self, key: str, value: Any) -> None:
        try:
            with open(self._path(key), "w", encoding="utf-8") as f:
                json.dump(value, f, ensure_ascii=False)
        except Exception as e:
            logger.warning(f"Disk cache write failed: {e}")

    def clear(self) -> None:
        import glob
        for f in glob.glob(os.path.join(self.cache_dir, "*.json")):
            try:
                os.remove(f)
            except Exception:
                pass

    def size(self) -> int:
        import glob
        return len(glob.glob(os.path.join(self.cache_dir, "*.json")))


class CacheManager:
    """Centralized cache management with prefetching and batch operations"""

    def __init__(self):
        self.law_lookup_cache = LRUCache(max_size=200, ttl=7200)  # 2 hours
        self.law_content_cache = LRUCache(max_size=50, ttl=3600)  # 1 hour
        self.article_cache = LRUCache(max_size=500, ttl=1800)     # 30 minutes
        self.disk_cache = DiskCache()  # 30 days on disk
        self.memory_monitor = MemoryMonitor()
        self.batch_pending = {}
        self.batch_lock = threading.Lock()

        # Common law articles for prefetching
        self.common_articles = [
            ("民法", "1"), ("民法", "192"), ("民法", "709"),
            ("憲法", "9"), ("憲法", "14"), ("憲法", "25"),
            ("会社法", "1"), ("会社法", "105"), ("会社法", "362"),
            ("刑法", "1"), ("刑法", "199"), ("刑法", "235"),
        ]

    def get_cache_key(self, law_name: str, article_number: str = None) -> str:
        """Generate cache key"""
        if article_number:
            return f"{law_name}:{article_number}"
        return law_name

    def should_clear_cache(self) -> bool:
        """Check if cache should be cleared due to memory pressure"""
        return self.memory_monitor.is_memory_limit_exceeded()

    def cleanup_if_needed(self) -> None:
        """Cleanup expired entries and manage memory"""
        if self.should_clear_cache():
            logger.info("Memory limit exceeded, clearing caches")
            self.law_lookup_cache.clear()
            self.law_content_cache.clear()
            self.article_cache.clear()
        else:
            self.law_lookup_cache.cleanup_expired()
            self.law_content_cache.cleanup_expired()
            self.article_cache.cleanup_expired()

    async def prefetch_common_articles(self, client: httpx.AsyncClient) -> None:
        """Prefetch commonly accessed articles"""
        logger.info("Starting prefetch of common articles")

        for law_name, article_number in self.common_articles:
            cache_key = self.get_cache_key(law_name, article_number)

            # Skip if already cached
            if self.article_cache.get(cache_key):
                continue

            try:
                # Get law number
                law_num = await self._get_law_number(law_name, client)
                if not law_num:
                    continue

                # Get law content
                law_content = await self._get_law_content(law_num, client)
                if law_content:
                    # Store in cache
                    self.law_content_cache.put(law_num, law_content)
                    logger.debug(f"Prefetched {law_name} content")

            except Exception as e:
                logger.warning(f"Failed to prefetch {law_name}: {e}")

    async def _get_law_number(self, law_name: str, client: httpx.AsyncClient) -> Optional[str]:
        """Get law number with caching"""
        cache_key = self.get_cache_key(law_name)

        # Check cache first
        cached_num = self.law_lookup_cache.get(cache_key)
        if cached_num:
            return cached_num

        # Check direct mapping
        if law_name in BASIC_LAWS:
            law_num = BASIC_LAWS[law_name]
            self.law_lookup_cache.put(cache_key, law_num)
            return law_num

        # API lookup
        try:
            response = await client.get("/laws", params={
                "law_title": law_name,
                "law_type": "Act",
                "limit": 5
            })
            response.raise_for_status()

            data = json.loads(response.text)
            laws = data.get("laws", [])

            if laws:
                law_num = laws[0].get("law_info", {}).get("law_num")
                if law_num:
                    self.law_lookup_cache.put(cache_key, law_num)
                    return law_num

        except Exception as e:
            logger.error(f"Failed to get law number for {law_name}: {e}")

        return None

    async def _get_law_content(self, law_num: str, client: httpx.AsyncClient) -> Optional[dict]:
        """Get law content with caching"""
        # Check cache first
        cached_content = self.law_content_cache.get(law_num)
        if cached_content:
            return cached_content

        try:
            response = await client.get(f"/law_data/{law_num}", params={
                "law_full_text_format": "xml"
            })
            response.raise_for_status()

            data = json.loads(response.text)
            self.law_content_cache.put(law_num, data)
            return data

        except Exception as e:
            logger.error(f"Failed to get law content for {law_num}: {e}")

        return None

    async def batch_request_laws(self, law_names: list[str], client: httpx.AsyncClient) -> dict[str, str]:
        """Batch request multiple law numbers"""
        results = {}

        # Separate cached and non-cached requests
        cached_requests = []
        api_requests = []

        for law_name in law_names:
            cache_key = self.get_cache_key(law_name)
            cached_num = self.law_lookup_cache.get(cache_key)

            if cached_num:
                results[law_name] = cached_num
                cached_requests.append(law_name)
            else:
                api_requests.append(law_name)

        logger.info(f"Batch request: {len(cached_requests)} cached, {len(api_requests)} API requests")

        # Process uncached requests
        if api_requests:
            # Group similar requests to reduce API calls
            unique_requests = list(set(api_requests))

            for law_name in unique_requests:
                try:
                    law_num = await self._get_law_number(law_name, client)
                    if law_num:
                        results[law_name] = law_num
                        # Also cache for other identical requests
                        for other_law in api_requests:
                            if other_law == law_name:
                                results[other_law] = law_num
                except Exception as e:
                    logger.error(f"Failed to get law number for {law_name} in batch: {e}")

        return results

# API configuration
API_URL = os.environ.get("EGOV_API_URL", "https://laws.e-gov.go.jp/api/2")
API_TOKEN = os.environ.get("EGOV_API_TOKEN", "")

# Create MCP server with Windows-compatible configuration
mcp = FastMCP(
    name=os.environ.get("MCP_SERVER_NAME", "e-Gov Law API Server v2"),
    mask_error_details=True,  # Security: mask internal error details
    on_duplicate_tools="warn",
    on_duplicate_resources="warn",
    on_duplicate_prompts="warn"
)

class ConfigLoader:
    """
    Configuration loader for law mappings with backward compatibility.

    Loads law aliases and basic laws from YAML configuration file.
    Falls back to hardcoded values for backward compatibility.
    """

    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize ConfigLoader with optional custom config path.

        Args:
            config_path: Path to YAML config file. If None, uses environment variable
                        LAW_CONFIG_PATH or defaults to config/laws.yaml
        """
        # Windows-compatible path handling
        default_config = Path(__file__).parent.parent / "config" / "laws.yaml"
        config_env = os.environ.get("LAW_CONFIG_PATH")
        if config_env:
            self.config_path = Path(config_env)
        elif config_path:
            self.config_path = Path(config_path)
        else:
            self.config_path = default_config
        self._law_aliases: Optional[dict[str, str]] = None
        self._basic_laws: Optional[dict[str, str]] = None
        self._related_laws: Optional[dict[str, dict]] = None

        # Fallback hardcoded values for backward compatibility
        self._fallback_law_aliases = {
            # 一般的な略称
            "道交法": "道路交通法",
            "労基法": "労働基準法",
            "独禁法": "独占禁止法",
            "消契法": "消費者契約法",
            "著作権": "著作権法",
            "特許": "特許法",
            "建基法": "建築基準法",

            # 分野別検索
            "税法": "所得税法",
            "労働法": "労働基準法",
            "知財法": "著作権法",
            "交通法": "道路交通法",

            # 一般的な呼び方
            "会社": "会社法",
            "民事": "民法",
            "刑事": "刑法",
            "訴訟": "民事訴訟法",
        }

        self._fallback_basic_laws = {
            # 六法 (Six Codes)
            "民法": "明治二十九年法律第八十九号",
            "憲法": "昭和二十一年憲法",
            "日本国憲法": "昭和二十一年憲法",
            "刑法": "明治四十年法律第四十五号",
            "商法": "昭和二十三年法律第二十五号",
            "民事訴訟法": "平成八年法律第百九号",
            "刑事訴訟法": "昭和二十三年法律第百三十一号",

            # 現代重要法 (Modern Key Laws)
            "会社法": "平成十七年法律第八十六号",
            "労働基準法": "昭和二十二年法律第四十九号",
            "所得税法": "昭和四十年法律第三十三号",
            "法人税法": "昭和四十年法律第三十四号",
            "著作権法": "昭和四十五年法律第四十八号",
            "特許法": "昭和三十四年法律第百二十一号",
            "道路交通法": "昭和三十五年法律第百五号",
            "建築基準法": "昭和二十五年法律第二百一号",
            "独占禁止法": "昭和二十二年法律第五十四号",
            "消費者契約法": "平成十二年法律第六十一号",
            "特定受託事業者に係る取引の適正化等に関する法律": "令和五年法律第二十五号",
        }

    def _load_config(self) -> dict[str, Any]:
        """Load configuration from YAML file with Windows support."""
        try:
            if self.config_path.exists():
                # Windows-compatible UTF-8 file reading
                with open(self.config_path, encoding='utf-8', newline='') as f:
                    config = yaml.safe_load(f)
                    logger.info(f"Loaded configuration from {self.config_path}")
                    return config or {}
            else:
                logger.warning(f"Config file not found at {self.config_path}, using fallback values")
                return {}
        except Exception as e:
            logger.error(f"Failed to load config from {self.config_path}: {e}")
            logger.info("Using fallback values for backward compatibility")
            return {}

    @property
    def law_aliases(self) -> dict[str, str]:
        """Get law aliases mapping."""
        if self._law_aliases is None:
            config = self._load_config()
            self._law_aliases = config.get('law_aliases', self._fallback_law_aliases)
        return self._law_aliases

    @property
    def basic_laws(self) -> dict[str, str]:
        """Get basic laws mapping."""
        if self._basic_laws is None:
            config = self._load_config()
            self._basic_laws = config.get('basic_laws', self._fallback_basic_laws)
        return self._basic_laws

    @property
    def related_laws(self) -> dict[str, dict]:
        """Get related laws mapping (base law -> 施行令/施行規則)."""
        if self._related_laws is None:
            config = self._load_config()
            self._related_laws = config.get('related_laws', {})
        return self._related_laws

    def reload_config(self) -> None:
        """Reload configuration from file."""
        self._law_aliases = None
        self._basic_laws = None
        self._related_laws = None
        logger.info("Configuration reloaded")

# Initialize global config loader, prompt loader, and cache manager
config_loader = ConfigLoader()
prompt_loader = PromptLoader()
cache_manager = CacheManager()

# LAW ALIASES MAPPING (略称・通称から正式名称へ) - now loaded from config
LAW_ALIASES = config_loader.law_aliases

# COMPREHENSIVE BASIC LAWS MAPPING (16 major laws) - now loaded from config
BASIC_LAWS = config_loader.basic_laws

async def get_http_client() -> httpx.AsyncClient:
    """Create HTTP client for e-Gov API."""
    headers = {
        "User-Agent": "e-Gov-Law-MCP-v2/2.0",
        "Accept": "application/json"
    }
    if API_TOKEN:
        headers["Authorization"] = f"Bearer {API_TOKEN}"

    return httpx.AsyncClient(
        base_url=API_URL,
        headers=headers,
        timeout=30.0,
        follow_redirects=True
    )

def extract_text_from_xml(obj) -> str:
    """
    Smart text extraction from e-Gov API response.
    Handles both Base64-encoded XML and structured JSON.
    """
    if isinstance(obj, str):
        # Handle Base64-encoded XML (XML format response)
        try:
            xml_bytes = base64.b64decode(obj)
            xml_string = xml_bytes.decode('utf-8')
            root = ET.fromstring(xml_string)

            def extract_xml_text(element):
                text = element.text or ''
                for child in element:
                    text += extract_xml_text(child)
                text += element.tail or ''
                return text

            return extract_xml_text(root)
        except Exception as e:
            logger.warning(f"XML decode failed: {e}")
            return str(obj)

    elif isinstance(obj, dict):
        # Handle structured JSON response
        if 'children' in obj:
            return ''.join(extract_text_from_xml(child) for child in obj['children'])
        elif 'text' in obj:
            return obj['text']
        return str(obj)

    elif isinstance(obj, list):
        return ''.join(extract_text_from_xml(item) for item in obj)

    return str(obj)

def extract_articles_from_xml(law_full_text) -> list[dict]:
    """
    Parse XML structure and extract all Article elements as structured data.
    This provides precise article matching by using XML Num attributes
    instead of unreliable regex on flattened text.
    
    Returns list of dicts with keys: num, caption, title, paragraphs, full_text
    """
    xml_string = None
    
    if isinstance(law_full_text, str):
        try:
            xml_bytes = base64.b64decode(law_full_text)
            xml_string = xml_bytes.decode('utf-8')
        except Exception:
            return []
    elif isinstance(law_full_text, dict):
        # Try to find XML content in dict structure
        return []
    else:
        return []
    
    if not xml_string:
        return []
    
    try:
        root = ET.fromstring(xml_string)
    except ET.ParseError:
        return []
    
    def elem_full_text(elem):
        """Recursively extract all text from an element."""
        text = elem.text or ''
        for child in elem:
            text += elem_full_text(child)
            text += child.tail or ''
        return text
    
    def _parse_subitems_recursive(parent_elem, level=1, max_level=10):
        """
        Recursively parse sub-items from Subitem1 → Subitem2 → ... → Subitem10.
        Returns a nested tree structure.
        """
        if level > max_level:
            return []
        tag = f'Subitem{level}'
        result = []
        for si_elem in parent_elem.findall(tag):
            si_num = si_elem.get('Num', '')
            si_title_elem = si_elem.find(f'{tag}Title')
            si_title = elem_full_text(si_title_elem).strip() if si_title_elem is not None else ''
            si_sentence_elem = si_elem.find(f'{tag}Sentence')
            si_sentence = elem_full_text(si_sentence_elem).strip() if si_sentence_elem is not None else ''
            entry = {
                'num': si_num,
                'title': si_title,
                'text': si_sentence,
                'level': tag,
            }
            # Recurse into deeper sub-items
            children = _parse_subitems_recursive(si_elem, level + 1, max_level)
            if children:
                entry['sub_items'] = children
            result.append(entry)
        return result

    # Build a mapping of elements to their location (本則 vs 附則)
    def _find_articles_in_subtree(parent, source_tag):
        """Extract articles from a subtree, tagging with source."""
        result = []
        for article_elem in parent.iter('Article'):
            num = article_elem.get('Num', '')
            
            # Extract caption (e.g., （定義）)
            caption = ''
            caption_elem = article_elem.find('ArticleCaption')
            if caption_elem is not None:
                caption = elem_full_text(caption_elem).strip()
            
            # Extract title (e.g., 第一条)
            title = ''
            title_elem = article_elem.find('ArticleTitle')
            if title_elem is not None:
                title = elem_full_text(title_elem).strip()
            
            # Extract paragraphs with items
            paragraphs = []
            for para_elem in article_elem.findall('Paragraph'):
                para_num = para_elem.get('Num', '')
                para_text = elem_full_text(para_elem).strip()
                
                items = []
                for item_elem in para_elem.findall('.//Item'):
                    item_num = item_elem.get('Num', '')
                    item_text = elem_full_text(item_elem).strip()
                    # Extract sub-items recursively (Subitem1→2→...→10)
                    sub_items = _parse_subitems_recursive(item_elem, level=1)
                    item_data = {'num': item_num, 'text': item_text}
                    if sub_items:
                        item_data['sub_items'] = sub_items
                    items.append(item_data)
                
                paragraphs.append({
                    'num': para_num,
                    'text': para_text,
                    'items': items
                })
            
            full_text = elem_full_text(article_elem).strip()
            
            result.append({
                'num': num,
                'caption': caption,
                'title': title,
                'paragraphs': paragraphs,
                'full_text': full_text,
                'source': source_tag,  # 'main' or 'suppl'
            })
        return result
    
    articles = []
    
    # Try to find MainProvision (本則) and SupplProvision (附則) separately
    main_provision = root.find('.//MainProvision')
    suppl_provisions = root.findall('.//SupplProvision')
    
    if main_provision is not None:
        articles.extend(_find_articles_in_subtree(main_provision, 'main'))
        for suppl in suppl_provisions:
            articles.extend(_find_articles_in_subtree(suppl, 'suppl'))
    else:
        # Fallback: treat all articles as main
        for article_elem in root.iter('Article'):
            num = article_elem.get('Num', '')
            caption = ''
            caption_elem = article_elem.find('ArticleCaption')
            if caption_elem is not None:
                caption = elem_full_text(caption_elem).strip()
            title = ''
            title_elem = article_elem.find('ArticleTitle')
            if title_elem is not None:
                title = elem_full_text(title_elem).strip()
            paragraphs = []
            for para_elem in article_elem.findall('Paragraph'):
                para_num = para_elem.get('Num', '')
                para_text = elem_full_text(para_elem).strip()
                items = []
                for item_elem in para_elem.findall('.//Item'):
                    item_num = item_elem.get('Num', '')
                    item_text = elem_full_text(item_elem).strip()
                    sub_items = _parse_subitems_recursive(item_elem, level=1)
                    item_data = {'num': item_num, 'text': item_text}
                    if sub_items:
                        item_data['sub_items'] = sub_items
                    items.append(item_data)
                paragraphs.append({'num': para_num, 'text': para_text, 'items': items})
            full_text = elem_full_text(article_elem).strip()
            articles.append({
                'num': num, 'caption': caption, 'title': title,
                'paragraphs': paragraphs, 'full_text': full_text,
                'source': 'main',
            })
    
    return articles


def extract_tables_from_xml(law_full_text) -> list[dict]:
    """
    Extract appended tables (別表) from law XML.
    Returns list of dicts with keys: title, related_articles, rows
    Each row is a list of cell texts.
    """
    xml_string = None
    if isinstance(law_full_text, str):
        try:
            xml_bytes = base64.b64decode(law_full_text)
            xml_string = xml_bytes.decode('utf-8')
        except Exception:
            return []
    if not xml_string:
        return []

    try:
        root = ET.fromstring(xml_string)
    except ET.ParseError:
        return []

    def elem_text(elem):
        text = elem.text or ''
        for child in elem:
            text += elem_text(child)
            text += child.tail or ''
        return text.strip()

    tables = []
    for appdx in root.iter('AppdxTable'):
        title_elem = appdx.find('AppdxTableTitle')
        title = elem_text(title_elem) if title_elem is not None else ""
        related = appdx.find('RelatedArticleNum')
        related_text = elem_text(related) if related is not None else ""
        
        # Also handle Remarks (備考)
        remarks = []
        
        for table_struct in appdx.iter('TableStruct'):
            rows = []
            for table in table_struct.iter('Table'):
                for tr in table.iter('TableRow'):
                    cells = []
                    for tc in tr.iter('TableColumn'):
                        cells.append(elem_text(tc))
                    if cells:
                        rows.append(cells)
            
            # Extract remarks from TableStruct
            for remark in table_struct.iter('Remarks'):
                remarks.append(elem_text(remark))
            
            if rows:
                table_data = {
                    "title": title,
                    "related_articles": related_text,
                    "rows": rows,
                }
                if remarks:
                    table_data["remarks"] = remarks
                tables.append(table_data)

    # Also handle AppdxStyle (別記様式 etc.)
    for appdx_style in root.iter('AppdxStyle'):
        title_elem = appdx_style.find('AppdxStyleTitle')
        if title_elem is not None:
            tables.append({
                "title": elem_text(title_elem),
                "related_articles": "",
                "rows": [],
                "note": "Style/form appendix (様式) - no table data"
            })

    return tables


def _normalize_number(s: str) -> str:
    """Normalize full-width digits, various separators for article numbers."""
    if not s:
        return s
    # Full-width → half-width digits
    s = s.translate(str.maketrans('０１２３４５６７８９', '0123456789'))
    return s.strip()


def _normalize_article_number(article_input: str) -> str:
    """
    Normalize article number to canonical form used by XML Num attribute.
    e.g., "6の2" → "6_2", "6-2" → "6_2", "６の２" → "6_2"
    """
    s = _normalize_number(article_input)
    # Remove 第...条 wrapping
    s = re.sub(r'^第', '', s)
    # Remove 条 but preserve what follows if it's a separator+number (e.g., 条の2 → の2)
    s = re.sub(r'条(?=[のノー─\-])', '', s)  # 条の2 → の2
    s = re.sub(r'条$', '', s)                 # trailing 条
    s = re.sub(r'条(?=\s|$)', '', s)          # 条 followed by space/end
    # Normalize separators: の, -, ─, ー → _
    s = re.sub(r'[のノー─\-]', '_', s)
    return s


def find_article_in_xml(law_full_text, article_number: str, 
                         paragraph: str = None, item: str = None,
                         main_only: bool = True) -> list[dict]:
    """
    Find specific article(s) from XML by Num attribute, with optional 
    paragraph (項) and item (号) filtering.
    
    Args:
        law_full_text: Raw law_full_text from API response
        article_number: Article number (e.g., "12", "12の2", "第12条第5項")
        paragraph: Optional paragraph number to filter (e.g., "5")
        item: Optional item number to filter (e.g., "1")
        main_only: If True, only return articles from 本則 (exclude 附則)
    
    Returns:
        List of matched article dicts with filtered content
    """
    articles = extract_articles_from_xml(law_full_text)
    if not articles:
        return []
    
    # Normalize inputs
    clean = _normalize_number(article_number.strip())
    clean = re.sub(r'^第', '', clean)
    if paragraph:
        paragraph = _normalize_number(paragraph)
    if item:
        item = _normalize_number(item)
    
    # Extract paragraph/item from compound patterns like "第12条第5項第1号"
    art_match = re.match(
        r'([0-9]+(?:[のノー─_\-][0-9]+)*)\s*条?\s*(?:第([0-9]+)項)?\s*(?:第([0-9]+)号)?',
        clean
    )
    
    target_num = None
    if art_match:
        target_num = art_match.group(1)
        if art_match.group(2) and not paragraph:
            paragraph = art_match.group(2)
        if art_match.group(3) and not item:
            item = art_match.group(3)
    
    if not target_num:
        # Fallback: extract first number sequence
        m = re.search(r'([0-9]+)', clean)
        if m:
            target_num = m.group(1)
    
    if not target_num:
        return []
    
    # Normalize target to underscore format (e.g., "6の2" → "6_2")
    target_normalized = re.sub(r'[のノー─\-]', '_', target_num)
    
    # Match articles by Num attribute
    matched = []
    for article in articles:
        # Filter by source (本則 vs 附則)
        if main_only and article.get('source') == 'suppl':
            continue
        
        a_num = article['num']
        
        # Try various matching strategies
        # API uses formats like "12", "12_2" for 第12条の2
        target_variants = [
            target_normalized,
            target_num,
            target_num.replace('の', '_'),
        ]
        # Remove duplicates
        target_variants = list(dict.fromkeys(target_variants))
        
        if a_num in target_variants:
            result = {
                'num': article['num'],
                'caption': article['caption'],
                'title': article['title'],
                'paragraphs': article['paragraphs'][:],
                'full_text': article['full_text']
            }
            
            # Filter by paragraph if specified
            if paragraph:
                filtered_paras = [
                    p for p in result['paragraphs'] 
                    if p['num'] == paragraph
                ]
                if filtered_paras:
                    result['paragraphs'] = filtered_paras
                    # Filter items within paragraphs if specified
                    if item:
                        for p in result['paragraphs']:
                            p['items'] = [
                                i for i in p['items'] 
                                if i['num'] == item
                            ]
                    # Reconstruct full_text from filtered paragraphs
                    result['full_text'] = '\n'.join(
                        p['text'] for p in result['paragraphs']
                    )
                    result['filtered'] = True
                    if paragraph:
                        result['filtered_paragraph'] = paragraph
                    if item:
                        result['filtered_item'] = item
            
            matched.append(result)
    
    # Annotate matched articles with structural metadata
    for m in matched:
        m['annotations'] = annotate_article_structure(m.get('full_text', ''), m.get('paragraphs', []))
    
    return matched


def annotate_article_structure(full_text: str, paragraphs: list) -> dict:
    """
    Analyze article text and produce structural annotations to help LLMs
    correctly interpret legal provisions.
    
    Detects:
    - provisos (ただし書き) and their scope
    - exclusion clauses (「...を除く」)
    - subject of each paragraph (誰が/何が)
    - delegation references (政令で定める/内閣府令で定める)
    - 「のほか」additional requirement patterns
    """
    annotations = {}
    
    # 1. Detect provisos (ただし書き)
    provisos = []
    for i, para in enumerate(paragraphs):
        text = para.get('text', '')
        proviso_matches = list(re.finditer(r'ただし、(.+?)(?:この限りでない|その定めに従う|この限りでは?ない)', text))
        for pm in proviso_matches:
            provisos.append({
                'paragraph': para.get('num', str(i+1)),
                'proviso_text': pm.group(0),
                'proviso_condition': pm.group(1).strip(),
            })
        # Also check items for provisos
        for item in para.get('items', []):
            item_text = item.get('text', '')
            for pm in re.finditer(r'ただし、(.+?)(?:この限りでない|その定めに従う)', item_text):
                provisos.append({
                    'paragraph': para.get('num', str(i+1)),
                    'item': item.get('num', ''),
                    'proviso_text': pm.group(0),
                    'proviso_condition': pm.group(1).strip(),
                })
    if provisos:
        annotations['provisos'] = provisos
    
    # 2. Detect exclusion clauses (「...を除く」)
    exclusions = []
    for exc_m in re.finditer(r'((?:[^。、（]+?)(?:を除く|を除き|から除く))', full_text):
        exclusions.append(exc_m.group(1).strip())
    if exclusions:
        annotations['exclusions'] = exclusions
    
    # 3. Detect subject of each paragraph
    subjects = []
    for i, para in enumerate(paragraphs):
        text = para.get('text', '')
        # Match patterns like 「Xは、」at the start of paragraph
        subj_m = re.match(r'^(.{2,40}?)は、', text)
        if subj_m:
            subjects.append({
                'paragraph': para.get('num', str(i+1)),
                'subject': subj_m.group(1).strip()
            })
    if subjects:
        annotations['subjects'] = subjects
    
    # 4. Detect delegation references
    delegations = []
    for del_m in re.finditer(r'(政令|内閣府令|厚生労働省令|内閣総理大臣)(?:で|が|の)定める', full_text):
        delegations.append(del_m.group(0))
    if delegations:
        annotations['delegations'] = list(set(delegations))
    
    # 5. Detect 「のほか」additional requirement patterns
    nohoka = []
    for nh_m in re.finditer(r'(.{5,60}?)のほか', full_text):
        nohoka.append(nh_m.group(0))
    if nohoka:
        annotations['additional_requirements'] = nohoka
    
    # 6. Detect numeric thresholds (期間・金額等)
    thresholds = []
    for th_m in re.finditer(r'((?:六月|一年|二年|三年|五年|十年|三十日|六十日|二十日|百分の[一-九十百千万]+(?:・[一-九十百千万]+)?|[一-九十百千万]+円|[一-九十百千万]+万円)(?:以内|以上|以下|を超え(?:ない|て))?)', full_text):
        thresholds.append(th_m.group(0))
    if thresholds:
        annotations['thresholds'] = list(set(thresholds))
    
    return annotations


def arabic_to_kanji(num_str: str) -> str:
    """Convert Arabic numbers to Kanji for Japanese legal text."""
    if not num_str.isdigit():
        return num_str

    num = int(num_str)
    if num == 0: return '〇'
    if 1 <= num <= 9: return '一二三四五六七八九'[num-1]
    if 10 <= num <= 19:
        return '十' if num == 10 else '十' + '一二三四五六七八九'[num%10-1]
    if 20 <= num <= 99:
        tens = '二三四五六七八九'[num//10-2] + '十'
        ones = '' if num % 10 == 0 else '一二三四五六七八九'[num%10-1]
        return tens + ones
    if 100 <= num <= 999:
        hundreds = '百' if num // 100 == 1 else '一二三四五六七八九'[num//100-1] + '百'
        remainder = num % 100
        if remainder == 0: return hundreds
        if remainder < 10: return hundreds + '一二三四五六七八九'[remainder-1]
        return hundreds + arabic_to_kanji(str(remainder))

    return num_str  # Fallback for large numbers


def kanji_to_arabic(kanji: str) -> str:
    """Convert kanji number to arabic number string (module-level version)."""
    kanji_map = {"一": 1, "二": 2, "三": 3, "四": 4, "五": 5,
                "六": 6, "七": 7, "八": 8, "九": 9}
    if not kanji:
        return "0"
    # Handle hundreds
    if "百" in kanji:
        parts = kanji.split("百")
        hundreds = kanji_map.get(parts[0], 1) if parts[0] else 1
        remainder = parts[1] if len(parts) > 1 and parts[1] else ""
        if remainder:
            return str(hundreds * 100 + int(kanji_to_arabic(remainder)))
        return str(hundreds * 100)
    # Handle tens
    if "十" in kanji:
        parts = kanji.split("十")
        tens = kanji_map.get(parts[0], 1) if parts[0] else 1
        ones = kanji_map.get(parts[1], 0) if len(parts) > 1 and parts[1] else 0
        return str(tens * 10 + ones)
    return str(kanji_map.get(kanji, 0))


def generate_search_patterns(article_input: str) -> list[str]:
    """Generate comprehensive search patterns for article numbers."""
    # Extract main number and patterns
    main_match = re.search(r'(\d+)', article_input)
    if not main_match:
        return [article_input]

    main_num = main_match.group(1)
    kanji_num = arabic_to_kanji(main_num)

    patterns = [
        f"第{kanji_num}条",     # e.g., 第百九十二条
        f"{kanji_num}条",      # e.g., 百九十二条
        f"第{main_num}条",     # e.g., 第192条
    ]

    # Handle 条の2, 条の2の3 patterns
    if 'の' in article_input:
        no_matches = re.findall(r'の(\d+)', article_input)
        if no_matches:
            # Build suffix like "の二の三"
            suffix_kanji = ''.join(f'の{arabic_to_kanji(n)}' for n in no_matches)
            suffix_arabic = ''.join(f'の{n}' for n in no_matches)
            patterns.extend([
                f"第{kanji_num}条{suffix_kanji}",  # e.g., 第九条の二の二
                f"{kanji_num}条{suffix_kanji}",   # e.g., 九条の二の二
                f"第{main_num}条{suffix_arabic}",  # e.g., 第9条の2の2
            ])

    # Handle 項・号 patterns
    if '項' in article_input:
        kou_match = re.search(r'第(\d+)項', article_input)
        if kou_match:
            kou_num = kou_match.group(1)
            kou_kanji = arabic_to_kanji(kou_num)
            patterns.extend([
                f"第{kanji_num}条第{kou_kanji}項",
                f"第{main_num}条第{kou_num}項",
            ])

    if '号' in article_input:
        gou_match = re.search(r'第(\d+)号', article_input)
        if gou_match:
            gou_num = gou_match.group(1)
            gou_kanji = arabic_to_kanji(gou_num)
            patterns.extend([
                f"第{kanji_num}条第{gou_kanji}号",
                f"第{main_num}条第{gou_num}号",
            ])

    # Add original patterns
    patterns.extend([
        f"第{article_input}条",
        f"{article_input}条",
        article_input
    ])

    # Remove duplicates while preserving order
    return list(dict.fromkeys(patterns))

async def smart_law_lookup(law_name: str) -> Optional[str]:
    """Smart law lookup with formal name verification and direct mapping fallback to search."""
    law_name_clean = law_name.strip()
    original_input = law_name_clean

    # Step 1: Check for aliases and convert to formal name
    if law_name_clean in LAW_ALIASES:
        formal_name = LAW_ALIASES[law_name_clean]
        logger.info(f"Alias conversion: '{original_input}' -> '{formal_name}'")
        law_name_clean = formal_name

    # Step 2: Check cache first
    cache_key = cache_manager.get_cache_key(law_name_clean)
    cached_result = cache_manager.law_lookup_cache.get(cache_key)
    if cached_result:
        logger.info(f"Cache hit for law lookup: {law_name_clean} -> {cached_result}")
        return cached_result

    # Step 3: Check direct mapping with formal name
    if law_name_clean in BASIC_LAWS:
        result = BASIC_LAWS[law_name_clean]
        logger.info(f"Direct mapping: {law_name_clean} -> {result}")
        # Cache the result
        cache_manager.law_lookup_cache.put(cache_key, result)
        return result

    # Step 4: Intelligent search for unknown laws
    # Detect law type from name
    search_law_type = "Act"
    if "施行令" in law_name_clean:
        search_law_type = "CabinetOrder"
    elif "施行規則" in law_name_clean:
        search_law_type = "MinisterialOrdinance"
    elif "省令" in law_name_clean:
        search_law_type = "MinisterialOrdinance"
    elif "内閣府令" in law_name_clean:
        search_law_type = "MinisterialOrdinance"
    elif "規則" in law_name_clean and "法" not in law_name_clean:
        search_law_type = "MinisterialOrdinance"
    elif "施行法" in law_name_clean:
        search_law_type = "Act"  # 施行法 is still an Act
    
    async with await get_http_client() as client:
        response = await client.get("/laws", params={
            "law_title": law_name_clean,
            "law_type": search_law_type,
            "limit": 20
        })
        response.raise_for_status()

        data = json.loads(response.text)
        laws = data.get("laws", [])

        if not laws and search_law_type == "Act":
            # Retry with MinisterialOrdinance for possible 府令/省令
            for fallback_type in ["MinisterialOrdinance", "CabinetOrder"]:
                response2 = await client.get("/laws", params={
                    "law_title": law_name_clean,
                    "law_type": fallback_type,
                    "limit": 20
                })
                response2.raise_for_status()
                data2 = json.loads(response2.text)
                laws2 = data2.get("laws", [])
                if laws2:
                    logger.info(f"Fallback search found {len(laws2)} laws as {fallback_type} for '{law_name_clean}'")
                    laws = laws2
                    break
        
        if not laws:
            logger.warning(f"No laws found for search term: {law_name_clean} (original: {original_input})")
            return None

        # Log search results for transparency
        logger.info(f"Found {len(laws)} candidate laws for '{law_name_clean}' (original: '{original_input}')")
        for i, law in enumerate(laws[:3]):  # Log top 3 candidates
            law_info = law.get('law_info', {})
            logger.info(f"  Candidate {i+1}: {law_info.get('law_title', 'N/A')} ({law_info.get('law_num', 'N/A')})")

        # Smart scoring for best law selection
        # Priority: exact title match > base law (oldest) > amendments (newer)
        def score_law(law_entry):
            law_info = law_entry.get("law_info", {})
            law_num = law_info.get("law_num", "")
            law_title = law_info.get("law_title", "")
            score = 0

            # Title match bonus (highest priority)
            # Exact match with search term
            if law_title == law_name_clean:
                score += 100000
            # Title starts with search term
            elif law_title.startswith(law_name_clean):
                score += 50000
            # Search term in title (partial match)
            elif law_name_clean in law_title:
                score += 10000
            # Penalty for "一部を改正する" (amendment laws) - these are NOT the base law
            if "一部を改正する" in law_title or "改正" in law_title:
                score -= 80000

            # For base laws, prefer the OLDEST (original) law
            # Base laws like 廃棄物処理法施行規則 (昭和46年) should beat 
            # amendment ordinances (令和2年) 
            if "令和" in law_num: score += 100
            elif "平成" in law_num: score += 200
            elif "昭和" in law_num: score += 300
            elif "大正" in law_num: score += 400
            elif "明治" in law_num: score += 500

            # Prefer shorter law numbers (basic laws tend to have shorter numbers)
            if len(law_num) < 25: score += 50

            return score

        best_law = max(laws, key=score_law)
        selected_law_num = best_law.get("law_info", {}).get("law_num")
        selected_law_title = best_law.get("law_info", {}).get("law_title")

        logger.info(f"Selected law: {selected_law_title} ({selected_law_num}) for search term '{law_name_clean}' (original: '{original_input}')")

        # Cache the result
        if selected_law_num:
            cache_manager.law_lookup_cache.put(cache_key, selected_law_num)

        return selected_law_num

@mcp.tool
async def find_law_article(law_name: str, article_number: str, paragraph: str = "", item: str = "", main_only: bool = True, ctx: Context = None) -> dict:
    """
    Find a specific article in Japanese law (ULTRA SMART & FAST)

    Supports 16+ major laws with direct mapping for instant access.
    Handles complex patterns like 条の2, 項, 号 automatically.
    Uses XML structure-based extraction for precise matching.
    Full-width digits (０-９) are automatically normalized.

    Args:
        law_name: Law name (e.g., "民法", "会社法", "憲法")
        article_number: Article number (e.g., "192", "325条の3", "第9条第2項"). Full-width digits OK.
        paragraph: Optional paragraph number (項) to filter (e.g., "5"). Full-width OK.
        item: Optional item number (号) to filter (e.g., "1"). Full-width OK.
        main_only: If True (default), only search 本則 (main provisions), excluding 附則 (supplementary). Set False to include 附則.
        ctx: FastMCP context for logging and progress reporting

    Returns:
        Dict with found article content and legal analysis metadata
    """
    if not law_name or not law_name.strip():
        raise ToolError("law_name is required")
    if not article_number or not article_number.strip():
        raise ToolError("article_number is required")

    try:
        # Log the search request
        await ctx.info(f"Searching for article {article_number} in {law_name}")
        
        # Cleanup cache if needed
        cache_manager.cleanup_if_needed()

        # Step 1: Smart law lookup with formal name verification
        original_law_input = law_name
        formal_law_name = law_name
        name_conversion_applied = False

        # Check if alias conversion is needed
        if law_name.strip() in LAW_ALIASES:
            formal_law_name = LAW_ALIASES[law_name.strip()]
            name_conversion_applied = True
            await ctx.debug(f"Alias conversion: {law_name} → {formal_law_name}")

        law_num = await smart_law_lookup(law_name)
        if not law_num:
            await ctx.error(f"Law '{law_name}' not found")
            raise ToolError(f"Law '{law_name}' not found")

        # Step 2: Get law text with XML format
        # Normalize full-width digits in all numeric inputs
        para_filter = _normalize_number(paragraph) if paragraph and paragraph.strip() else None
        item_filter = _normalize_number(item) if item and item.strip() else None
        
        # Check content cache: memory → disk → API
        cached_content = cache_manager.law_content_cache.get(law_num)
        if cached_content:
            data = cached_content
        else:
            disk_content = cache_manager.disk_cache.get(f"law_data:{law_num}")
            if disk_content:
                data = disk_content
                cache_manager.law_content_cache.put(law_num, data)
            else:
                async with await get_http_client() as client:
                    response = await client.get(f"/law_data/{law_num}", params={
                        "law_full_text_format": "xml"
                    })
                    response.raise_for_status()
                    data = json.loads(response.text)
                    cache_manager.law_content_cache.put(law_num, data)
                    cache_manager.disk_cache.put(f"law_data:{law_num}", data)

        law_full_text = data.get('law_full_text', {})
        
        # Step 3: XML structure-based article search (primary method)
        xml_matches = find_article_in_xml(
            law_full_text, article_number, 
            paragraph=para_filter, item=item_filter,
            main_only=main_only
        )
        
        matches = []
        xml_used = False
        
        if xml_matches:
            xml_used = True
            for m in xml_matches:
                matches.append(m['full_text'])
        
        # Step 3b: Fallback to regex-based search if XML parsing failed
        if not matches:
            extracted_text = extract_text_from_xml(law_full_text)
            patterns = generate_search_patterns(article_number)

            for pattern in patterns:
                article_pattern = re.escape(pattern)
                all_matches = []

                for match in re.finditer(article_pattern, extracted_text):
                    pos = match.start()
                    context_before = extracted_text[max(0, pos-50):pos]
                    context_after = extracted_text[pos:pos+100]
                    if re.search(r'第\d+条.*第\d+条', context_before + context_after):
                        continue

                    is_actual_article = False
                    if re.search(rf'{article_pattern}\s*\n\s*\n\s*\n\s*', extracted_text[pos:pos+200]):
                        is_actual_article = True
                    if context_before.endswith('\n              ') or context_before.endswith('            '):
                        is_actual_article = True
                    if re.search(rf'{article_pattern}\s*\n.*?\n.*?\n.*?[あ-ん]', extracted_text[pos:pos+300], re.DOTALL):
                        is_actual_article = True

                    if not is_actual_article:
                        continue

                    strategies = [
                        f"{article_pattern}.*?(?=第[一二三四五六七八九十百千]+条)",
                        f"{article_pattern}[^第]*",
                        f"{article_pattern}.*?(?=（[^）]*）\\s*第)",
                        f"{article_pattern}.{{0,2000}}"
                    ]

                    context_start = max(0, pos - 20)
                    context_end = min(len(extracted_text), pos + 3000)
                    context = extracted_text[context_start:context_end]

                    for strategy in strategies:
                        matches_found = re.findall(strategy, context, re.DOTALL | re.MULTILINE)
                        if matches_found:
                            candidate = matches_found[0].strip()
                            if len(candidate) > 50:
                                content_score = 10
                                if len(candidate) > 200: content_score += 3
                                elif len(candidate) > 100: content_score += 2
                                if '。' in candidate: content_score += 3
                                if '、' in candidate: content_score += 2
                                if re.search(r'[あ-ん]+', candidate): content_score += 3
                                if '―' in candidate: content_score -= 5
                                if candidate.count('第') > 3: content_score -= 2
                                all_matches.append((content_score, candidate, pos))
                                break

                all_matches.sort(key=lambda x: x[0], reverse=True)

                for score, candidate, pos in all_matches[:3]:
                    if score > 10:
                        clean_match = candidate.strip()
                        if len(clean_match) > 30 and clean_match not in matches:
                            if clean_match.endswith(('。', '）', '）。', '号', '項', '条')):
                                matches.append(clean_match)
                            else:
                                for ending in ['。', '）。', '号。', '項。']:
                                    if ending in clean_match:
                                        last_pos = clean_match.rfind(ending)
                                        if last_pos > len(clean_match) * 0.7:
                                            matches.append(clean_match[:last_pos + len(ending)])
                                            break
                                else:
                                    if len(clean_match) > 100:
                                        matches.append(clean_match)

        # Format result with formal name verification info
        law_info_data = data.get('law_info', {})
        actual_law_title = law_info_data.get('law_title', formal_law_name)

        result = {
            "law_info": law_info_data,
            "search_law_name": original_law_input,
            "formal_law_name_used": formal_law_name,
            "actual_law_title": actual_law_title,
            "name_conversion_applied": name_conversion_applied,
            "search_article": article_number,
            "found_law": actual_law_title,
            "law_number": law_num,
            "matches_found": len(matches),
            "articles": matches[:3] if matches else [],
            "extraction_method": "xml_structure" if xml_used else "regex_fallback",
            "note": f"Searched for article '{article_number}'{' paragraph ' + para_filter if para_filter else ''}{' item ' + item_filter if item_filter else ''} in '{actual_law_title}'{' (converted from: ' + original_law_input + ')' if name_conversion_applied else ''}",
            "legal_analysis_instruction": prompt_loader.get_legal_analysis_instruction()
        }
        
        # Include structured XML data if available
        if xml_used and xml_matches:
            result["structured_articles"] = xml_matches

        if not matches:
            # Smart suggestions for missing articles
            main_num = re.search(r'(\d+)', article_number)
            if main_num:
                article_num = main_num.group(1)
                kanji_num = arabic_to_kanji(article_num)
                basic_patterns = [f"第{kanji_num}条", f"第{article_num}条"]

                extracted_text = extract_text_from_xml(law_full_text)
                basic_found = any(re.search(re.escape(p), extracted_text) for p in basic_patterns)

                if basic_found:
                    if 'の' in article_number:
                        result["suggestion"] = f"Article {article_num} exists, but the specified 'の' variation may not exist."
                    elif '項' in article_number or '号' in article_number:
                        result["suggestion"] = f"Article {article_num} exists, but the specified paragraph (項) or subparagraph (号) may not exist."
                    else:
                        result["suggestion"] = f"Article found with different formatting. Try searching for just '{article_num}'."
                else:
                    result["suggestion"] = f"Article {article_number} not found in {law_name}. Please verify the article number."

            result["search_patterns_used"] = generate_search_patterns(article_number)[:5]

        await ctx.info(f"Successfully found article {article_number} in {result.get('actual_law_title', law_name)}")
        return result

    except ToolError:
        # Re-raise ToolError to send proper error to client
        raise
    except Exception as e:
        logger.error(f"Find law article error: {e}")
        await ctx.error(f"Search failed: {str(e)}")
        raise ToolError(f"Search failed: {str(e)}")

@mcp.tool
async def search_laws(
    law_title: str = "",
    law_type: str = "",
    law_num: str = "",
    limit: int = 10,
    offset: int = 0,
    ctx: Context = None
) -> dict:
    """
    Search Japanese laws with smart filtering

    Args:
        law_title: Law title (partial match)
        law_type: Law type (Act, CabinetOrder, etc.)
        law_num: Law number (partial match)
        limit: Maximum results (1-500)
        offset: Starting position
        ctx: FastMCP context for logging

    Returns:
        Dict with search results
    """
    # Input validation
    if limit < 1 or limit > 500:
        raise ToolError("limit must be between 1 and 500")
    if offset < 0:
        raise ToolError("offset must be 0 or greater")
    
    if ctx:
        await ctx.info(f"Searching laws with title='{law_title}', type='{law_type}', limit={limit}")

    params = {"limit": limit, "offset": offset}
    if law_title: params["law_title"] = law_title
    if law_type: params["law_type"] = law_type
    if law_num: params["law_num"] = law_num

    try:
        async with await get_http_client() as client:
            response = await client.get("/laws", params=params)
            response.raise_for_status()
            
            # Parse JSON and return dict for FastMCP auto-serialization
            result = json.loads(response.text)
            if ctx:
                law_count = len(result.get("laws", []))
                await ctx.info(f"Found {law_count} laws matching search criteria")
            return result
    except Exception as e:
        logger.error(f"Search laws error: {e}")
        if ctx:
            await ctx.error(f"Search failed: {str(e)}")
        raise ToolError(f"Search failed: {str(e)}")

@mcp.tool
async def search_laws_by_keyword(keyword: str, law_type: str = "", limit: int = 5, ctx: Context = None) -> dict:
    """
    Full-text keyword search in Japanese laws

    Args:
        keyword: Search keyword (required)
        law_type: Law type filter (optional)
        limit: Maximum results (1-20)
        ctx: FastMCP context for logging

    Returns:
        Dict with search results
    """
    if not keyword or not keyword.strip():
        raise ToolError("keyword is required")
    if limit < 1 or limit > 20:
        raise ToolError("limit must be between 1 and 20")
    
    if ctx:
        await ctx.info(f"Searching for keyword: '{keyword}' with limit={limit}")

    params = {"keyword": keyword.strip(), "limit": limit}
    if law_type: params["law_type"] = law_type

    try:
        async with await get_http_client() as client:
            response = await client.get("/keyword", params=params)
            response.raise_for_status()
            
            # Parse JSON and return dict for FastMCP auto-serialization
            result = json.loads(response.text)
            if ctx:
                result_count = len(result.get("laws", []))
                await ctx.info(f"Found {result_count} laws containing keyword '{keyword}'")
            return result
    except Exception as e:
        logger.error(f"Keyword search error: {e}")
        if ctx:
            await ctx.error(f"Keyword search failed: {str(e)}")
        raise ToolError(f"Keyword search failed: {str(e)}")

@mcp.tool
async def get_law_content(law_id: str = "", law_num: str = "", response_format: str = "json", elm: str = "", ctx: Context = None) -> dict:
    """
    Get law content (optimized per API spec with size limits)

    Args:
        law_id: Law ID
        law_num: Law number
        response_format: "json" or "xml"
        elm: Element to retrieve (currently disabled due to API 400 errors)
        ctx: FastMCP context for logging

    Returns:
        Dict with law content. For large laws (>800KB), returns summary with recommendation to use find_law_article for specific articles.

    Note:
        - elm parameter is currently disabled due to e-Gov API 400 errors
        - Large laws like Company Law (会社法) will return a summary instead of full text
        - Use find_law_article tool for specific article searches in large laws
    """
    if not law_id and not law_num:
        raise ToolError("Either law_id or law_num must be specified")
    if response_format not in ["json", "xml"]:
        raise ToolError("response_format must be 'json' or 'xml'")

    law_identifier = law_id if law_id else law_num
    
    if ctx:
        await ctx.info(f"Getting law content for {law_identifier} in {response_format} format")
    params = {}
    if response_format == "xml":
        params["law_full_text_format"] = "xml"

    # Note: elm parameter causes 400 errors with current e-Gov API
    # Commented out to avoid API errors
    # if elm:
    #     params["elm"] = elm

    try:
        async with await get_http_client() as client:
            response = await client.get(f"/law_data/{law_identifier}", params=params)
            response.raise_for_status()

            if response_format == "json":
                # Format JSON response for better readability
                data = json.loads(response.text)

                # Check response size and truncate if necessary
                response_str = json.dumps(data, ensure_ascii=False, indent=2)
                if len(response_str) > 800000:  # 800KB limit (留余裕給其他數據)
                    # Create summary instead of full text for large laws
                    law_info = data.get('law_info', {})
                    summary = {
                        "law_info": law_info,
                        "warning": "法令全文が長すぎるため、概要のみ表示しています。",
                        "recommendation": "特定の条文を検索する場合は find_law_article ツールを使用してください。",
                        "law_stats": {
                            "original_size_bytes": len(response_str),
                            "law_title": law_info.get('law_title', ''),
                            "law_num": law_info.get('law_num', ''),
                            "promulgation_date": law_info.get('promulgation_date', '')
                        }
                    }

                    # Try to include table of contents if available
                    law_full_text = data.get('law_full_text', {})
                    if isinstance(law_full_text, dict):
                        # Extract structure information
                        if 'chapters' in str(law_full_text).lower() or '章' in str(law_full_text):
                            summary["structure_note"] = "この法令は章立て構造を持っています。"
                        if 'sections' in str(law_full_text).lower() or '節' in str(law_full_text):
                            summary["structure_note"] = summary.get("structure_note", "") + " 節による区分があります。"

                    if ctx:
                        await ctx.info(f"Large law content truncated to summary ({len(response_str)} bytes)")
                    return summary

                # For smaller responses, add readable text
                law_full_text = data.get('law_full_text', {})
                if isinstance(law_full_text, str):
                    # Extract readable text from XML
                    data['law_full_text_readable'] = extract_text_from_xml(law_full_text)

                if ctx:
                    await ctx.info(f"Successfully retrieved law content ({len(response_str)} bytes)")
                return data
            else:
                # For XML format, check size and truncate if needed
                if len(response.text) > 800000:
                    if ctx:
                        await ctx.info(f"Large XML content truncated ({len(response.text)} bytes)")
                    return {
                        "format": "xml",
                        "warning": "法令全文が長すぎるため、概要のみ表示しています。",
                        "recommendation": "特定の条文を検索する場合は find_law_article ツールを使用してください。",
                        "original_size_bytes": len(response.text),
                        "truncated_content": response.text[:1000] + "..."
                    }
                else:
                    if ctx:
                        await ctx.info(f"Successfully retrieved XML content ({len(response.text)} bytes)")
                    return {
                        "format": "xml",
                        "content": response.text
                    }

    except Exception as e:
        logger.error(f"Get law content error: {e}")
        if ctx:
            await ctx.error(f"Failed to get law content: {str(e)}")
        raise ToolError(f"Failed to get law content: {str(e)}")

@mcp.tool
async def batch_find_articles(law_article_pairs: str, ctx: Context) -> dict:
    """
    Batch find multiple law articles efficiently

    Args:
        law_article_pairs: JSON string with law-article pairs, e.g. '[{"law":"民法","article":"192"},{"law":"憲法","article":"9"}]'
        ctx: FastMCP context for logging

    Returns:
        Dict with batch results and performance stats
    """
    try:
        pairs = json.loads(law_article_pairs)
        if not isinstance(pairs, list):
            raise ToolError("law_article_pairs must be a JSON array")
            
        await ctx.info(f"Starting batch search for {len(pairs)} law-article pairs")

        results = []
        cache_hits = 0
        api_calls = 0

        async with await get_http_client() as client:
            # Prefetch if cache is empty
            if cache_manager.law_lookup_cache.size() == 0:
                await cache_manager.prefetch_common_articles(client)

            for i, pair in enumerate(pairs):
                if not isinstance(pair, dict) or "law" not in pair or "article" not in pair:
                    results.append({"error": "Invalid pair format"})
                    continue
                    
                await ctx.debug(f"Processing pair {i+1}/{len(pairs)}: {pair['law']} - {pair['article']}")

                law_name = pair["law"]
                article_number = pair["article"]

                # Check cache first
                cache_key = cache_manager.get_cache_key(law_name, article_number)
                cached_result = cache_manager.article_cache.get(cache_key)

                if cached_result:
                    results.append(cached_result)
                    cache_hits += 1
                else:
                    # Perform law article search directly
                    try:
                        # Internal article search logic (similar to find_law_article)
                        if not law_name or not law_name.strip():
                            results.append({"error": "law_name is required"})
                            continue
                        if not article_number or not article_number.strip():
                            results.append({"error": "article_number is required"})
                            continue
                            
                        # Use smart_law_lookup to get law number
                        law_num = await smart_law_lookup(law_name)
                        if not law_num:
                            results.append({"error": f"Law '{law_name}' not found"})
                            continue
                            
                        # Get law content (memory → disk → API)
                        cached_content = cache_manager.law_content_cache.get(law_num)
                        if cached_content:
                            data = cached_content
                        else:
                            disk_content = cache_manager.disk_cache.get(f"law_data:{law_num}")
                            if disk_content:
                                data = disk_content
                                cache_manager.law_content_cache.put(law_num, data)
                            else:
                                response = await client.get(f"/law_data/{law_num}", params={
                                    "law_full_text_format": "xml"
                                })
                                response.raise_for_status()
                                data = json.loads(response.text)
                                cache_manager.law_content_cache.put(law_num, data)
                                cache_manager.disk_cache.put(f"law_data:{law_num}", data)
                                api_calls += 1
                        
                        law_full_text = data.get('law_full_text', {})
                        
                        # Use XML structure-based extraction (same as find_law_article)
                        xml_matches = find_article_in_xml(law_full_text, article_number)
                        
                        law_info_data = data.get('law_info', {})
                        if xml_matches:
                            result = {
                                "law_info": law_info_data,
                                "search_law_name": law_name,
                                "search_article": article_number,
                                "law_number": law_num,
                                "found_article": xml_matches[0]['full_text'],
                                "matches_found": len(xml_matches),
                                "extraction_method": "xml_structure",
                            }
                        else:
                            # Regex fallback
                            extracted_text = extract_text_from_xml(law_full_text)
                            patterns = generate_search_patterns(article_number)
                            found_match = None
                            for pattern in patterns[:3]:
                                article_pattern = re.escape(pattern)
                                matches = re.findall(f"{article_pattern}.{{0,500}}", extracted_text, re.DOTALL)
                                if matches:
                                    found_match = matches[0].strip()
                                    break
                            result = {
                                "law_info": law_info_data,
                                "search_law_name": law_name,
                                "search_article": article_number,
                                "law_number": law_num,
                                "found_article": found_match,
                                "matches_found": 1 if found_match else 0,
                                "extraction_method": "regex_fallback",
                            }
                        
                        results.append(result)
                        cache_manager.article_cache.put(cache_key, result)
                            
                    except Exception as e:
                        await ctx.error(f"Batch search item failed: {str(e)}")
                        results.append({"error": str(e)})
                        api_calls += 1

        batch_result = {
            "results": results,
            "performance_stats": {
                "total_requests": len(pairs),
                "cache_hits": cache_hits,
                "api_calls": api_calls,
                "cache_hit_rate": f"{(cache_hits / len(pairs) * 100):.1f}%" if pairs else "0%"
            }
        }
        
        await ctx.info(f"Batch search completed: {cache_hits} cache hits, {api_calls} API calls")
        return batch_result

    except ToolError:
        raise
    except Exception as e:
        logger.error(f"Batch find articles error: {e}")
        await ctx.error(f"Batch search failed: {str(e)}")
        raise ToolError(f"Batch search failed: {str(e)}")

@mcp.tool
async def prefetch_common_laws(ctx: Context) -> dict:
    """
    Prefetch commonly accessed laws for better performance

    Args:
        ctx: FastMCP context for logging
        
    Returns:
        Dict with prefetch results and cache status
    """
    try:
        await ctx.info("Starting prefetch of common laws...")
        
        async with await get_http_client() as client:
            await cache_manager.prefetch_common_articles(client)

            result = {
                "status": "success",
                "message": "Common laws prefetched successfully",
                "cache_stats": {
                    "law_lookup_cache_size": cache_manager.law_lookup_cache.size(),
                    "law_content_cache_size": cache_manager.law_content_cache.size(),
                    "article_cache_size": cache_manager.article_cache.size(),
                    "memory_usage_mb": cache_manager.memory_monitor.get_memory_usage_mb() if PERFORMANCE_MONITORING_AVAILABLE else "N/A"
                }
            }
            
            await ctx.info(f"Prefetch completed. Cache sizes: lookup={result['cache_stats']['law_lookup_cache_size']}, content={result['cache_stats']['law_content_cache_size']}, articles={result['cache_stats']['article_cache_size']}")
            return result

    except Exception as e:
        logger.error(f"Prefetch common laws error: {e}")
        await ctx.error(f"Prefetch failed: {str(e)}")
        raise ToolError(f"Prefetch failed: {str(e)}")

@mcp.tool
async def get_cache_stats(ctx: Context) -> dict:
    """
    Get current cache statistics and performance metrics

    Args:
        ctx: FastMCP context for logging
        
    Returns:
        Dict with detailed cache statistics
    """
    try:
        await ctx.info("Getting cache statistics...")
        cache_manager.cleanup_if_needed()

        result = {
            "cache_statistics": {
                "law_lookup_cache": {
                    "size": cache_manager.law_lookup_cache.size(),
                    "max_size": cache_manager.law_lookup_cache.max_size,
                    "ttl_seconds": cache_manager.law_lookup_cache.ttl
                },
                "law_content_cache": {
                    "size": cache_manager.law_content_cache.size(),
                    "max_size": cache_manager.law_content_cache.max_size,
                    "ttl_seconds": cache_manager.law_content_cache.ttl
                },
                "article_cache": {
                    "size": cache_manager.article_cache.size(),
                    "max_size": cache_manager.article_cache.max_size,
                    "ttl_seconds": cache_manager.article_cache.ttl
                },
                "disk_cache": {
                    "size": cache_manager.disk_cache.size(),
                    "ttl_seconds": cache_manager.disk_cache.ttl,
                    "directory": cache_manager.disk_cache.cache_dir
                }
            },
            "memory_monitoring": {
                "current_usage_mb": cache_manager.memory_monitor.get_memory_usage_mb() if PERFORMANCE_MONITORING_AVAILABLE else "N/A",
                "max_memory_mb": cache_manager.memory_monitor.max_memory_mb,
                "memory_limit_exceeded": cache_manager.memory_monitor.is_memory_limit_exceeded(),
                "monitoring_available": PERFORMANCE_MONITORING_AVAILABLE
            },
            "performance_features": [
                "🚀 LRU caching with TTL support",
                "💾 Memory-aware cache management",
                "⚡ Batch request optimization",
                "🔄 Automatic prefetching of common articles",
                "📊 Real-time cache statistics"
            ]
        }
        
        total_cache_items = sum([
            result["cache_statistics"]["law_lookup_cache"]["size"],
            result["cache_statistics"]["law_content_cache"]["size"],
            result["cache_statistics"]["article_cache"]["size"]
        ])
        
        await ctx.info(f"Cache statistics retrieved: {total_cache_items} total cached items")
        return result

    except Exception as e:
        logger.error(f"Get cache stats error: {e}")
        await ctx.error(f"Failed to get cache stats: {str(e)}")
        raise ToolError(f"Failed to get cache stats: {str(e)}")

@mcp.tool
async def clear_cache(cache_type: str = "all", ctx: Context = None) -> dict:
    """
    Clear specified cache or all caches

    Args:
        cache_type: Cache type to clear ("all", "law_lookup", "law_content", "article")
        ctx: FastMCP context for logging

    Returns:
        Dict with clear operation results
    """
    try:
        if ctx:
            await ctx.info(f"Clearing cache: {cache_type}")
            
        if cache_type == "all":
            cache_manager.law_lookup_cache.clear()
            cache_manager.law_content_cache.clear()
            cache_manager.article_cache.clear()
            cache_manager.disk_cache.clear()
            message = "All caches cleared successfully (including disk)"
        elif cache_type == "law_lookup":
            cache_manager.law_lookup_cache.clear()
            message = "Law lookup cache cleared successfully"
        elif cache_type == "law_content":
            cache_manager.law_content_cache.clear()
            message = "Law content cache cleared successfully"
        elif cache_type == "article":
            cache_manager.article_cache.clear()
            message = "Article cache cleared successfully"
        else:
            raise ToolError(f"Invalid cache_type: {cache_type}. Use 'all', 'law_lookup', 'law_content', or 'article'")

        result = {
            "status": "success",
            "message": message,
            "cache_stats_after_clear": {
                "law_lookup_cache_size": cache_manager.law_lookup_cache.size(),
                "law_content_cache_size": cache_manager.law_content_cache.size(),
                "article_cache_size": cache_manager.article_cache.size()
            }
        }
        
        if ctx:
            await ctx.info(f"Cache cleared successfully: {cache_type}")
        return result

    except ToolError:
        raise
    except Exception as e:
        logger.error(f"Clear cache error: {e}")
        if ctx:
            await ctx.error(f"Failed to clear cache: {str(e)}")
        raise ToolError(f"Failed to clear cache: {str(e)}")

@mcp.tool
async def find_related_laws(law_name: str, ctx: Context = None) -> dict:
    """
    Find related laws (施行令, 施行規則) for a given base law.

    Args:
        law_name: Base law name (e.g., "廃棄物処理法", "民法", "会社法")
        ctx: FastMCP context for logging

    Returns:
        Dict with base law info and related enforcement ordinances/ministerial ordinances
    """
    if not law_name or not law_name.strip():
        raise ToolError("law_name is required")

    try:
        if ctx:
            await ctx.info(f"Finding related laws for: {law_name}")

        # Resolve alias
        resolved_name = law_name.strip()
        if resolved_name in LAW_ALIASES:
            resolved_name = LAW_ALIASES[resolved_name]

        # Check related_laws config
        related = config_loader.related_laws.get(resolved_name, {})

        # Also try to search for related laws via API
        if not related:
            # Try searching for 施行令 and 施行規則
            async with await get_http_client() as client:
                related = {}
                for suffix, law_type in [("施行令", "CabinetOrder"), ("施行規則", "MinisterialOrdinance")]:
                    search_title = resolved_name + suffix
                    try:
                        response = await client.get("/laws", params={
                            "law_title": search_title,
                            "law_type": law_type,
                            "limit": 3
                        })
                        response.raise_for_status()
                        data = json.loads(response.text)
                        laws = data.get("laws", [])
                        if laws:
                            law_info = laws[0].get("revision_info", {}) or laws[0].get("law_info", {})
                            related[suffix] = law_info.get("law_title", search_title)
                            related[f"{suffix}番号"] = laws[0].get("law_info", {}).get("law_num", "")
                    except Exception as e:
                        logger.warning(f"Failed to search for {search_title}: {e}")

        result = {
            "base_law": resolved_name,
            "alias_used": law_name.strip() if law_name.strip() != resolved_name else None,
            "related_laws": related,
        }

        if ctx:
            await ctx.info(f"Found {len(related)} related law entries for {resolved_name}")
        return result

    except ToolError:
        raise
    except Exception as e:
        logger.error(f"Find related laws error: {e}")
        if ctx:
            await ctx.error(f"Failed to find related laws: {str(e)}")
        raise ToolError(f"Failed to find related laws: {str(e)}")


@mcp.tool
async def follow_law_chain(law_name: str, article_number: str, paragraph: str = "", max_depth: int = 3, ctx: Context = None) -> dict:
    """
    Follow cross-references from a law article to 施行令 and 施行規則.
    When a law says "政令で定める" or "環境省令で定める", this tool automatically
    finds and retrieves the corresponding articles in the enforcement ordinance/ministerial ordinance.

    Args:
        law_name: Starting law name (e.g., "廃棄物処理法")
        article_number: Article number to start from (e.g., "12")
        paragraph: Optional paragraph number to narrow the search
        max_depth: Maximum chain depth (default 3: 法→施行令→施行規則)
        ctx: FastMCP context for logging

    Returns:
        Dict with the chain of referenced articles across law/ordinance/regulation
    """
    try:
        if ctx:
            await ctx.info(f"Following law chain from {law_name} art.{article_number}...")

        chain = []
        visited = set()

        async def _get_article_text(name, art, para=None):
            """Get article text via cache-aware lookup."""
            num = await smart_law_lookup(name)
            if not num:
                return None, None
            cached = cache_manager.law_content_cache.get(num)
            if cached:
                data = cached
            else:
                disk = cache_manager.disk_cache.get(f"law_data:{num}")
                if disk:
                    data = disk
                    cache_manager.law_content_cache.put(num, data)
                else:
                    async with await get_http_client() as client:
                        response = await client.get(f"/law_data/{num}", params={"law_full_text_format": "xml"})
                        response.raise_for_status()
                        data = json.loads(response.text)
                        cache_manager.law_content_cache.put(num, data)
                        cache_manager.disk_cache.put(f"law_data:{num}", data)
            matches = find_article_in_xml(data.get("law_full_text", {}), art, paragraph=para)
            if matches:
                return matches[0].get("full_text", ""), matches
            return None, None

        def _extract_references(text, source_law_name):
            """Extract cross-references to 施行令/施行規則 from article text."""
            refs = []
            # Determine the base law (strip 施行令/施行規則 suffixes)
            base_law = source_law_name
            for suffix in ["施行規則", "施行令"]:
                if base_law.endswith(suffix):
                    base_law = base_law[:-len(suffix)]
                    break

            # Pattern 1: "政令で定める" → 施行令
            if "政令で定める" in text:
                refs.append({"type": "施行令", "law": base_law + "施行令", "reason": "政令で定める"})

            # Pattern 2: "環境省令で定める" / "省令で定める" / "厚生省令で定める" → 施行規則
            if re.search(r'(?:環境|厚生)?省令で定める', text):
                refs.append({"type": "施行規則", "law": base_law + "施行規則", "reason": "省令で定める"})

            # Pattern 3: Explicit article references "令第○条" "規則第○条"
            for m in re.finditer(r'令第([一二三四五六七八九十百]+)条(?:の([一二三四五六七八九十百]+))?', text):
                kanji_num = m.group(1)
                kanji_sub = m.group(2)
                art_num = _kanji_to_arabic(kanji_num)
                if kanji_sub:
                    art_num += "_" + _kanji_to_arabic(kanji_sub)
                refs.append({"type": "施行令", "law": base_law + "施行令",
                            "article": art_num, "reason": f"令第{m.group(0)[1:]}"})

            for m in re.finditer(r'規則第([一二三四五六七八九十百]+)条(?:の([一二三四五六七八九十百]+))?', text):
                kanji_num = m.group(1)
                kanji_sub = m.group(2)
                art_num = _kanji_to_arabic(kanji_num)
                if kanji_sub:
                    art_num += "_" + _kanji_to_arabic(kanji_sub)
                refs.append({"type": "施行規則", "law": base_law + "施行規則",
                            "article": art_num, "reason": f"規則第{m.group(0)[2:]}"})

            # Pattern 4: 準用 (mutatis mutandis application)
            # e.g., "第十四条の規定は、…について準用する"
            # e.g., "第七条から第九条までの規定を準用する"
            if "準用" in text:
                junyou_seen = set()
                # Extract sentences containing 準用
                for sentence in re.split(r'[。\n]', text):
                    if "準用" not in sentence:
                        continue
                    # Get the part before 準用 to find referenced articles
                    before_junyou = sentence[:sentence.index("準用")]

                    # Range: "第○条から第○条まで"
                    for m in re.finditer(r'第([一二三四五六七八九十百]+)条から第([一二三四五六七八九十百]+)条まで', before_junyou):
                        start_art = int(_kanji_to_arabic(m.group(1)))
                        end_art = int(_kanji_to_arabic(m.group(2)))
                        for art in range(start_art, end_art + 1):
                            if art not in junyou_seen:
                                junyou_seen.add(art)
                                refs.append({"type": "準用", "law": source_law_name,
                                            "article": str(art), "reason": f"第{m.group(1)}条〜第{m.group(2)}条を準用"})

                    # Single: "第○条"
                    for m in re.finditer(r'第([一二三四五六七八九十百]+)条(?:の([一二三四五六七八九十百]+))?', before_junyou):
                        kanji_num = m.group(1)
                        kanji_sub = m.group(2)
                        art_num = _kanji_to_arabic(kanji_num)
                        if kanji_sub:
                            art_num += "_" + _kanji_to_arabic(kanji_sub)
                        art_int = int(art_num.split("_")[0])
                        if art_int not in junyou_seen:
                            junyou_seen.add(art_int)
                            refs.append({"type": "準用", "law": source_law_name,
                                        "article": art_num, "reason": f"第{kanji_num}条を準用"})

            return refs

        def _kanji_to_arabic(kanji: str) -> str:
            """Convert kanji number to arabic (simplified)."""
            kanji_map = {"一": 1, "二": 2, "三": 3, "四": 4, "五": 5,
                        "六": 6, "七": 7, "八": 8, "九": 9}
            if not kanji:
                return "0"
            # Handle tens
            if "十" in kanji:
                parts = kanji.split("十")
                tens = kanji_map.get(parts[0], 1) if parts[0] else 1
                ones = kanji_map.get(parts[1], 0) if len(parts) > 1 and parts[1] else 0
                # Handle hundreds
                if "百" in kanji:
                    # Simplified: just parse basic patterns
                    return str(int(kanji.replace("百", "00").replace("十", "0")))
                return str(tens * 10 + ones)
            return str(kanji_map.get(kanji, 0))

        # Start chain traversal
        current_law = law_name
        current_article = article_number
        current_para = paragraph if paragraph else None

        for depth in range(max_depth):
            visit_key = f"{current_law}:{current_article}:{current_para or ''}"
            if visit_key in visited:
                break
            visited.add(visit_key)

            text, structured = await _get_article_text(current_law, current_article, current_para)
            if not text:
                chain.append({
                    "depth": depth,
                    "law": current_law,
                    "article": current_article,
                    "paragraph": current_para or "",
                    "status": "not_found",
                })
                break

            # Extract cross-references
            refs = _extract_references(text, current_law)

            node = {
                "depth": depth,
                "law": current_law,
                "article": current_article,
                "paragraph": current_para or "",
                "text_preview": text[:300].replace("\n", " ").strip() + ("..." if len(text) > 300 else ""),
                "references_found": len(refs),
                "references": [{"type": r["type"], "law": r["law"],
                               "article": r.get("article", ""), "reason": r["reason"]} for r in refs],
            }
            chain.append(node)

            # Follow the first specific article reference (if any)
            specific_refs = [r for r in refs if r.get("article")]
            if specific_refs:
                ref = specific_refs[0]
                current_law = ref["law"]
                current_article = ref["article"]
                current_para = None
            elif refs:
                # Generic reference ("政令で定める" / "省令で定める") without specific article number.
                # Strategy: search the target law (施行令/施行規則) for reverse references
                # back to the source article. e.g., 施行令 will say "法第十二条の政令で定める基準は..."
                ref = refs[0]
                target_law = ref["law"]
                
                # Build the reverse-reference pattern: "法第{current_article}条"
                # Handle compound article numbers like "8_5" → "法第八条の五"
                art_parts = str(current_article).split("_")
                if art_parts[0].isdigit():
                    main_kanji = arabic_to_kanji(art_parts[0])
                    if len(art_parts) > 1:
                        sub_parts = []
                        for p in art_parts[1:]:
                            sub_parts.append(arabic_to_kanji(p) if p.isdigit() else p)
                        kanji_art = main_kanji + "条の" + "の".join(sub_parts)
                        reverse_pattern = f"法第{kanji_art}"
                    else:
                        reverse_pattern = f"法第{main_kanji}条"
                else:
                    reverse_pattern = f"法第{str(current_article)}条"
                
                # Get all articles from the target law
                target_num = await smart_law_lookup(target_law)
                found_reverse = False
                if target_num:
                    target_cached = cache_manager.law_content_cache.get(target_num)
                    if target_cached:
                        target_data = target_cached
                    else:
                        target_disk = cache_manager.disk_cache.get(f"law_data:{target_num}")
                        if target_disk:
                            target_data = target_disk
                            cache_manager.law_content_cache.put(target_num, target_data)
                        else:
                            async with await get_http_client() as client:
                                resp = await client.get(f"/law_data/{target_num}", params={"law_full_text_format": "xml"})
                                resp.raise_for_status()
                                target_data = json.loads(resp.text)
                                cache_manager.law_content_cache.put(target_num, target_data)
                                cache_manager.disk_cache.put(f"law_data:{target_num}", target_data)
                    
                    target_articles = extract_articles_from_xml(target_data.get("law_full_text", {}))
                    # Search for articles that reference back to our source article
                    for tart in target_articles:
                        ttext = tart.get("full_text", "")
                        if reverse_pattern in ttext:
                            found_reverse = True
                            node["generic_follow"] = {
                                "strategy": "reverse_reference",
                                "pattern": reverse_pattern,
                                "target_law": target_law,
                                "target_article": tart.get("num", ""),
                                "found": True,
                            }
                            current_law = target_law
                            current_article = tart.get("num", "")
                            current_para = None
                            break
                
                if not found_reverse:
                    node["generic_follow"] = {
                        "strategy": "reverse_reference",
                        "pattern": reverse_pattern,
                        "target_law": target_law,
                        "found": False,
                        "note": f"{target_law}内に「{reverse_pattern}」への逆参照が見つかりませんでした。"
                    }
                    break
            else:
                break

        result = {
            "starting_point": f"{law_name} 第{article_number}条" + (f" 第{paragraph}項" if paragraph else ""),
            "chain_depth": len(chain),
            "chain": chain,
        }

        if ctx:
            await ctx.info(f"Chain traversal complete: {len(chain)} nodes")
        return result

    except ToolError:
        raise
    except Exception as e:
        if ctx:
            await ctx.error(f"Failed to follow chain: {str(e)}")
        raise ToolError(f"Failed to follow chain: {str(e)}")


# --- Definition label extraction ---
# Extracts defined terms from legal text, e.g., 「以下「安定型産業廃棄物」という。」
_DEFINITION_PATTERN = re.compile(
    r'以下「([^」]{1,30})」という。'
)
_EXCLUSION_DEFINITION_PATTERN = re.compile(
    r'([^。、（）]{2,40}?)以外の([^。、（）]{2,40}?)の'
)


def _extract_definition_label(text: str) -> str:
    """
    Extract a definition label from legal sub-item text.
    
    Detects patterns like:
    - 「以下「安定型産業廃棄物」という。」→ "安定型産業廃棄物"
    - 「安定型産業廃棄物の埋立処分」→ keyword-based detection
    - 「公共の水域及び地下水と遮断されている場所」→ "遮断型" inference
    
    Returns the label string or empty string if none found.
    """
    # Primary: explicit definition with 「以下「X」という。」
    m = _DEFINITION_PATTERN.search(text)
    if m:
        return m.group(1)
    
    return ""


def _annotate_subitems_with_labels(items: list) -> None:
    """
    Post-process items/sub_items to add 'definition_label' field
    when a defined term is found in the text.
    Mutates items in place.
    """
    for item in items:
        # Check item-level text
        text = item.get("text", "")
        label = _extract_definition_label(text)
        if label:
            item["definition_label"] = label
        # Check sub_items
        for si in item.get("sub_items", []):
            si_text = si.get("text", "")
            si_label = _extract_definition_label(si_text)
            if si_label:
                si["definition_label"] = si_label


# --- Internal sub-item cross-reference resolution ---
# Maps for sub-item title hierarchies
_IROHA_TITLES = list("イロハニホヘトチリヌルヲワカヨタレソツネナラムウヰノオクヤマケフコエテアサキユメミシヱヒモセス")
_IROHA_SET = set(_IROHA_TITLES)
# Katakana number titles for Subitem2: (1)(2)... rendered as ⑴⑵ or （１）（２）
_SUBITEM2_RE = re.compile(r'[⑴⑵⑶⑷⑸⑹⑺⑻⑼⑽⑾⑿⒀⒁⒂⒃⒄⒅⒆⒇]|（([０-９]+)）|\((\d+)\)')

# Patterns for detecting internal sub-item references within text
# e.g., "イに規定する", "ロに掲げる", "イ又はロに該当する", "イからハまでに掲げる"
_SUBREF_SINGLE = re.compile(
    r'([イロハニホヘトチリヌルヲワカヨタレソツネナラムウヰノオクヤマケフコエテアサキユメミシヱヒモセス])'
    r'(?:に規定する|に掲げる|に該当する|に定める|の規定)'
)
_SUBREF_LIST = re.compile(
    r'([イロハニホヘトチリヌルヲワ])(?:[、，]([イロハニホヘトチリヌルヲワ]))*'
    r'(?:又は|若しくは|及び|並びに)([イロハニホヘトチリヌルヲワ])'
    r'(?:に規定する|に掲げる|に該当する|に定める|の規定)'
)
_SUBREF_RANGE = re.compile(
    r'([イロハニホヘトチリヌルヲワ])から([イロハニホヘトチリヌルヲワ])まで'
)


def _resolve_internal_subrefs(expanded_item_refs: list) -> None:
    """
    Post-process expanded_item_refs to resolve internal sub-item cross-references.

    When sub-item ハ's text says "イに規定する..." or "イからロまでに掲げる...",
    we find イ (and ロ) in the same item's sub_items and attach their content
    as 'resolved_refs' on the referencing sub-item.

    Also handles:
    - Item-to-item refs within the same expansion (号→号)
    - Subitem2 level ((1)→(2) style)
    - Range references (イからハまで)
    - List references (イ及びロ, イ又はロ)

    Mutates expanded_item_refs in place.
    """
    for ref_group in expanded_item_refs:
        items = ref_group.get("items", [])
        # Also handle sub_items directly on ref_group (from 7b subitem expansion)
        sub_items = ref_group.get("sub_items", [])

        # Annotate definition labels on items and sub_items
        _annotate_subitems_with_labels(items)
        _annotate_subitems_with_labels(sub_items)

        # --- Resolve within items' sub_items ---
        for item in items:
            item_subs = item.get("sub_items", [])
            if not item_subs:
                continue
            # Build lookup by title
            by_title = {si["title"]: si for si in item_subs}
            for si in item_subs:
                text = si.get("text", "")
                referenced_titles = _extract_subref_titles(text)
                if not referenced_titles:
                    continue
                resolved = []
                for title in referenced_titles:
                    if title == si.get("title"):
                        continue  # skip self-reference
                    target = by_title.get(title)
                    if target:
                        resolved.append({"title": target["title"], "text": target["text"]})
                if resolved:
                    si["resolved_refs"] = resolved

        # --- Resolve within sub_items (from 7b) ---
        if sub_items:
            by_title = {si["title"]: si for si in sub_items}
            # Also include all sibling sub_items for out-of-range resolution
            all_siblings = ref_group.get("_all_sibling_subs", [])
            all_by_title = {si["title"]: si for si in all_siblings} if all_siblings else {}
            for si in sub_items:
                text = si.get("text", "")
                referenced_titles = _extract_subref_titles(text)
                if not referenced_titles:
                    continue
                resolved = []
                for title in referenced_titles:
                    if title == si.get("title"):
                        continue
                    target = by_title.get(title) or all_by_title.get(title)
                    if target:
                        resolved.append({"title": target["title"], "text": target["text"]})
                if resolved:
                    si["resolved_refs"] = resolved

        # --- Resolve item-to-item refs within same expansion ---
        if len(items) > 1:
            by_num = {it.get("item_num"): it for it in items}
            for item in items:
                text = item.get("text", "")
                # Detect "前号" (previous item), "次号" (next item), "第X号" refs
                item_refs = []
                inum = item.get("item_num")
                if isinstance(inum, int):
                    if "前号" in text and (inum - 1) in by_num:
                        item_refs.append(inum - 1)
                    if "次号" in text and (inum + 1) in by_num:
                        item_refs.append(inum + 1)
                # Explicit refs: "第X号に" within same paragraph context
                for m in re.finditer(r'第([一二三四五六七八九十]+)号(?:に規定する|に掲げる|に該当する|に定める)', text):
                    try:
                        ref_num = int(kanji_to_arabic(m.group(1)))
                        if ref_num != inum and ref_num in by_num:
                            item_refs.append(ref_num)
                    except (ValueError, KeyError):
                        pass
                if item_refs:
                    resolved = []
                    for rn in dict.fromkeys(item_refs):  # dedupe preserving order
                        target = by_num[rn]
                        resolved.append({"item_num": rn, "text": target.get("text", "")[:200]})
                    item["resolved_refs"] = resolved


def _extract_subref_titles(text: str) -> list[str]:
    """Extract referenced sub-item titles (イ,ロ,ハ,...) from text."""
    titles = []
    seen = set()

    # Range: イからハまで
    for m in _SUBREF_RANGE.finditer(text):
        start_char, end_char = m.group(1), m.group(2)
        try:
            start_idx = _IROHA_TITLES.index(start_char)
            end_idx = _IROHA_TITLES.index(end_char)
            for i in range(start_idx, end_idx + 1):
                t = _IROHA_TITLES[i]
                if t not in seen:
                    titles.append(t)
                    seen.add(t)
        except ValueError:
            pass

    # List: イ及びロ, イ又はロ, イ、ロ及びハ
    for m in _SUBREF_LIST.finditer(text):
        for g in m.groups():
            if g and g in _IROHA_SET and g not in seen:
                titles.append(g)
                seen.add(g)

    # Single: イに規定する
    for m in _SUBREF_SINGLE.finditer(text):
        t = m.group(1)
        if t not in seen:
            titles.append(t)
            seen.add(t)

    return titles


@mcp.tool
async def explain_law_article(law_name: str, article_number: str, paragraph: str = "", deep: bool = False, ctx: Context = None) -> dict:
    """
    Deep explanation of a law article: fetches the article text and automatically
    resolves ALL cross-references in parallel (施行令, 施行規則, 準用) into a tree.

    Unlike follow_law_chain (which follows a single linear chain), this tool
    expands ALL references from one article simultaneously, giving a complete
    picture of what "政令で定める", "省令で定める", and "準用する" actually mean.

    Features:
    - Delegation expansion: 施行令 and 施行規則 reverse-reference search
    - Item cross-reference expansion: 「第X条第Y項第Z号から第W号まで」auto-resolve
    - 準用 (mutatis mutandis) with 読替え (substitution) parsing
    - Exception/omission rule extraction: 「にかかわらず」「場合に限り」patterns
    - Deep mode: recursively expand key 施行規則 articles' item references too

    Args:
        law_name: Law name (e.g., "廃棄物処理法")
        article_number: Article number (e.g., "14", "9の2の2")
        paragraph: Optional paragraph to focus on
        deep: If True, also expand item references in key delegated articles (施行規則)
        ctx: FastMCP context

    Returns:
        Dict with the article text and all resolved references in a tree structure
    """
    try:
        if ctx:
            await ctx.info(f"Explaining {law_name} art.{article_number}...")

        # Helper: get law data with caching
        async def _get_law_data(name):
            num = await smart_law_lookup(name)
            if not num:
                return None, None
            cached = cache_manager.law_content_cache.get(num)
            if cached:
                return num, cached
            disk = cache_manager.disk_cache.get(f"law_data:{num}")
            if disk:
                cache_manager.law_content_cache.put(num, disk)
                return num, disk
            async with await get_http_client() as client:
                resp = await client.get(f"/law_data/{num}", params={"law_full_text_format": "xml"})
                resp.raise_for_status()
                data = json.loads(resp.text)
                cache_manager.law_content_cache.put(num, data)
                cache_manager.disk_cache.put(f"law_data:{num}", data)
                return num, data

        # 1. Get the main article
        law_num, law_data = await _get_law_data(law_name)
        if not law_data:
            raise ToolError(f"Law '{law_name}' not found")

        matches = find_article_in_xml(law_data.get("law_full_text", {}), article_number, paragraph=paragraph if paragraph else None)
        if not matches:
            raise ToolError(f"Article {article_number} not found in {law_name}")

        article = matches[0]
        article_text = article.get("full_text", "")

        # Determine base law name
        resolved_name = law_data.get("law_info", {}).get("law_title", law_name)
        base_law = resolved_name
        for suffix in ["施行規則", "施行令"]:
            if base_law.endswith(suffix):
                base_law = base_law[:-len(suffix)]
                break

        # 2. Extract all "政令で定める___" and "省令で定める___" with context
        seirei_contexts = []
        kisoku_contexts = []
        for m in re.finditer(r'政令で定める([^。、）]{1,20})', article_text):
            seirei_contexts.append(m.group(1).strip())
        for m in re.finditer(r'(?:環境|厚生)?省令で定める([^。、）]{1,20})', article_text):
            kisoku_contexts.append(m.group(1).strip())

        # 3. Build reverse reference pattern
        art_clean = article.get("num", article_number)
        # Convert to kanji for reverse search
        parts = art_clean.split("_")
        if parts[0].isdigit():
            kanji_parts = [arabic_to_kanji(parts[0])]
            for p in parts[1:]:
                if p.isdigit():
                    kanji_parts.append(arabic_to_kanji(p))
                else:
                    kanji_parts.append(p)
            kanji_art = "の".join(kanji_parts) if len(kanji_parts) > 1 else kanji_parts[0]
        else:
            kanji_art = art_clean
        reverse_pattern = f"法第{kanji_art}条"

        # 4. Search 施行令 for reverse references
        seirei_refs = []
        if seirei_contexts or "政令で定める" in article_text:
            seirei_name = base_law + "施行令"
            _, seirei_data = await _get_law_data(seirei_name)
            if seirei_data:
                seirei_articles = extract_articles_from_xml(seirei_data.get("law_full_text", {}))
                for sa in seirei_articles:
                    sa_text = sa.get("full_text", "")
                    if reverse_pattern in sa_text:
                        preview = sa_text[:300].replace("\n", " ").strip()
                        seirei_refs.append({
                            "article_num": sa.get("num", ""),
                            "caption": sa.get("caption", ""),
                            "preview": preview + ("..." if len(sa_text) > 300 else ""),
                        })

        # 5. Search 施行規則 for reverse references
        kisoku_refs = []
        if kisoku_contexts or re.search(r'(?:環境|厚生)?省令で定める', article_text):
            kisoku_name = base_law + "施行規則"
            _, kisoku_data = await _get_law_data(kisoku_name)
            if kisoku_data:
                kisoku_articles = extract_articles_from_xml(kisoku_data.get("law_full_text", {}))
                for ka in kisoku_articles:
                    ka_text = ka.get("full_text", "")
                    if reverse_pattern in ka_text:
                        preview = ka_text[:300].replace("\n", " ").strip()
                        kisoku_refs.append({
                            "article_num": ka.get("num", ""),
                            "caption": ka.get("caption", ""),
                            "preview": preview + ("..." if len(ka_text) > 300 else ""),
                        })

        # 6. Extract 準用 references and fetch their text
        junyou_refs = []
        if "準用" in article_text:
            junyou_seen = set()
            for sentence in re.split(r'[。\n]', article_text):
                if "準用" not in sentence:
                    continue
                before = sentence[:sentence.index("準用")]
                for m in re.finditer(r'第([一二三四五六七八九十百]+)条(?:の([一二三四五六七八九十百]+))*', before):
                    full_match = m.group(0)
                    # Parse all の parts
                    sub_parts = re.findall(r'の([一二三四五六七八九十百]+)', full_match)
                    main_kanji = m.group(1)
                    main_num = _normalize_article_number(full_match)
                    if main_num in junyou_seen:
                        continue
                    junyou_seen.add(main_num)
                    # Fetch the referenced article
                    ref_matches = find_article_in_xml(law_data.get("law_full_text", {}), main_num)
                    ref_preview = ""
                    ref_caption = ""
                    if ref_matches:
                        ref_text = ref_matches[0].get("full_text", "")
                        ref_caption = ref_matches[0].get("caption", "")
                        ref_preview = ref_text[:300].replace("\n", " ").strip()
                        if len(ref_text) > 300:
                            ref_preview += "..."
                    junyou_refs.append({
                        "article_num": main_num,
                        "caption": ref_caption,
                        "preview": ref_preview,
                        "source_sentence": sentence.strip()[:200],
                    })

        # 7. Expand item cross-references (号の引用展開)
        # Detect patterns like "第X条第Y項第Z号から第W号まで" or "第X条第Y項第Z号に掲げる"
        expanded_item_refs = []
        # Pattern: 第X条(のN)*第Y項第Z号(から第W号まで)?
        # Note: "条の二" pattern — の comes AFTER 条 (e.g., 第九条の二第二項第六号)
        item_ref_pattern = re.compile(
            r'第([一二三四五六七八九十百]+)条(?:の([一二三四五六七八九十百]+))*'
            r'(?:第([一二三四五六七八九十百]+)項)?'
            r'第([一二三四五六七八九十百]+)号'
            r'(?:から第([一二三四五六七八九十百]+)号まで)?'
        )
        seen_item_refs = set()
        for m in item_ref_pattern.finditer(article_text):
            ref_art_main_kanji = m.group(1)
            # Collect all の-sub numbers via findall on the full match
            full_match = m.group(0)
            no_subs = re.findall(r'条の([一二三四五六七八九十百]+)', full_match)
            ref_para_kanji = m.group(3)
            ref_item_start_kanji = m.group(4)
            ref_item_end_kanji = m.group(5)

            # Build article number: e.g., 九条の二 → 9_2
            art_num = kanji_to_arabic(ref_art_main_kanji)
            for sub in no_subs:
                art_num += "_" + kanji_to_arabic(sub)
            ref_art = art_num
            ref_para = kanji_to_arabic(ref_para_kanji) if ref_para_kanji else None
            ref_item_start = int(kanji_to_arabic(ref_item_start_kanji))
            ref_item_end = int(kanji_to_arabic(ref_item_end_kanji)) if ref_item_end_kanji else ref_item_start

            ref_key = f"{ref_art}:{ref_para}:{ref_item_start}-{ref_item_end}"
            if ref_key in seen_item_refs:
                continue
            seen_item_refs.add(ref_key)

            # Skip self-references (references to the same article we're explaining)
            if ref_art == art_clean and not ref_para_kanji:
                continue

            # Determine which law to search - check if it's in the same law or 施行規則
            # By default, search in the same law the article belongs to
            ref_law_data = law_data
            ref_law_name_used = law_name

            # Try to find the article in the current law first
            ref_matches = find_article_in_xml(
                ref_law_data.get("law_full_text", {}), ref_art,
                paragraph=ref_para
            )

            # If not found and we have 施行規則 data, try there
            if not ref_matches and kisoku_refs:
                kisoku_name = base_law + "施行規則"
                _, kisoku_data_full = await _get_law_data(kisoku_name)
                if kisoku_data_full:
                    ref_matches = find_article_in_xml(
                        kisoku_data_full.get("law_full_text", {}), ref_art,
                        paragraph=ref_para
                    )
                    if ref_matches:
                        ref_law_name_used = kisoku_name

            if not ref_matches:
                continue

            ref_article = ref_matches[0]
            # Extract specific items from the paragraph
            ref_items = []
            for p in ref_article.get("paragraphs", []):
                # If paragraph specified, match it; otherwise take first
                if ref_para and str(p.get("num", "")) != str(ref_para):
                    continue
                for item in p.get("items", []):
                    item_num_str = item.get("num", "")
                    # Parse item number (kanji to int)
                    item_text_raw = item.get("text", "").strip()
                    # Extract the numeric part from the item
                    item_kanji_match = re.match(r'([一二三四五六七八九十百]+)', item_text_raw)
                    if item_kanji_match:
                        try:
                            item_int = int(kanji_to_arabic(item_kanji_match.group(1)))
                        except (ValueError, KeyError):
                            continue
                    else:
                        try:
                            item_int = int(item_num_str)
                        except (ValueError, TypeError):
                            continue

                    if ref_item_start <= item_int <= ref_item_end:
                        # Clean up item text - remove leading kanji number
                        clean_text = re.sub(r'^[一二三四五六七八九十百]+\s*', '', item_text_raw).strip()
                        item_entry = {
                            "item_num": item_int,
                            "text": clean_text[:200],
                        }
                        # Include sub_items (イロハ) if present
                        if item.get("sub_items"):
                            item_entry["sub_items"] = [
                                {"title": si["title"], "text": si["text"][:200]}
                                for si in item["sub_items"]
                            ]
                        ref_items.append(item_entry)

            if ref_items:
                expanded_item_refs.append({
                    "reference": m.group(0),
                    "source_law": ref_law_name_used,
                    "article": ref_art,
                    "paragraph": ref_para,
                    "item_range": f"{ref_item_start}-{ref_item_end}" if ref_item_end != ref_item_start else str(ref_item_start),
                    "items": ref_items,
                })

        # 7b. Expand sub-item references (イロハ level)
        # Pattern: 第X条第Y号イ, 第X条第Y号ロ及びハ, 令第七条第十四号ロ
        subitem_ref_pattern = re.compile(
            r'(?:令)?第([一二三四五六七八九十百]+)条(?:の([一二三四五六七八九十百]+))*'
            r'(?:第([一二三四五六七八九十百]+)項)?'
            r'第([一二三四五六七八九十百]+)号([イロハニホヘトチリヌルヲワ])'
            r'(?:(?:及び|並びに|から)([イロハニホヘトチリヌルヲワ])(?:まで)?)?'
        )
        iroha_map = {"イ": 1, "ロ": 2, "ハ": 3, "ニ": 4, "ホ": 5, "ヘ": 6, "ト": 7,
                     "チ": 8, "リ": 9, "ヌ": 10, "ル": 11, "ヲ": 12, "ワ": 13}
        seen_subitem_refs = set()
        for m in subitem_ref_pattern.finditer(article_text):
            full = m.group(0)
            art_main = m.group(1)
            no_subs = re.findall(r'条の([一二三四五六七八九十百]+)', full)
            para_kanji = m.group(3)
            item_kanji = m.group(4)
            si_start_char = m.group(5)
            si_end_char = m.group(6)

            art_num = kanji_to_arabic(art_main)
            for sub in no_subs:
                art_num += "_" + kanji_to_arabic(sub)
            item_num = int(kanji_to_arabic(item_kanji))
            si_start = iroha_map.get(si_start_char, 0)
            si_end = iroha_map.get(si_end_char, si_start) if si_end_char else si_start

            ref_key = f"si:{art_num}:{item_num}:{si_start}-{si_end}"
            if ref_key in seen_subitem_refs:
                continue
            seen_subitem_refs.add(ref_key)

            # Determine target law — "令第..." means 施行令
            is_seirei = full.startswith("令")
            if is_seirei:
                target_name = base_law + "施行令"
                _, target_data = await _get_law_data(target_name)
            else:
                target_data = law_data
                target_name = law_name

            if not target_data:
                continue

            ref_matches = find_article_in_xml(
                target_data.get("law_full_text", {}), art_num,
                paragraph=kanji_to_arabic(para_kanji) if para_kanji else None
            )
            if not ref_matches:
                continue

            # Find the specific item and its sub_items
            for p in ref_matches[0].get("paragraphs", []):
                for item in p.get("items", []):
                    try:
                        i_num = int(item.get("num", "0"))
                    except (ValueError, TypeError):
                        continue
                    if i_num != item_num:
                        continue
                    if not item.get("sub_items"):
                        continue
                    matched_subs = []
                    for si in item["sub_items"]:
                        si_idx = iroha_map.get(si.get("title", ""), 0)
                        if si_start <= si_idx <= si_end:
                            matched_subs.append({
                                "title": si["title"],
                                "text": si["text"][:200],
                            })
                    if matched_subs:
                        # Also collect ALL sub_items for cross-reference resolution
                        # (e.g., if ハ references イ which is outside the ロ〜ハ range)
                        all_subs_for_resolution = [
                            {"title": si.get("title", ""), "text": si["text"][:200]}
                            for si in item["sub_items"]
                        ]
                        expanded_item_refs.append({
                            "reference": full,
                            "source_law": target_name,
                            "article": art_num,
                            "item_num": item_num,
                            "sub_item_range": f"{si_start_char}" + (f"〜{si_end_char}" if si_end_char else ""),
                            "sub_items": matched_subs,
                            "_all_sibling_subs": all_subs_for_resolution,
                        })

        # 7c. Resolve internal sub-item cross-references
        # When ハ says "イに規定する..." we attach イ's content to ハ
        _resolve_internal_subrefs(expanded_item_refs)

        # 7d. Compute excluded sub-items (complement of referenced set)
        for ref_group in expanded_item_refs:
            all_subs = ref_group.get("_all_sibling_subs", [])
            if all_subs and ref_group.get("sub_items"):
                matched_titles = {si["title"] for si in ref_group["sub_items"]}
                excluded = [
                    {"title": si["title"], "text": si["text"]}
                    for si in all_subs
                    if si["title"] not in matched_titles
                ]
                if excluded:
                    ref_group["excluded_sub_items"] = excluded
            ref_group.pop("_all_sibling_subs", None)

        # 8. Extract 準用 読替え (substitution rules)
        yomikae_rules = []
        # Normalize whitespace for sentence splitting (XML text has lots of newlines/spaces)
        normalized_text = re.sub(r'\s+', ' ', article_text)
        if "とあるのは" in normalized_text:
            # Find all 読替え blocks: text between "この場合において" and "読み替えるものとする"
            # or just find all substitution pairs globally
            yomikae_blocks = re.finditer(
                r'((?:この場合において|場合において)[^読]{0,2000}?読み替える(?:ものとする)?)',
                normalized_text
            )
            for block_match in yomikae_blocks:
                block = block_match.group(1)
                pairs = re.findall(r'「([^」]+)」とあるのは「([^」]+)」と', block)
                if pairs:
                    junyou_art = ""
                    art_match = re.search(r'第([一二三四五六七八九十百]+(?:の[一二三四五六七八九十百]+)*)条', block)
                    if art_match:
                        junyou_art = _normalize_article_number(art_match.group(0))
                    yomikae_rules.append({
                        "target_article": junyou_art,
                        "substitutions": [{"from": f, "to": t} for f, t in pairs],
                        "source_sentence": block.strip()[:500],
                    })

        # 9. Extract exception/omission rules (省略規定・例外規定)
        exception_rules = []
        for sentence in normalized_text.split("。"):
            sentence = sentence.strip()
            if not sentence:
                continue
            # Pattern 1: 「…の規定にかかわらず」
            if "の規定にかかわらず" in sentence or "にかかわらず" in sentence:
                condition = ""
                cond_match = re.search(r'(.{5,50}?)(?:の規定)?にかかわらず', sentence)
                if cond_match:
                    condition = cond_match.group(1).strip()
                effect = ""
                eff_match = re.search(r'にかかわらず[、，](.{5,100}?)(?:こと|もの|する)', sentence)
                if eff_match:
                    effect = eff_match.group(1).strip()
                exception_rules.append({
                    "type": "exception",
                    "pattern": "にかかわらず",
                    "condition": condition,
                    "effect": effect,
                    "sentence": sentence[:200],
                })
            # Pattern 2: 「…の場合に限り」
            elif "場合に限り" in sentence:
                condition = ""
                cond_match = re.search(r'(.{5,60}?)場合に限り', sentence)
                if cond_match:
                    condition = cond_match.group(1).strip()
                exception_rules.append({
                    "type": "limitation",
                    "pattern": "場合に限り",
                    "condition": condition,
                    "sentence": sentence[:200],
                })
            # Pattern 3: 「…を要しない」
            elif "を要しない" in sentence:
                what = ""
                what_match = re.search(r'(.{2,30}?)を要しない', sentence)
                if what_match:
                    what = what_match.group(1).strip()
                exception_rules.append({
                    "type": "exemption",
                    "pattern": "を要しない",
                    "what_exempted": what,
                    "sentence": sentence[:200],
                })
            # Pattern 4: 「…この限りでない」
            elif "この限りでない" in sentence:
                condition = ""
                cond_match = re.search(r'ただし[、，](.{5,80}?)(?:ときは|場合は|については)[、，]?この限りでない', sentence)
                if cond_match:
                    condition = cond_match.group(1).strip()
                exception_rules.append({
                    "type": "proviso",
                    "pattern": "この限りでない",
                    "condition": condition,
                    "sentence": sentence[:200],
                })

        # 10. Deep mode: expand key delegated articles' item references
        deep_expansions = []
        if deep and (kisoku_refs or seirei_refs):
            # Pick key articles: those with "許可の申請", "基準", "届出" in caption
            key_keywords = ["許可の申請", "許可の基準", "届出", "申請"]
            target_refs = []
            for ref in kisoku_refs:
                cap = ref.get("caption", "")
                if any(kw in cap for kw in key_keywords):
                    target_refs.append(("施行規則", base_law + "施行規則", ref))
            for ref in seirei_refs:
                cap = ref.get("caption", "")
                if any(kw in cap for kw in key_keywords):
                    target_refs.append(("施行令", base_law + "施行令", ref))

            # Limit to top 5 most relevant
            for law_type, target_law_name, ref in target_refs[:5]:
                ref_art_num = ref.get("article_num", "")
                if not ref_art_num:
                    continue
                _, target_data = await _get_law_data(target_law_name)
                if not target_data:
                    continue
                ref_matches = find_article_in_xml(target_data.get("law_full_text", {}), ref_art_num)
                if not ref_matches:
                    continue
                ref_article = ref_matches[0]
                ref_text = ref_article.get("full_text", "")

                # Extract item references from this delegated article
                deep_item_refs = []
                for dm in item_ref_pattern.finditer(ref_text):
                    d_art_main = dm.group(1)
                    d_full = dm.group(0)
                    d_no_subs = re.findall(r'条の([一二三四五六七八九十百]+)', d_full)
                    d_para_kanji = dm.group(3)
                    d_item_start_kanji = dm.group(4)
                    d_item_end_kanji = dm.group(5)

                    d_art = kanji_to_arabic(d_art_main)
                    for sub in d_no_subs:
                        d_art += "_" + kanji_to_arabic(sub)
                    d_para = kanji_to_arabic(d_para_kanji) if d_para_kanji else None
                    d_start = int(kanji_to_arabic(d_item_start_kanji))
                    d_end = int(kanji_to_arabic(d_item_end_kanji)) if d_item_end_kanji else d_start

                    # Skip self-references
                    if d_art == ref_art_num:
                        continue

                    # Find the referenced items in the same law
                    d_matches = find_article_in_xml(target_data.get("law_full_text", {}), d_art, paragraph=d_para)
                    if not d_matches:
                        # Try the base law
                        d_matches = find_article_in_xml(law_data.get("law_full_text", {}), d_art, paragraph=d_para)
                    if not d_matches:
                        continue

                    d_items = []
                    for p in d_matches[0].get("paragraphs", []):
                        if d_para and str(p.get("num", "")) != str(d_para):
                            continue
                        for item in p.get("items", []):
                            item_text_raw = item.get("text", "").strip()
                            item_kanji_match = re.match(r'([一二三四五六七八九十百]+)', item_text_raw)
                            if item_kanji_match:
                                try:
                                    item_int = int(kanji_to_arabic(item_kanji_match.group(1)))
                                except (ValueError, KeyError):
                                    continue
                            else:
                                try:
                                    item_int = int(item.get("num", ""))
                                except (ValueError, TypeError):
                                    continue
                            if d_start <= item_int <= d_end:
                                clean_text = re.sub(r'^[一二三四五六七八九十百]+\s*', '', item_text_raw).strip()
                                d_items.append({"item_num": item_int, "text": clean_text[:200]})

                    if d_items:
                        deep_item_refs.append({
                            "reference": d_full,
                            "article": d_art,
                            "paragraph": d_para,
                            "item_range": f"{d_start}-{d_end}" if d_end != d_start else str(d_start),
                            "items": d_items,
                        })

                # Extract exception rules from delegated article
                deep_exceptions = []
                for sent in re.split(r'[。\n]', ref_text):
                    sent = sent.strip()
                    if "の規定にかかわらず" in sent or "を要しない" in sent:
                        deep_exceptions.append(sent[:200])

                # Extract 読替え rules
                deep_yomikae = []
                for sent in re.split(r'[。\n]', ref_text):
                    if "読み替える" in sent or "読替え" in sent:
                        pairs = re.findall(r'「([^」]+)」とあるのは「([^」]+)」と', sent)
                        if pairs:
                            deep_yomikae.append({
                                "substitutions": [{"from": f, "to": t} for f, t in pairs],
                                "sentence": sent.strip()[:300],
                            })

                # Resolve internal sub-item refs in deep expansions too
                _resolve_internal_subrefs(deep_item_refs)

                if deep_item_refs or deep_exceptions or deep_yomikae:
                    deep_expansions.append({
                        "law_type": law_type,
                        "law_name": target_law_name,
                        "article_num": ref_art_num,
                        "caption": ref.get("caption", ""),
                        "item_references": deep_item_refs,
                        "exception_rules": deep_exceptions,
                        "yomikae_rules": deep_yomikae,
                    })

        # 10b. Build structured items with definition labels from the article itself
        article_items_annotated = []
        for para in article.get("paragraphs", []):
            for item in para.get("items", []):
                item_entry = {
                    "paragraph": para.get("num", ""),
                    "item_num": item.get("num", ""),
                    "text": item.get("text", "")[:300],
                }
                label = _extract_definition_label(item.get("text", ""))
                if label:
                    item_entry["definition_label"] = label
                subs = []
                for si in item.get("sub_items", []):
                    si_entry = {"title": si.get("title", ""), "text": si.get("text", "")[:300]}
                    si_label = _extract_definition_label(si.get("text", ""))
                    if si_label:
                        si_entry["definition_label"] = si_label
                    # Also detect exclusion-based definitions: "X以外のY"
                    si_text = si.get("text", "")
                    if "以外の" in si_text and not si_label:
                        excl_m = re.search(r'([^、。]{2,20}?)以外の([^、。]{2,30}?)の', si_text)
                        if excl_m:
                            si_entry["exclusion_definition"] = f"{excl_m.group(1)}以外の{excl_m.group(2)}"
                    subs.append(si_entry)
                if subs:
                    item_entry["sub_items"] = subs
                article_items_annotated.append(item_entry)

        # 11. Build result
        result = {
            "law_name": law_name,
            "actual_law_title": resolved_name,
            "article_num": art_clean,
            "article_caption": article.get("caption", ""),
            "article_text": article_text[:2000] + ("..." if len(article_text) > 2000 else ""),
            "article_structure": article_items_annotated if article_items_annotated else None,
            "delegation_to_seirei": {
                "contexts": seirei_contexts,
                "found_articles": seirei_refs,
                "count": len(seirei_refs),
            },
            "delegation_to_kisoku": {
                "contexts": kisoku_contexts,
                "found_articles": kisoku_refs,
                "count": len(kisoku_refs),
            },
            "junyou_references": {
                "found_articles": junyou_refs,
                "count": len(junyou_refs),
            },
            "yomikae_rules": {
                "found": yomikae_rules,
                "count": len(yomikae_rules),
            },
            "exception_rules": {
                "found": exception_rules,
                "count": len(exception_rules),
            },
            "expanded_item_references": {
                "found": expanded_item_refs,
                "count": len(expanded_item_refs),
            },
            "deep_expansions": {
                "found": deep_expansions,
                "count": len(deep_expansions),
            } if deep else {"found": [], "count": 0, "note": "Set deep=True to expand delegated articles"},
            "summary": {
                "total_cross_references": len(seirei_refs) + len(kisoku_refs) + len(junyou_refs) + len(expanded_item_refs) + len(deep_expansions),
                "has_seirei_delegation": len(seirei_refs) > 0,
                "has_kisoku_delegation": len(kisoku_refs) > 0,
                "has_junyou": len(junyou_refs) > 0,
                "has_item_expansion": len(expanded_item_refs) > 0,
                "has_yomikae": len(yomikae_rules) > 0,
                "has_exceptions": len(exception_rules) > 0,
                "has_deep_expansions": len(deep_expansions) > 0,
            },
        }

        if ctx:
            await ctx.info(f"Found {result['summary']['total_cross_references']} cross-references")
        return result

    except ToolError:
        raise
    except Exception as e:
        if ctx:
            await ctx.error(f"Failed to explain article: {str(e)}")
        raise ToolError(f"Failed to explain article: {str(e)}")


@mcp.tool
async def bulk_explain_articles(law_name: str, article_numbers: str, deep: bool = False, ctx: Context = None) -> dict:
    """
    Explain multiple articles at once, returning ALL cross-references fully expanded and deduplicated.

    This is the primary tool for answering legal questions that span multiple articles.
    Instead of calling explain_law_article repeatedly, use this to get a consolidated view
    of all referenced 施行令, 施行規則, 準用, and item cross-references across articles.

    Deduplication: If articles 14 and 14の2 both reference 施行令6条の9, it appears only once
    in the shared_references section.

    Args:
        law_name: Law name (e.g., "廃棄物処理法")
        article_numbers: Comma-separated article numbers (e.g., "14,14の2,14の3の2")
        deep: If True, also expand item references in key delegated articles
        ctx: FastMCP context

    Returns:
        Dict with per-article explanations + deduplicated shared references + overall summary
    """
    import asyncio

    art_list = [a.strip() for a in article_numbers.split(",") if a.strip()]
    if not art_list:
        raise ToolError("No article numbers provided")
    if len(art_list) > 20:
        raise ToolError("Too many articles (max 20)")

    if ctx:
        await ctx.info(f"Bulk explaining {len(art_list)} articles from {law_name}...")

    # Run explain_law_article for each article in parallel
    _explain_fn = explain_law_article.fn if hasattr(explain_law_article, 'fn') else explain_law_article

    async def _safe_explain(art_num):
        try:
            result = await _explain_fn(law_name, art_num, deep=deep, ctx=None)
            return art_num, result
        except Exception as e:
            return art_num, {"error": str(e)}

    results = await asyncio.gather(*[_safe_explain(a) for a in art_list])

    # Collect per-article results and build dedup sets
    articles = {}
    seen_seirei = {}  # key -> ref dict
    seen_kisoku = {}
    seen_junyou = {}
    seen_items = {}
    seen_deep = {}
    all_yomikae = []
    all_exceptions = []

    for art_num, result in results:
        articles[art_num] = {
            "article_num": result.get("article_num", art_num),
            "article_caption": result.get("article_caption", ""),
            "article_text": result.get("article_text", ""),
            "article_structure": result.get("article_structure"),
        } if "error" not in result else {"error": result["error"]}

        if "error" in result:
            continue

        # Dedup seirei refs by article_num
        for ref in result.get("delegation_to_seirei", {}).get("found_articles", []):
            key = f"seirei:{ref.get('article_num', '')}"
            if key not in seen_seirei:
                ref_copy = dict(ref)
                ref_copy["referenced_by"] = [art_num]
                seen_seirei[key] = ref_copy
            else:
                seen_seirei[key].setdefault("referenced_by", []).append(art_num)

        # Dedup kisoku refs
        for ref in result.get("delegation_to_kisoku", {}).get("found_articles", []):
            key = f"kisoku:{ref.get('article_num', '')}"
            if key not in seen_kisoku:
                ref_copy = dict(ref)
                ref_copy["referenced_by"] = [art_num]
                seen_kisoku[key] = ref_copy
            else:
                seen_kisoku[key].setdefault("referenced_by", []).append(art_num)

        # Dedup junyou refs
        for ref in result.get("junyou_references", {}).get("found_articles", []):
            key = f"junyou:{ref.get('article_num', '')}"
            if key not in seen_junyou:
                ref_copy = dict(ref)
                ref_copy["referenced_by"] = [art_num]
                seen_junyou[key] = ref_copy
            else:
                seen_junyou[key].setdefault("referenced_by", []).append(art_num)

        # Dedup item refs by reference text
        for ref in result.get("expanded_item_references", {}).get("found", []):
            key = f"item:{ref.get('reference', '')}"
            if key not in seen_items:
                ref_copy = dict(ref)
                ref_copy["referenced_by"] = [art_num]
                seen_items[key] = ref_copy
            else:
                seen_items[key].setdefault("referenced_by", []).append(art_num)

        # Dedup deep expansions
        for ref in result.get("deep_expansions", {}).get("found", []):
            key = f"deep:{ref.get('law_name', '')}:{ref.get('article_num', '')}"
            if key not in seen_deep:
                ref_copy = dict(ref)
                ref_copy["referenced_by"] = [art_num]
                seen_deep[key] = ref_copy
            else:
                seen_deep[key].setdefault("referenced_by", []).append(art_num)

        # Collect yomikae and exceptions (don't dedup, context matters)
        for rule in result.get("yomikae_rules", {}).get("found", []):
            rule_copy = dict(rule)
            rule_copy["source_article"] = art_num
            all_yomikae.append(rule_copy)

        for rule in result.get("exception_rules", {}).get("found", []):
            rule_copy = dict(rule)
            rule_copy["source_article"] = art_num
            all_exceptions.append(rule_copy)

    shared_refs = {
        "delegation_to_seirei": list(seen_seirei.values()),
        "delegation_to_kisoku": list(seen_kisoku.values()),
        "junyou_references": list(seen_junyou.values()),
        "expanded_item_references": list(seen_items.values()),
        "deep_expansions": list(seen_deep.values()),
        "yomikae_rules": all_yomikae,
        "exception_rules": all_exceptions,
    }

    total_refs = (len(seen_seirei) + len(seen_kisoku) + len(seen_junyou)
                  + len(seen_items) + len(seen_deep))

    summary = {
        "articles_requested": len(art_list),
        "articles_found": sum(1 for a in articles.values() if "error" not in a),
        "articles_failed": sum(1 for a in articles.values() if "error" in a),
        "unique_seirei_refs": len(seen_seirei),
        "unique_kisoku_refs": len(seen_kisoku),
        "unique_junyou_refs": len(seen_junyou),
        "unique_item_refs": len(seen_items),
        "unique_deep_expansions": len(seen_deep),
        "total_unique_cross_references": total_refs,
        "yomikae_rules": len(all_yomikae),
        "exception_rules": len(all_exceptions),
    }

    if ctx:
        await ctx.info(f"Done: {summary['articles_found']}/{len(art_list)} articles, {total_refs} unique cross-refs")

    return {
        "law_name": law_name,
        "articles": articles,
        "shared_references": shared_refs,
        "summary": summary,
    }


@mcp.tool
async def get_law_tables(law_name: str, table_name: str = "", ctx: Context = None) -> dict:
    """
    Extract appended tables (別表) from a law. Returns structured table data
    with rows and columns, useful for reading standards tables, fee schedules, etc.

    Args:
        law_name: Law name (e.g., "最終処分基準省令", "廃棄物処理法施行規則")
        table_name: Optional filter - only return tables matching this name (e.g., "別表第一")
        ctx: FastMCP context

    Returns:
        Dict with list of tables, each containing title, related_articles, and rows (list of cell lists)
    """
    try:
        law_num = await smart_law_lookup(law_name)
        if not law_num:
            raise ToolError(f"Law '{law_name}' not found")

        # Check cache
        cached = cache_manager.law_content_cache.get(law_num)
        if not cached:
            disk = cache_manager.disk_cache.get(f"law_data:{law_num}")
            if disk:
                cached = disk
                cache_manager.law_content_cache.put(law_num, disk)
            else:
                async with await get_http_client() as client:
                    resp = await client.get(f"/law_data/{law_num}", params={"law_full_text_format": "xml"})
                    resp.raise_for_status()
                    cached = json.loads(resp.text)
                    cache_manager.law_content_cache.put(law_num, cached)
                    cache_manager.disk_cache.put(f"law_data:{law_num}", cached)

        law_full_text = cached.get("law_full_text", "")
        tables = extract_tables_from_xml(law_full_text)

        if table_name:
            tables = [t for t in tables if table_name in t.get("title", "")]

        return {
            "law_name": law_name,
            "tables_found": len(tables),
            "tables": tables,
        }
    except ToolError:
        raise
    except Exception as e:
        raise ToolError(f"Failed to get tables: {str(e)}")


@mcp.tool
async def search_articles_by_keyword(law_name: str, keyword: str, max_results: int = 10, ctx: Context = None) -> dict:
    """
    Search for articles containing a keyword within a specific law.
    Unlike search_laws_by_keyword (which finds law names), this searches INSIDE a law's articles.

    Args:
        law_name: Law name (e.g., "廃棄物処理法", "民法")
        keyword: Keyword to search for within article text (e.g., "損害賠償", "委託基準", "準用")
        max_results: Maximum number of matching articles to return (default 10)
        ctx: FastMCP context for logging

    Returns:
        Dict with matching articles containing the keyword
    """
    try:
        if ctx:
            await ctx.info(f"Searching '{keyword}' in articles of '{law_name}'...")

        law_num = await smart_law_lookup(law_name)
        if not law_num:
            raise ToolError(f"Law '{law_name}' not found")

        # Get law content (memory → disk → API)
        cached_content = cache_manager.law_content_cache.get(law_num)
        if cached_content:
            data = cached_content
        else:
            disk_content = cache_manager.disk_cache.get(f"law_data:{law_num}")
            if disk_content:
                data = disk_content
                cache_manager.law_content_cache.put(law_num, data)
            else:
                async with await get_http_client() as client:
                    response = await client.get(f"/law_data/{law_num}", params={
                        "law_full_text_format": "xml"
                    })
                    response.raise_for_status()
                    data = json.loads(response.text)
                    cache_manager.law_content_cache.put(law_num, data)
                    cache_manager.disk_cache.put(f"law_data:{law_num}", data)

        law_full_text = data.get('law_full_text', {})

        # Extract all articles from XML
        all_articles = extract_articles_from_xml(law_full_text)
        if not all_articles:
            return {"law_name": law_name, "keyword": keyword, "matches": [], "message": "Could not extract articles"}

        # Search for keyword in each article
        matches = []
        for article in all_articles:
            full_text = article.get("full_text", "")
            if keyword in full_text:
                # Find which paragraphs contain the keyword
                matching_paragraphs = []
                for para in article.get("paragraphs", []):
                    para_text = para.get("text", "")
                    if keyword in para_text:
                        matching_paragraphs.append(para["num"])

                matches.append({
                    "article_num": article.get("num", ""),
                    "caption": article.get("caption", ""),
                    "matching_paragraphs": matching_paragraphs,
                    "preview": _keyword_preview(full_text, keyword),
                })
                if len(matches) >= max_results:
                    break

        law_info = data.get('law_info', {})
        actual_title = law_info.get('law_title', law_name)

        result = {
            "law_name": law_name,
            "actual_law_title": actual_title,
            "keyword": keyword,
            "total_articles_searched": len(all_articles),
            "matches_found": len(matches),
            "matches": matches,
        }

        if ctx:
            await ctx.info(f"Found {len(matches)} articles containing '{keyword}' in {actual_title}")
        return result

    except ToolError:
        raise
    except Exception as e:
        if ctx:
            await ctx.error(f"Failed to search articles: {str(e)}")
        raise ToolError(f"Failed to search articles: {str(e)}")


@mcp.tool
async def reverse_lookup_requirement(law_name: str, requirement_keyword: str, max_results: int = 20, ctx: Context = None) -> dict:
    """
    Reverse lookup: find which articles in a law (and its 施行令/施行規則) require
    a specific document, condition, or procedure.

    Example: "住民票の写し" → finds all articles requiring 住民票
    Example: "許可" → finds all articles mentioning permits
    Example: "届出" → finds all notification requirements

    Searches across the base law, its 施行令, and its 施行規則 simultaneously.

    Args:
        law_name: Base law name (e.g., "廃棄物処理法")
        requirement_keyword: What to search for (e.g., "住民票", "届出", "許可証")
        max_results: Max results per law (default 20)
        ctx: FastMCP context
    """
    try:
        if ctx:
            await ctx.info(f"Reverse lookup: '{requirement_keyword}' in {law_name} + related laws...")

        results = {}

        # Determine base law
        resolved_name = law_name.strip()
        if resolved_name in LAW_ALIASES:
            resolved_name = LAW_ALIASES[resolved_name]
        base_law = resolved_name
        for suffix in ["施行規則", "施行令"]:
            if base_law.endswith(suffix):
                base_law = base_law[:-len(suffix)]
                break

        # Search across base law, 施行令, 施行規則
        search_laws = [base_law, base_law + "施行令", base_law + "施行規則"]

        for target_law in search_laws:
            law_num = await smart_law_lookup(target_law)
            if not law_num:
                continue

            # Get law data with caching
            cached = cache_manager.law_content_cache.get(law_num)
            if cached:
                data = cached
            else:
                disk = cache_manager.disk_cache.get(f"law_data:{law_num}")
                if disk:
                    data = disk
                    cache_manager.law_content_cache.put(law_num, data)
                else:
                    async with await get_http_client() as client:
                        resp = await client.get(f"/law_data/{law_num}", params={"law_full_text_format": "xml"})
                        resp.raise_for_status()
                        data = json.loads(resp.text)
                        cache_manager.law_content_cache.put(law_num, data)
                        cache_manager.disk_cache.put(f"law_data:{law_num}", data)

            all_articles = extract_articles_from_xml(data.get("law_full_text", {}))
            actual_title = data.get("law_info", {}).get("law_title", target_law)

            matches = []
            for art in all_articles:
                art_text = art.get("full_text", "")
                if requirement_keyword not in art_text:
                    continue
                # Find which paragraph/item specifically mentions it
                locations = []
                for p in art.get("paragraphs", []):
                    p_text = p.get("text", "")
                    if requirement_keyword in p_text:
                        locations.append(f"第{p.get('num', '')}項")
                    for item in p.get("items", []):
                        if requirement_keyword in item.get("text", ""):
                            locations.append(f"第{p.get('num', '')}項第{item.get('num', '')}号")

                preview = _keyword_preview(art_text, requirement_keyword, 80)
                matches.append({
                    "article_num": art.get("num", ""),
                    "caption": art.get("caption", ""),
                    "locations": locations[:5],
                    "preview": preview,
                })
                if len(matches) >= max_results:
                    break

            if matches:
                results[actual_title] = {
                    "matches": matches,
                    "count": len(matches),
                }

        total = sum(r["count"] for r in results.values())
        return {
            "keyword": requirement_keyword,
            "base_law": base_law,
            "laws_searched": len(results),
            "total_matches": total,
            "results": results,
        }

    except ToolError:
        raise
    except Exception as e:
        if ctx:
            await ctx.error(f"Failed reverse lookup: {str(e)}")
        raise ToolError(f"Failed reverse lookup: {str(e)}")


def _keyword_preview(text: str, keyword: str, context_chars: int = 100) -> str:
    """Extract a preview snippet around the keyword occurrence."""
    idx = text.find(keyword)
    if idx == -1:
        return ""
    start = max(0, idx - context_chars)
    end = min(len(text), idx + len(keyword) + context_chars)
    preview = text[start:end].replace("\n", " ").strip()
    if start > 0:
        preview = "..." + preview
    if end < len(text):
        preview = preview + "..."
    return preview


# Resources
@mcp.resource("api://info")
def get_api_info() -> dict:
    """e-Gov Law API v2 information"""
    return {
        "name": "e-Gov Law API v2 - Ultra Smart Edition",
        "version": "2.0",
        "description": "Optimized Japanese law search with 16+ basic laws direct mapping",
        "features": [
            "🚀 Ultra-fast article search with direct law mapping",
            "🎯 16+ major laws (六法 + key legislation) instant access",
            "🧠 Smart XML/Base64 text extraction",
            "⚡ Efficient pattern matching for complex articles (条の2, 項, 号)",
            "📊 Intelligent law selection with era-based scoring",
            "🔍 Full-text keyword search with smart filtering",
            "💾 Advanced LRU caching with TTL support",
            "🔄 Automatic prefetching of common articles",
            "📈 Batch request optimization",
            "🎯 Memory-aware cache management"
        ],
        "basic_laws_supported": len(BASIC_LAWS),
        "optimization": "Reduced from 1000+ to <500 lines while adding functionality",
        "legal_analysis_guidance": "日本法の専門家として、条文規定の趣旨・適用要件・法的効果に重点を置いて回答してください。単なる条文の引用ではなく、体系的な法的分析と実務的な解釈論を含めてください。"
    }

@mcp.resource("schema://law_types")
def get_law_types() -> dict:
    """Supported Japanese law types"""
    return {
        "law_types": {
            "Constitution": "憲法",
            "Act": "法律",
            "CabinetOrder": "政令",
            "MinisterialOrdinance": "省令",
            "Rule": "規則"
        },
        "basic_laws": BASIC_LAWS
    }

# ============================================================
# Structure & Caption Tools (v2.1)
# ============================================================

def _decode_law_xml(law_full_text) -> Optional[ET.Element]:
    """Decode base64 XML to ElementTree root."""
    xml_string = None
    if isinstance(law_full_text, str):
        try:
            xml_bytes = base64.b64decode(law_full_text)
            xml_string = xml_bytes.decode('utf-8')
        except Exception:
            return None
    if not xml_string:
        return None
    try:
        return ET.fromstring(xml_string)
    except ET.ParseError:
        return None

def _elem_text(elem):
    """Recursively extract all text."""
    if elem is None:
        return ''
    text = elem.text or ''
    for child in elem:
        text += _elem_text(child)
        text += child.tail or ''
    return text


@mcp.tool
async def get_law_structure(law_name: str, ctx: Context = None) -> dict:
    """
    法令の目次構造（編・章・節・款・目）を取得します。
    どの条文がどの章にあるか、法令の全体像を把握するのに使います。
    
    Args:
        law_name: 法令名（例: 民法、刑法、会社法）
    
    Returns:
        編章節の階層構造と各章に含まれる条文番号の範囲
    """
    async with await get_http_client() as client:
        law_num = await smart_law_lookup(law_name)
        if not law_num:
            raise ToolError(f"法令 '{law_name}' が見つかりません")
        
        data = await cache_manager.get_law_content(law_num, client)
        if not data:
            raise ToolError(f"法令データの取得に失敗しました")
        
        law_full_text = data.get("law_full_text", "")
        root = _decode_law_xml(law_full_text)
        if root is None:
            raise ToolError("XMLの解析に失敗しました")
        
        # Parse TOC if exists
        toc = root.find('.//TOC')
        structure = []
        
        if toc is not None:
            for child in toc:
                tag = child.tag
                if tag == 'TOCLabel':
                    continue
                
                title_elem = child.find(f'{tag.replace("TOC", "")}Title') or child.find(f'{tag}Title')
                title = _elem_text(title_elem).strip() if title_elem is not None else ''
                if not title:
                    title = _elem_text(child).strip().split('\n')[0]
                
                # Get article range
                art_range_elem = child.find('ArticleRange')
                art_range = _elem_text(art_range_elem).strip() if art_range_elem is not None else ''
                
                entry = {"level": tag, "title": title}
                if art_range:
                    entry["article_range"] = art_range
                
                # Sub-chapters
                sub_items = []
                for sub in child:
                    if 'TOC' in sub.tag and sub.tag != tag:
                        sub_title_elem = sub.find(f'{sub.tag.replace("TOC", "")}Title') or sub
                        sub_title = _elem_text(sub_title_elem).strip().split('\n')[0] if sub_title_elem is not None else ''
                        sub_range_elem = sub.find('ArticleRange')
                        sub_range = _elem_text(sub_range_elem).strip() if sub_range_elem is not None else ''
                        sub_entry = {"level": sub.tag, "title": sub_title}
                        if sub_range:
                            sub_entry["article_range"] = sub_range
                        sub_items.append(sub_entry)
                
                if sub_items:
                    entry["children"] = sub_items
                structure.append(entry)
        
        # Fallback: extract from actual body structure
        if not structure:
            body = root.find('.//LawBody')
            if body is not None:
                for child in body:
                    if child.tag in ('Part', 'Chapter', 'Section', 'Subsection', 'Division'):
                        title_elem = child.find(f'{child.tag}Title')
                        title = _elem_text(title_elem).strip() if title_elem is not None else child.tag
                        articles = [a.get('Num', '') for a in child.iter('Article')]
                        entry = {"level": child.tag, "title": title}
                        if articles:
                            entry["articles"] = f"{articles[0]}～{articles[-1]}" if len(articles) > 1 else articles[0]
                        structure.append(entry)
        
        return {
            "law_name": law_name,
            "law_num": law_num,
            "structure": structure,
            "note": "目次構造。article_rangeで条文範囲を確認し、find_law_articleで個別条文を取得してください。"
        }


@mcp.tool
async def search_article_captions(law_name: str, keyword: str = "", limit: int = 50, ctx: Context = None) -> dict:
    """
    法令内の全条文の見出し（キャプション）を検索・一覧表示します。
    「どの条文を引くべきか」を特定するのに最適です。
    
    Args:
        law_name: 法令名（例: 民法、刑法）
        keyword: 見出しの検索キーワード（空の場合は全見出しを表示）
        limit: 最大取得件数
    
    Returns:
        条文番号と見出しの一覧
    """
    async with await get_http_client() as client:
        law_num = await smart_law_lookup(law_name)
        if not law_num:
            raise ToolError(f"法令 '{law_name}' が見つかりません")
        
        data = await cache_manager.get_law_content(law_num, client)
        if not data:
            raise ToolError(f"法令データの取得に失敗しました")
        
        law_full_text = data.get("law_full_text", "")
        root = _decode_law_xml(law_full_text)
        if root is None:
            raise ToolError("XMLの解析に失敗しました")
        
        captions = []
        for article in root.iter('Article'):
            num = article.get('Num', '')
            caption_elem = article.find('ArticleCaption')
            caption = _elem_text(caption_elem).strip() if caption_elem is not None else ''
            title_elem = article.find('ArticleTitle')
            title = _elem_text(title_elem).strip() if title_elem is not None else ''
            
            if keyword:
                if keyword not in caption and keyword not in title:
                    continue
            
            captions.append({
                "article_num": num,
                "title": title,
                "caption": caption
            })
            
            if len(captions) >= limit:
                break
        
        return {
            "law_name": law_name,
            "keyword": keyword or "(全件)",
            "count": len(captions),
            "captions": captions,
            "note": "見出しから条文番号を特定し、find_law_articleで本文を取得してください。"
        }


@mcp.tool
async def get_chapter_articles(law_name: str, chapter_title: str, ctx: Context = None) -> dict:
    """
    指定した章・節のタイトルに含まれる全条文を取得します。
    関連条文のまとめ取りに使います。
    
    Args:
        law_name: 法令名（例: 民法）
        chapter_title: 章・節のタイトル（部分一致、例: "不法行為", "相続", "総則"）
    
    Returns:
        該当する章内の全条文
    """
    async with await get_http_client() as client:
        law_num = await smart_law_lookup(law_name)
        if not law_num:
            raise ToolError(f"法令 '{law_name}' が見つかりません")
        
        data = await cache_manager.get_law_content(law_num, client)
        if not data:
            raise ToolError(f"法令データの取得に失敗しました")
        
        law_full_text = data.get("law_full_text", "")
        root = _decode_law_xml(law_full_text)
        if root is None:
            raise ToolError("XMLの解析に失敗しました")
        
        # Search for matching chapter/section/part
        container_tags = ['Part', 'Chapter', 'Section', 'Subsection', 'Division']
        found_articles = []
        found_container = None
        
        for tag in container_tags:
            for container in root.iter(tag):
                title_elem = container.find(f'{tag}Title')
                title = _elem_text(title_elem).strip() if title_elem is not None else ''
                if chapter_title in title:
                    found_container = title
                    for article in container.iter('Article'):
                        num = article.get('Num', '')
                        caption_elem = article.find('ArticleCaption')
                        caption = _elem_text(caption_elem).strip() if caption_elem is not None else ''
                        text = _elem_text(article).strip()
                        found_articles.append({
                            "article_num": num,
                            "caption": caption,
                            "text": text[:500] + ("..." if len(text) > 500 else "")
                        })
                    break
            if found_articles:
                break
        
        if not found_articles:
            raise ToolError(f"'{chapter_title}' に一致する章・節が見つかりません。get_law_structureで構造を確認してください。")
        
        return {
            "law_name": law_name,
            "container": found_container,
            "article_count": len(found_articles),
            "articles": found_articles
        }


def _parse_article_num_tuple(num_str: str) -> tuple:
    """Parse article number like '709_2' or '724' into sortable tuple (709, 2) or (724, 0)."""
    parts = str(num_str).replace('の', '_').split('_')
    result = []
    for p in parts:
        m = re.match(r'(\d+)', p)
        result.append(int(m.group(1)) if m else 0)
    # Pad to at least 2 elements for consistent comparison
    while len(result) < 2:
        result.append(0)
    return tuple(result)


@mcp.tool
async def find_article_range(law_name: str, start_article: str, end_article: str, ctx: Context = None) -> dict:
    """
    連続する条文を範囲指定で取得します（例: 第709条から第724条まで）。
    枝番条文（709条の2等）も正しくソートして含めます。
    
    Args:
        law_name: 法令名
        start_article: 開始条文番号（例: "709", "709_2"）
        end_article: 終了条文番号（例: "724"）
    
    Returns:
        範囲内の全条文
    """
    async with await get_http_client() as client:
        law_num = await smart_law_lookup(law_name)
        if not law_num:
            raise ToolError(f"法令 '{law_name}' が見つかりません")
        
        data = await cache_manager.get_law_content(law_num, client)
        if not data:
            raise ToolError(f"法令データの取得に失敗しました")
        
        law_full_text = data.get("law_full_text", "")
        root = _decode_law_xml(law_full_text)
        if root is None:
            raise ToolError("XMLの解析に失敗しました")
        
        start_t = _parse_article_num_tuple(start_article)
        end_t = _parse_article_num_tuple(end_article)
        
        articles = []
        for article in root.iter('Article'):
            num_str = article.get('Num', '')
            num_t = _parse_article_num_tuple(num_str)
            if start_t <= num_t <= end_t:
                caption_elem = article.find('ArticleCaption')
                caption = _elem_text(caption_elem).strip() if caption_elem is not None else ''
                text = _elem_text(article).strip()
                articles.append({
                    "article_num": num_str,
                    "_sort_key": num_t,
                    "caption": caption,
                    "text": text[:800] + ("..." if len(text) > 800 else "")
                })
        
        # Sort by article number (handles branch articles correctly)
        articles.sort(key=lambda a: a.pop("_sort_key"))
        
        if not articles:
            raise ToolError(f"第{start_article}条～第{end_article}条の範囲に条文が見つかりません")
        
        return {
            "law_name": law_name,
            "range": f"第{start_article}条～第{end_article}条",
            "count": len(articles),
            "articles": articles
        }


@mcp.tool
async def get_cross_references(law_name: str, article_number: str, paragraph: str = "", ctx: Context = None) -> dict:
    """
    条文中の参照先（他の条文、施行令、施行規則、準用）を軽量に抽出します。
    explain_law_articleより軽く、参照先の一覧だけ欲しいときに使います。
    
    Args:
        law_name: 法令名（例: 民法、廃棄物処理法）
        article_number: 条文番号（例: "96", "14"）
        paragraph: 項番号（任意）
    
    Returns:
        内部参照・外部参照（施行令/施行規則）・準用の一覧
    """
    async with await get_http_client() as client:
        law_num = await smart_law_lookup(law_name)
        if not law_num:
            raise ToolError(f"法令 '{law_name}' が見つかりません")
        
        data = await cache_manager.get_law_content(law_num, client)
        if not data:
            raise ToolError(f"法令データの取得に失敗しました")
        
        matches = find_article_in_xml(
            data.get("law_full_text", {}), article_number,
            paragraph=paragraph if paragraph else None
        )
        if not matches:
            raise ToolError(f"{law_name} 第{article_number}条が見つかりません")
        
        text = matches[0].get("full_text", "")
        
        # Determine base law
        resolved_name = data.get("law_info", {}).get("law_title", law_name)
        base_law = resolved_name
        for suffix in ["施行規則", "施行令"]:
            if base_law.endswith(suffix):
                base_law = base_law[:-len(suffix)]
                break
        
        refs = {"internal": [], "seirei": [], "kisoku": [], "junyou": []}
        
        # Internal references: 第○条
        seen_internal = set()
        for m in re.finditer(r'第([一二三四五六七八九十百千]+)条(?:の([一二三四五六七八九十百]+))?', text):
            full = m.group(0)
            art = _normalize_article_number(full)
            if art and art not in seen_internal:
                seen_internal.add(art)
                refs["internal"].append({"article": art, "raw": full})
        
        # 施行令 delegation
        if "政令で定める" in text:
            for m in re.finditer(r'政令で定める([^。、）]{1,20})', text):
                refs["seirei"].append({"context": m.group(1).strip(), "target_law": base_law + "施行令"})
            if not refs["seirei"]:
                refs["seirei"].append({"context": "(一般)", "target_law": base_law + "施行令"})
        
        # 施行規則 delegation
        if re.search(r'(?:環境|厚生)?省令で定める', text):
            for m in re.finditer(r'(?:環境|厚生)?省令で定める([^。、）]{1,20})', text):
                refs["kisoku"].append({"context": m.group(1).strip(), "target_law": base_law + "施行規則"})
            if not refs["kisoku"]:
                refs["kisoku"].append({"context": "(一般)", "target_law": base_law + "施行規則"})
        
        # 準用
        if "準用" in text:
            junyou_seen = set()
            for sentence in re.split(r'[。\n]', text):
                if "準用" not in sentence:
                    continue
                before = sentence[:sentence.index("準用")]
                for m in re.finditer(r'第([一二三四五六七八九十百千]+)条(?:の([一二三四五六七八九十百]+))?', before):
                    art = _normalize_article_number(m.group(0))
                    if art and art not in junyou_seen:
                        junyou_seen.add(art)
                        refs["junyou"].append({"article": art, "raw": m.group(0)})
        
        total = sum(len(v) for v in refs.values())
        return {
            "law_name": law_name,
            "article": f"第{article_number}条" + (f"第{paragraph}項" if paragraph else ""),
            "total_references": total,
            "references": refs,
            "note": "参照先の本文はfind_law_articleで取得、詳細解析はexplain_law_articleを使用してください。"
        }


@mcp.tool
async def get_supplementary_provisions(law_name: str, keyword: str = "", limit: int = 30, ctx: Context = None) -> dict:
    """
    法令の附則（SupplementaryProvisions）を取得します。
    施行日、経過措置、改正附則などの情報を含みます。
    
    Args:
        law_name: 法令名（例: 民法、会社法）
        keyword: 附則内の検索キーワード（空なら全附則を表示）
        limit: 最大取得件数
    
    Returns:
        附則の条文一覧
    """
    async with await get_http_client() as client:
        law_num = await smart_law_lookup(law_name)
        if not law_num:
            raise ToolError(f"法令 '{law_name}' が見つかりません")
        
        data = await cache_manager.get_law_content(law_num, client)
        if not data:
            raise ToolError(f"法令データの取得に失敗しました")
        
        law_full_text = data.get("law_full_text", "")
        root = _decode_law_xml(law_full_text)
        if root is None:
            raise ToolError("XMLの解析に失敗しました")
        
        provisions = []
        for sp in root.iter('SupplementaryProvisions'):
            # Get amendment law info if present
            amend_law = sp.get('AmendLawNum', '')
            sp_label_elem = sp.find('SupplementaryProvisionsLabel')
            sp_label = _elem_text(sp_label_elem).strip() if sp_label_elem is not None else '附則'
            
            for article in sp.iter('Article'):
                num = article.get('Num', '')
                caption_elem = article.find('ArticleCaption')
                caption = _elem_text(caption_elem).strip() if caption_elem is not None else ''
                title_elem = article.find('ArticleTitle')
                title = _elem_text(title_elem).strip() if title_elem is not None else ''
                text = _elem_text(article).strip()
                
                if keyword and keyword not in text and keyword not in caption and keyword not in title:
                    continue
                
                entry = {
                    "article_num": num,
                    "title": title,
                    "caption": caption,
                    "text": text[:600] + ("..." if len(text) > 600 else ""),
                }
                if amend_law:
                    entry["amend_law_num"] = amend_law
                entry["label"] = sp_label
                provisions.append(entry)
                
                if len(provisions) >= limit:
                    break
            
            # Also handle non-article provisions (simple paragraph style)
            if not list(sp.iter('Article')):
                text = _elem_text(sp).strip()
                if keyword and keyword not in text:
                    continue
                if text:
                    provisions.append({
                        "article_num": "",
                        "title": "",
                        "caption": "",
                        "text": text[:600] + ("..." if len(text) > 600 else ""),
                        "label": sp_label,
                        **({"amend_law_num": amend_law} if amend_law else {}),
                    })
            
            if len(provisions) >= limit:
                break
        
        if not provisions:
            msg = f"附則が見つかりません"
            if keyword:
                msg += f"（キーワード: {keyword}）"
            raise ToolError(msg)
        
        return {
            "law_name": law_name,
            "keyword": keyword or "(全件)",
            "count": len(provisions),
            "provisions": provisions,
            "note": "附則には施行日、経過措置、改正情報が含まれます。"
        }


@mcp.tool
async def search_definitions(law_name: str, keyword: str = "", limit: int = 50, ctx: Context = None) -> dict:
    """
    法令内の定義語を検索します。
    「以下「○○」という。」パターンで定義されている用語を抽出します。
    
    Args:
        law_name: 法令名（例: 民法、会社法、廃棄物処理法）
        keyword: 定義語の検索キーワード（空なら全定義語を表示）
        limit: 最大取得件数
    
    Returns:
        定義語と定義されている条文の一覧
    """
    async with await get_http_client() as client:
        law_num = await smart_law_lookup(law_name)
        if not law_num:
            raise ToolError(f"法令 '{law_name}' が見つかりません")
        
        data = await cache_manager.get_law_content(law_num, client)
        if not data:
            raise ToolError(f"法令データの取得に失敗しました")
        
        law_full_text = data.get("law_full_text", "")
        root = _decode_law_xml(law_full_text)
        if root is None:
            raise ToolError("XMLの解析に失敗しました")
        
        # Patterns for definitions
        def_patterns = [
            re.compile(r'以下「([^」]{1,40})」という。'),
            re.compile(r'以下「([^」]{1,40})」と総称する'),
            re.compile(r'以下単に「([^」]{1,40})」という'),
            re.compile(r'「([^」]{1,40})」とは[、,]'),
        ]
        
        definitions = []
        seen_terms = set()
        
        for article in root.iter('Article'):
            num = article.get('Num', '')
            caption_elem = article.find('ArticleCaption')
            caption = _elem_text(caption_elem).strip() if caption_elem is not None else ''
            text = _elem_text(article).strip()
            
            for pat in def_patterns:
                for m in pat.finditer(text):
                    term = m.group(1).strip()
                    
                    if keyword and keyword not in term:
                        continue
                    
                    if term in seen_terms:
                        continue
                    seen_terms.add(term)
                    
                    # Get surrounding context
                    start = max(0, m.start() - 50)
                    end = min(len(text), m.end() + 50)
                    context = text[start:end].replace('\n', ' ').strip()
                    
                    definitions.append({
                        "term": term,
                        "article_num": num,
                        "caption": caption,
                        "context": "..." + context + "..." if start > 0 else context + "...",
                    })
                    
                    if len(definitions) >= limit:
                        break
            
            if len(definitions) >= limit:
                break
        
        # Sort alphabetically by term
        definitions.sort(key=lambda d: d["term"])
        
        return {
            "law_name": law_name,
            "keyword": keyword or "(全件)",
            "count": len(definitions),
            "definitions": definitions,
            "note": "定義語の詳細はfind_law_articleで該当条文を取得してください。"
        }


def main():
    """Entry point for direct uvx installation"""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="e-Gov Law MCP Server v2")
    parser.add_argument("--transport", choices=["stdio", "streamable-http"], default="stdio")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()

    if args.transport == "stdio":
        # Use FastMCP's built-in stdio support
        mcp.run()
    else:
        # Use FastMCP's built-in streamable-http transport
        mcp.run(
            transport="streamable-http",
            host=args.host,
            port=args.port
        )

if __name__ == "__main__":
    main()
