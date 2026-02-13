"""
Prompt Loader for e-Gov Law MCP Server

This module provides a clean way to load prompts from external files,
separating prompt content from business logic.
"""

import logging
from pathlib import Path

logger = logging.getLogger(__name__)


class PromptLoader:
    """
    A class to load prompts from external files.

    This allows for better maintainability and separation of concerns
    by keeping prompt templates separate from the main application logic.
    """

    def __init__(self, prompts_dir: str = "prompts"):
        """
        Initialize the PromptLoader.

        Args:
            prompts_dir: Directory containing prompt files (default: "prompts")
        """
        self.prompts_dir = Path(prompts_dir)
        self._cache: dict[str, str] = {}

        # Ensure prompts directory exists
        if not self.prompts_dir.exists():
            logger.warning(f"Prompts directory '{self.prompts_dir}' does not exist")

    def load_prompt(self, prompt_name: str) -> str:
        """
        Load a prompt from a file.

        Args:
            prompt_name: Name of the prompt file (without extension)

        Returns:
            The prompt content as a string

        Raises:
            FileNotFoundError: If the prompt file doesn't exist
            IOError: If there's an error reading the file
        """
        # Check cache first
        if prompt_name in self._cache:
            return self._cache[prompt_name]

        # Try to load from file
        prompt_file = self.prompts_dir / f"{prompt_name}.md"

        if not prompt_file.exists():
            raise FileNotFoundError(f"Prompt file '{prompt_file}' not found")

        try:
            with open(prompt_file, encoding='utf-8') as f:
                content = f.read().strip()

            # Cache the result
            self._cache[prompt_name] = content
            logger.info(f"Loaded prompt '{prompt_name}' from {prompt_file}")

            return content

        except Exception as e:
            logger.error(f"Error loading prompt '{prompt_name}': {e}")
            raise OSError(f"Failed to load prompt '{prompt_name}': {e}")

    def get_legal_analysis_instruction(self) -> str:
        """
        Get the legal analysis instruction prompt.

        Returns:
            The legal analysis instruction as a string
        """
        try:
            return self.load_prompt("legal_analysis")
        except (OSError, FileNotFoundError) as e:
            logger.warning(f"Failed to load legal_analysis prompt: {e}")
            # Return fallback instruction for backward compatibility
            return self._get_fallback_legal_analysis_instruction()

    def _get_fallback_legal_analysis_instruction(self) -> str:
        """
        Fallback legal analysis instruction for backward compatibility.

        Returns:
            The hardcoded legal analysis instruction
        """
        return """【重要】日本の法律の専門家として、この条文について以下のように回答してください：

■ 0. 検索対象法律の確認（必須）
検索結果の「actual_law_title」と「law_number」を確認し、正しい法律で検索されたことを明記してください。
「name_conversion_applied」がtrueの場合は、略称から正式名称への変換が行われたことも説明してください。

例：
「民法（明治二十九年法律第八十九号）の第百九十二条について分析します。」
「労基法として検索されましたが、正式名称は労働基準法です。」

■ 1. 条文の正確な全文引用（必須）
検索結果の「articles」に含まれる条文テキストを、一字一句正確に引用してください。条文番号、項、号まで含めて完全に表示してください。

例：
「第百九十二条　取引行為によって、平穏に、かつ、公然と動産の占有を始めた者は、善意であり、かつ、過失がないときは、即時にその動産について行使する権利を取得する。」

■ 2. 法的分析（条文を引用しながら説明）
上記で引用した条文の重要な文言を「」で再度引用しながら、以下の観点から詳細に分析してください：
・条文の趣旨（立法目的・背景）
・要件（適用要件・前提条件）
・法的効果（権利義務の発生・変更・消滅）
・実務上の注意点・関連判例
・他の条文との関係性

例：「取引行為によって」という要件は有償取引を前提とし、「善意であり、かつ、過失がない」という要件は主観的要件を示します。

正式法律名の確認、条文の正確な引用、法的分析を組み合わせた専門的で実用的な回答をお願いします。"""

    def clear_cache(self):
        """Clear the prompt cache."""
        self._cache.clear()
        logger.info("Prompt cache cleared")

    def list_available_prompts(self) -> list:
        """
        List available prompt files.

        Returns:
            List of available prompt names (without extensions)
        """
        if not self.prompts_dir.exists():
            return []

        return [f.stem for f in self.prompts_dir.glob("*.md")]
