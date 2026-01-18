# Copyright (c) 2025 CoReason, Inc.
#
# This software is proprietary and dual-licensed.
# Licensed under the Prosperity Public License 3.0 (the "License").
# A copy of the license is available at https://prosperitylicense.com/versions/3.0.0
# For details, see the LICENSE file.
# Commercial use beyond a 30-day trial requires a separate license.
#
# Source Code: https://github.com/CoReason-AI/coreason_archive

import re
from typing import List, Pattern, Tuple

from coreason_archive.interfaces import EntityExtractor
from coreason_archive.utils.logger import logger


class RegexEntityExtractor(EntityExtractor):
    """
    A heuristic entity extractor using Regular Expressions.
    Extracts entities from text and formats them as 'Type:Value' strings.
    """

    def __init__(self, patterns: List[Tuple[str, str]] | None = None) -> None:
        r"""
        Initialize the extractor with a list of (Entity Type, Regex Pattern) tuples.

        Args:
            patterns: A list of tuples where the first element is the Entity Type
                      (e.g., "Project") and the second is the regex pattern
                      (e.g., r"Project\s+(\w+)").
                      The regex should contain at least one capturing group.
        """
        if patterns is None:
            # Default patterns based on PRD scenarios
            patterns = [
                ("Project", r"(?i)\bProject[:\s]+([\w-]+)"),
                ("User", r"(?i)\bUser[:\s]+([\w-]+)"),
                ("Dept", r"(?i)\b(?:Dept|Department)[:\s]+([\w-]+)"),
                ("Client", r"(?i)\bClient[:\s]+([\w-]+)"),
                ("Drug", r"(?i)\bDrug[:\s]+([\w-]+)"),
                ("Concept", r"(?i)\bConcept[:\s]+([\w-]+)"),
            ]

        self.patterns: List[Tuple[str, Pattern[str]]] = [(type_, re.compile(pattern)) for type_, pattern in patterns]

    async def extract(self, text: str) -> List[str]:
        """
        Extracts entities from the given text asynchronously.

        Args:
            text: The text to analyze.

        Returns:
            A list of unique entity strings in 'Type:Value' format.
        """
        entities = set()

        for entity_type, regex in self.patterns:
            matches = regex.findall(text)
            for match in matches:
                # regex.findall returns a string or tuple of strings.
                # We expect the first capturing group to be the value.
                val = match if isinstance(match, str) else match[0]

                # Clean up the value (trim whitespace)
                val = val.strip()

                if val:
                    entity_str = f"{entity_type}:{val}"
                    entities.add(entity_str)

        result = list(entities)
        logger.debug(f"Extracted {len(result)} entities: {result}")
        return result
