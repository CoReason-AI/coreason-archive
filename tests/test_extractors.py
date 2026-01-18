# Copyright (c) 2025 CoReason, Inc.
#
# This software is proprietary and dual-licensed.
# Licensed under the Prosperity Public License 3.0 (the "License").
# A copy of the license is available at https://prosperitylicense.com/versions/3.0.0
# For details, see the LICENSE file.
# Commercial use beyond a 30-day trial requires a separate license.
#
# Source Code: https://github.com/CoReason-AI/coreason_archive

import pytest

from coreason_archive.extractors import RegexEntityExtractor


@pytest.mark.asyncio
async def test_extract_defaults() -> None:
    """Test extraction using default patterns."""
    extractor = RegexEntityExtractor()
    text = "User Alice is working on Project Apollo for Client Nasa."

    entities = await extractor.extract(text)

    assert len(entities) == 3
    assert "User:Alice" in entities
    assert "Project:Apollo" in entities
    assert "Client:Nasa" in entities


@pytest.mark.asyncio
async def test_extract_case_insensitive() -> None:
    """Test that default patterns are case insensitive."""
    extractor = RegexEntityExtractor()
    text = "user bob project zeus dept hr"

    entities = await extractor.extract(text)

    assert "User:bob" in entities
    assert "Project:zeus" in entities
    assert "Dept:hr" in entities


@pytest.mark.asyncio
async def test_extract_with_colon() -> None:
    """Test extraction when input already uses colons."""
    extractor = RegexEntityExtractor()
    text = "We have User:Charlie and Project:Gemini."

    entities = await extractor.extract(text)

    assert "User:Charlie" in entities
    assert "Project:Gemini" in entities


@pytest.mark.asyncio
async def test_extract_multiple_occurrences() -> None:
    """Test that entities are deduplicated."""
    extractor = RegexEntityExtractor()
    text = "Project Apollo is important. Project Apollo must succeed."

    entities = await extractor.extract(text)

    assert len(entities) == 1
    assert "Project:Apollo" in entities


@pytest.mark.asyncio
async def test_extract_no_matches() -> None:
    """Test extraction with text containing no entities."""
    extractor = RegexEntityExtractor()
    text = "Just some random text with no keywords."

    entities = await extractor.extract(text)

    assert entities == []


@pytest.mark.asyncio
async def test_custom_patterns() -> None:
    """Test extraction with custom patterns."""
    # Pattern to extract standard ticket numbers like TICKET-123
    custom_patterns = [("Ticket", r"(TICKET-\d+)"), ("Email", r"([\w\.-]+@[\w\.-]+)")]
    extractor = RegexEntityExtractor(patterns=custom_patterns)
    text = "Reference TICKET-999 and email test@example.com"

    entities = await extractor.extract(text)

    assert "Ticket:TICKET-999" in entities
    assert "Email:test@example.com" in entities
    # Default patterns should not work
    assert "User:Reference" not in entities


@pytest.mark.asyncio
async def test_complex_drug_example() -> None:
    """Test the Drug entity example from the PRD."""
    extractor = RegexEntityExtractor()
    text = "Issues with Drug Cisplatin regarding crystallization."

    entities = await extractor.extract(text)

    assert "Drug:Cisplatin" in entities


# --- Edge Cases & Complex Scenarios ---


@pytest.mark.asyncio
async def test_unicode_entities() -> None:
    r"""
    Test extraction of non-ASCII entities.
    Python's \w matches Unicode alphanumeric characters.
    """
    extractor = RegexEntityExtractor()
    text = "User:Jülès is leading Project:Ωmega."

    entities = await extractor.extract(text)

    assert "User:Jülès" in entities
    assert "Project:Ωmega" in entities


@pytest.mark.asyncio
async def test_whitespace_and_formatting() -> None:
    """Test irregular whitespace handling (tabs, newlines, multiple spaces)."""
    extractor = RegexEntityExtractor()
    # "User\tAlice" should match "User Alice" pattern logic (since \s matches \t)
    text = """
    User\tAlice
    Project:    Big-Data
    Dept
    Finance
    """

    entities = await extractor.extract(text)

    assert "User:Alice" in entities
    assert "Project:Big-Data" in entities
    assert "Dept:Finance" in entities


@pytest.mark.asyncio
async def test_false_positive_awareness() -> None:
    """
    Document behavior for sentences where keywords are used as nouns.
    The simple regex approach will extract these as entities.
    This test ensures we are aware of this limitation (it's a feature of the heuristic).
    """
    extractor = RegexEntityExtractor()
    text = "Good Project management is essential. User behavior is complex."

    entities = await extractor.extract(text)

    # These are technically false positives in an NLP sense,
    # but correct behavior for this Regex extractor.
    assert "Project:management" in entities
    assert "User:behavior" in entities


@pytest.mark.asyncio
async def test_custom_pattern_capturing_whitespace() -> None:
    """Test that the extractor strips whitespace from captured groups."""
    # Pattern capturing everything after colon, including potential spaces
    patterns = [("Title", r"Title:\s*(.*)")]
    extractor = RegexEntityExtractor(patterns=patterns)

    text = "Title:   The Great Gatsby   "

    entities = await extractor.extract(text)

    # Should be stripped
    assert "Title:The Great Gatsby" in entities


@pytest.mark.asyncio
async def test_complex_messy_scenario() -> None:
    """
    Test a complex, messy input simulating a raw email or log dump.
    Mixes multiple entity types, noise, and formatting.
    """
    extractor = RegexEntityExtractor()
    text = """
    From: User:admin
    Date: 2024-01-01
    Subject: Re: Project Alpha-One Update

    Hi Team,

    regarding Project:Alpha-One (and Project Beta), we need to consult
    Dept:Legal and Dept:Compliance immediately.

    Client:MegaCorp is waiting.

    Also, User:John_Doe mentioned Drug:Aspirin side effects.

    Thanks.
    """

    entities = await extractor.extract(text)

    expected = {
        "User:admin",
        "Project:Alpha-One",
        # "Project Beta" might be missed if it doesn't have a colon and text has ) immediately?
        # Pattern: r"(?i)\bProject[:\s]+([\w-]+)"
        # "Project Beta)" -> "Beta" matches \w. ")" stops it. So "Project:Beta"
        "Project:Beta",
        "Dept:Legal",
        "Dept:Compliance",
        "Client:MegaCorp",
        "User:John_Doe",
        "Drug:Aspirin",
    }

    # Convert list to set for comparison
    entity_set = set(entities)

    # Check that all expected entities are present
    assert expected.issubset(entity_set)

    # Note: "Project Alpha-One" is extracted as "Project:Alpha-One" because \w- includes hyphens.
