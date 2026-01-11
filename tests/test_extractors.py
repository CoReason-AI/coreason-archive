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
