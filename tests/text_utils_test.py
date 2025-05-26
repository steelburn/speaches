from speaches.text_utils import (
    srt_format_timestamp,
    vtt_format_timestamp,
    strip_markdown_emphasis,
)


def test_srt_format_timestamp() -> None:
    assert srt_format_timestamp(0.0) == "00:00:00,000"
    assert srt_format_timestamp(1.0) == "00:00:01,000"
    assert srt_format_timestamp(1.234) == "00:00:01,234"
    assert srt_format_timestamp(60.0) == "00:01:00,000"
    assert srt_format_timestamp(61.0) == "00:01:01,000"
    assert srt_format_timestamp(61.234) == "00:01:01,234"
    assert srt_format_timestamp(3600.0) == "01:00:00,000"
    assert srt_format_timestamp(3601.0) == "01:00:01,000"
    assert srt_format_timestamp(3601.234) == "01:00:01,234"
    assert srt_format_timestamp(23423.4234) == "06:30:23,423"


def test_vtt_format_timestamp() -> None:
    assert vtt_format_timestamp(0.0) == "00:00:00.000"
    assert vtt_format_timestamp(1.0) == "00:00:01.000"
    assert vtt_format_timestamp(1.234) == "00:00:01.234"
    assert vtt_format_timestamp(60.0) == "00:01:00.000"
    assert vtt_format_timestamp(61.0) == "00:01:01.000"
    assert vtt_format_timestamp(61.234) == "00:01:01.234"
    assert vtt_format_timestamp(3600.0) == "01:00:00.000"
    assert vtt_format_timestamp(3601.0) == "01:00:01.000"
    assert vtt_format_timestamp(3601.234) == "01:00:01.234"
    assert vtt_format_timestamp(23423.4234) == "06:30:23.423"


def test_strip_markdown_emphasis() -> None:
    assert strip_markdown_emphasis("Hello my name is **Jon**") == "Hello my name is Jon"
    assert strip_markdown_emphasis("I *really* like this") == "I really like this"
    assert strip_markdown_emphasis("This is __underlined__") == "This is underlined"
    assert strip_markdown_emphasis("This is _italic_") == "This is italic"
    assert strip_markdown_emphasis("Mixed **bold** and *italic* text") == "Mixed bold and italic text"
    assert strip_markdown_emphasis("No markdown here") == "No markdown here"
    assert strip_markdown_emphasis("**Bold** at the *beginning* and _end_ of **text**") == "Bold at the beginning and end of text"
    assert strip_markdown_emphasis("Nested **bold *with italic* inside**") == "Nested bold with italic inside"
