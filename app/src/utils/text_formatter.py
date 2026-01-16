"""Text formatting utilities for Qt tooltips and user interface display.

Provides functions for intelligent text wrapping, HTML formatting, and
parameter description formatting optimized for Qt tooltip display.

Author: Hugo Demule
Date: January 2026
"""

import re
import textwrap


def format_tooltip_text(text: str, width: int = 60, use_html: bool = True) -> str:
    """Format text for Qt tooltips with intelligent wrapping.

    Args:
        text (str): Input text to format.
        width (int): Maximum line width in characters. Default: 60.
        use_html (bool): Whether to use HTML formatting. Default: True.

    Returns:
        str: Formatted text suitable for Qt tooltips.
    """
    if not text:
        return text

    if use_html:
        # For HTML, let CSS handle the wrapping - just clean up the text
        return format_as_html(text)
    else:
        # For plain text, do manual wrapping
        paragraphs = text.split("\n\n")
        formatted_paragraphs = []

        for paragraph in paragraphs:
            # Remove single \n within paragraphs and replace with spaces
            clean_paragraph = re.sub(r"(?<!\n)\n(?!\n)", " ", paragraph.strip())

            # Wrap the paragraph while preserving word boundaries
            wrapped_lines = textwrap.fill(
                clean_paragraph,
                width=width,
                break_long_words=False,
                break_on_hyphens=True,
                expand_tabs=False,
            )

            formatted_paragraphs.append(wrapped_lines)

        # Join paragraphs back together
        return "\n\n".join(formatted_paragraphs)


def format_as_html(text: str) -> str:
    """Convert plain text to HTML format for Qt tooltip display.

    Args:
        text (str): Plain text to convert.

    Returns:
        str: HTML formatted text with proper escaping and styling.
    """
    # Escape HTML special characters
    text = text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")

    # Split into paragraphs (preserve \n\n breaks)
    paragraphs = text.split("\n\n")
    html_paragraphs = []

    for paragraph in paragraphs:
        # Clean up the paragraph - remove single \n and extra whitespace
        clean_paragraph = re.sub(r"(?<!\n)\n(?!\n)", " ", paragraph.strip())
        clean_paragraph = re.sub(r"\s+", " ", clean_paragraph)  # Normalize whitespace

        # Check if this looks like an "Intuition:" paragraph
        if clean_paragraph.startswith("Intuition:"):
            # Make "Intuition:" bold and add some top margin
            clean_paragraph = re.sub(
                r"^Intuition:", "<b>Intuition:</b>", clean_paragraph
            )
            html_paragraphs.append(
                f'<p style="margin-top: 10px;">{clean_paragraph}</p>'
            )
        else:
            html_paragraphs.append(f"<p>{clean_paragraph}</p>")

    # Wrap everything in a div with CSS word-wrap properties
    html_content = "".join(html_paragraphs)
    return f'<div style="max-width: 400px; word-wrap: break-word; white-space: normal;">{html_content}</div>'


def format_parameter_description(description: str) -> str:
    """Format parameter descriptions with standard + intuition format.

    Args:
        description (str): Parameter description with potential "Intuition:" section.

    Returns:
        str: Formatted description optimized for parameter tooltips.
    """
    formatted_description = format_tooltip_text(description, width=60, use_html=True)
    print(formatted_description)
    return formatted_description


def wrap_text_simple(text: str, width: int = 60) -> str:
    """Simple text wrapping without HTML formatting.

    Args:
        text (str): Text to wrap.
        width (int): Maximum line width. Default: 60.

    Returns:
        str: Wrapped plain text.
    """
    return format_tooltip_text(text, width=width, use_html=False)
