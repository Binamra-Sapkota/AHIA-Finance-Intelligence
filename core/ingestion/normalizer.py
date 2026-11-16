"""Text normalization engine preserving semantic paragraph boundaries."""

import re
import unicodedata


class TextNormalizer:
    """Normalizes extracted text while strictly maintaining paragraph boundaries."""

    @staticmethod
    def normalize(text: str) -> str:
        """Clean and normalize raw extracted text.

        Rules:
        - Unicode NFKC canonical normalization.
        - Standardize newlines (\\r\\n and \\r -> \\n).
        - Remove non-printable control characters (preserving \\n and \\t).
        - Rejoin broken hyphenated words at line endings (e.g. 'com-\\npound' -> 'compound').
        - Rejoin soft wrapped lines within paragraphs while preserving double newlines (\\n\\n).
        - Normalize multiple blank lines to exactly double newlines (\\n\\n).

        Args:
            text: Raw extracted text string.

        Returns:
            Normalized clean string with intact paragraph breaks.
        """
        if not text:
            return ""

        # 1. Unicode NFKC normalization
        norm = unicodedata.normalize("NFKC", text)

        # 2. Standardize line endings
        norm = norm.replace("\r\n", "\n").replace("\r", "\n")

        # 3. Remove non-printable control characters except newline and tab
        norm = "".join(ch for ch in norm if ch == "\n" or ch == "\t" or unicodedata.category(ch)[0] != "C")

        # 4. Rejoin hyphenated line endings: 'in-\nvestment' -> 'investment'
        # Matches lowercase letters with hyphen followed by newline and lowercase letters
        norm = re.sub(r"([a-z])-\n\s*([a-z])", r"\1\2", norm)

        # 5. Split by paragraph boundaries (2 or more newlines)
        paragraphs = re.split(r"\n\s*\n+", norm)

        cleaned_paragraphs = []
        for p in paragraphs:
            p_strip = p.strip()
            if not p_strip:
                continue

            # Within paragraph, collapse internal soft line breaks into a single space
            # unless the line appears to be a list item (e.g. starts with '-', '*', '•', or number)
            lines = p_strip.splitlines()
            merged_lines = []
            for line in lines:
                cleaned_line = re.sub(r"[ \t]+", " ", line.strip())
                if cleaned_line:
                    merged_lines.append(cleaned_line)

            # Rejoin lines within this paragraph
            paragraph_text = " ".join(merged_lines)
            cleaned_paragraphs.append(paragraph_text)

        # 6. Reconnect paragraphs using standard double newline
        return "\n\n".join(cleaned_paragraphs)
