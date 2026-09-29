import re

MAX_CHARS = 500


def decode_bytes(raw_bytes):
    """Try common encodings, since uploaded text files aren't always UTF-8."""
    for encoding in ("utf-8", "utf-16", "utf-8-sig"):
        try:
            return raw_bytes.decode(encoding)
        except (UnicodeDecodeError, UnicodeError):
            continue
    return raw_bytes.decode("latin-1")  # last resort, this one never fails

def chunk_text(text, max_chars=MAX_CHARS):
    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    chunks = []
    current = []
    for sentence in sentences:
        if current and len(" ".join(current + [sentence])) > max_chars:
            chunks.append(" ".join(current))
            current = [current[-1]]
        current.append(sentence)
    if current:
        chunks.append(" ".join(current))
    return chunks