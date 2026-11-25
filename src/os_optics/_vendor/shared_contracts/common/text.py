"""Text an author writes into a Project or a workspace: names, labels, descriptions and the basis of a stated value.

``AuthorName`` (1 to 200 characters) and ``AuthorText`` (1 to 1000 characters) take any Unicode text: a name in Chinese, a
degree or micro sign, an accented letter. On read the text is normalized to Unicode NFC, so two spellings of one text (a
precomposed and a decomposed accent) are one stored value and one hash. It needs at least one visible character; control
characters (``Cc``) and unpaired surrogates are refused. The published catalog keeps its printable-ASCII text
(``design_catalog.optical_catalog.PublicName`` / ``PublicText``).
"""

from __future__ import annotations

import unicodedata
from typing import Annotated

from pydantic import AfterValidator, BeforeValidator, Field, StringConstraints


def nfc(value: object) -> object:
    """The NFC form of a text (anything else, and a text with an unpaired surrogate, is passed on unchanged for the
    string type or ``check_author_text`` to refuse)."""

    if isinstance(value, str) and not any(0xD800 <= ord(c) <= 0xDFFF for c in value):
        return unicodedata.normalize("NFC", value)
    return value


def check_author_text(value: str) -> str:
    """``value`` when it is author text: no control character, no unpaired surrogate, at least one visible character."""

    for character in value:
        category = unicodedata.category(character)
        if category == "Cc":
            raise ValueError(f"a control character (U+{ord(character):04X}) is not text: write the text on one line without it")
        if category == "Cs":
            raise ValueError(f"an unpaired surrogate (U+{ord(character):04X}) is not a character: write the text as valid Unicode")
    if not any(unicodedata.category(c)[0] not in "ZC" for c in value):
        raise ValueError("the text needs at least one visible character")
    return value


AuthorName = Annotated[
    str,
    BeforeValidator(nfc),
    StringConstraints(strict=True, min_length=1, max_length=200),
    AfterValidator(check_author_text),
    Field(description="A name as the author writes it: any Unicode text (stored in NFC), 1 to 200 characters, at least one "
                      "visible, no control character."),
]

AuthorText = Annotated[
    str,
    BeforeValidator(nfc),
    StringConstraints(strict=True, min_length=1, max_length=1000),
    AfterValidator(check_author_text),
    Field(description="Text as the author writes it: any Unicode text (stored in NFC), 1 to 1000 characters, at least one "
                      "visible, no control character."),
]
