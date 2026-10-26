"""Runtime grounding check: is the generated answer supported by the retrieved text?

Until now faithfulness was only measured OFFLINE (RAGAS on a handful of questions). This
is the cheap, classical (non-LLM, per constraint 1) check that runs on every English
policy answer: a model that "helpfully" adds a worked example with a number the
regulation never states -- a real failure mode for lending answers -- is caught here
instead of being served as if it were policy.

It is deliberately lexical, so it is conservative in one direction: it can reject a
correct answer that paraphrases heavily (a false escalation, which costs a human
hand-off), but it will not pass an invented figure. Not applied to Hindi/Kannada, where
a translated answer shares no words with the English source.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from config.settings import GROUNDING_MIN_WORD_OVERLAP

_NUMBER_PATTERN = re.compile(r"\d[\d,]*(?:\.\d+)?")
_WORD_PATTERN = re.compile(r"[a-z]{4,}")
_STOPWORDS = frozenset(
    """that this with from have been will your their there which would about under into than then
    them they when where what shall must also such these those only more most some other over
    each very just like make made does done while upon within without whether being because
    please sorry help human coach connect""".split()
)
_NUMBER_WORDS = {  # spelled-out numbers in a source count as the digit form
    "one": "1", "two": "2", "three": "3", "four": "4", "five": "5", "six": "6", "seven": "7",
    "eight": "8", "nine": "9", "ten": "10", "twelve": "12", "fifteen": "15", "twenty": "20",
    "thirty": "30", "sixty": "60", "ninety": "90", "hundred": "100",
}


@dataclass
class GroundingResult:
    grounded: bool
    word_overlap: float
    ungrounded_numbers: list[str] = field(default_factory=list)
    reason: str = ""


def _numbers(text: str) -> set[str]:
    found = {m.group(0).replace(",", "").rstrip(".") for m in _NUMBER_PATTERN.finditer(text)}
    lowered = text.lower()
    found |= {digit for word, digit in _NUMBER_WORDS.items() if re.search(rf"\b{word}\b", lowered)}
    return found


def _stem(word: str) -> str:
    return word[:5]  # crude on purpose: "consent"/"consents", "stored"/"storage" must match


def _content_words(text: str) -> set[str]:
    return {_stem(w) for w in _WORD_PATTERN.findall(text.lower()) if w not in _STOPWORDS}


def check_grounding(answer: str, context_chunks: list[str], query: str = "") -> GroundingResult:
    context = "\n".join(context_chunks)
    allowed_numbers = _numbers(context) | _numbers(query)
    ungrounded = sorted(n for n in _numbers(answer) if n not in allowed_numbers)
    if ungrounded:
        return GroundingResult(
            False, 0.0, ungrounded, f"answer states figure(s) not in the retrieved text: {', '.join(ungrounded)}"
        )

    answer_words = _content_words(answer)
    if not answer_words:
        return GroundingResult(True, 1.0)
    overlap = len(answer_words & _content_words(context + " " + query)) / len(answer_words)
    if overlap < GROUNDING_MIN_WORD_OVERLAP:
        return GroundingResult(
            False, overlap, [], f"only {overlap:.0%} of the answer's content words appear in the retrieved text"
        )
    return GroundingResult(True, overlap)
