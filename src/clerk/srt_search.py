from __future__ import annotations

import logging
import re
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)

TIMESTAMP_LINE_RE = re.compile(
    r"^(\d{2}:\d{2}:\d{2})[,\.]\d{3}\s*-->\s*(\d{2}:\d{2}:\d{2})[,\.]\d{3}"
)
LINE_PREFIX_RE = re.compile(r"^(\d+):")

STOPWORDS: frozenset[str] = frozenset(
    {
        "a",
        "ao",
        "aos",
        "aquela",
        "aquele",
        "as",
        "ata",
        "ate",
        "até",
        "com",
        "como",
        "da",
        "das",
        "de",
        "dela",
        "dele",
        "do",
        "dos",
        "e",
        "ela",
        "ele",
        "em",
        "entao",
        "então",
        "era",
        "essa",
        "esse",
        "esta",
        "está",
        "este",
        "eu",
        "foi",
        "for",
        "in",
        "is",
        "isso",
        "isto",
        "ja",
        "já",
        "la",
        "lá",
        "mais",
        "mas",
        "me",
        "mesmo",
        "meu",
        "minha",
        "muito",
        "na",
        "nas",
        "nao",
        "não",
        "nem",
        "no",
        "nos",
        "nós",
        "o",
        "of",
        "on",
        "os",
        "ou",
        "para",
        "pela",
        "pelas",
        "pelo",
        "pelos",
        "por",
        "qual",
        "quando",
        "que",
        "quem",
        "resumo",
        "se",
        "sem",
        "ser",
        "seu",
        "sobre",
        "sua",
        "tambem",
        "também",
        "tem",
        "the",
        "to",
        "um",
        "uma",
        "umas",
        "uns",
        "voce",
        "você",
        "adicione",
        "atualize",
        "coloque",
        "edite",
        "melhore",
        "mude",
        "procure",
        "busque",
        "falou",
        "disse",
    }
)


@dataclass(frozen=True)
class SrtBlock:
    """Represents a single parsed subtitle block from an SRT file.

    Example:
        >>> block = SrtBlock(1, "00:00:01", "00:00:04", 1, 4, "Olá mundo")
        >>> block.start_time
        '00:00:01'
    """

    index: int
    start_time: str
    end_time: str
    start_line: int
    end_line: int
    text: str


@dataclass(frozen=True)
class SrtMatchWindow:
    """Represents a contiguous window of SRT subtitle blocks around a search match.

    Example:
        >>> win = SrtMatchWindow("00:01:00", "00:01:10", "Texto falado", (2, 3))
        >>> win.format_citation()
        '[00:01:00 -> 00:01:10] Texto falado'
    """

    start_time: str
    end_time: str
    text: str
    matched_indices: tuple[int, ...]

    def format_citation(self) -> str:
        """Formats the window with timestamp brackets for LLM or TUI display."""
        return f"[{self.start_time} -> {self.end_time}] {self.text}"


def _flush_block(
    idx: int | None,
    start_ts: str | None,
    end_ts: str | None,
    start_line: int,
    end_line: int,
    text_lines: list[str],
) -> SrtBlock | None:
    if start_ts is None or end_ts is None or not text_lines:
        return None
    joined = " ".join(text_lines).strip()
    if not joined:
        return None
    return SrtBlock(
        index=idx if idx is not None else 0,
        start_time=start_ts,
        end_time=end_ts,
        start_line=start_line,
        end_line=end_line,
        text=joined,
    )


def parse_srt_blocks(srt_content: str) -> list[SrtBlock]:
    """Parses raw SRT subtitle content into structured SrtBlock instances.

    Example:
        >>> blocks = parse_srt_blocks("1\\n00:00:01,000 --> 00:00:02,000\\nOi\\n")
        >>> len(blocks)
        1
    """
    if not srt_content or not srt_content.strip():
        return []

    blocks: list[SrtBlock] = []
    cur_idx: int | None = None
    cur_start: str | None = None
    cur_end: str | None = None
    block_start_line = 1
    last_line_num = 1
    text_lines: list[str] = []

    for line_num, raw_line in enumerate(srt_content.splitlines(), start=1):
        stripped = raw_line.strip()
        if not stripped:
            flushed = _flush_block(cur_idx, cur_start, cur_end, block_start_line, last_line_num, text_lines)
            if flushed is not None:
                blocks.append(flushed)
            cur_idx, cur_start, cur_end, text_lines = None, None, None, []
            continue

        if cur_start is None and stripped.isdigit():
            cur_idx = int(stripped)
            block_start_line = line_num
            last_line_num = line_num
            continue

        ts_match = TIMESTAMP_LINE_RE.match(stripped)
        if ts_match:
            if cur_idx is None:
                block_start_line = line_num
            cur_start, cur_end = ts_match.group(1), ts_match.group(2)
            last_line_num = line_num
            continue

        if cur_start is not None:
            text_lines.append(stripped)
            last_line_num = line_num

    flushed = _flush_block(cur_idx, cur_start, cur_end, block_start_line, last_line_num, text_lines)
    if flushed is not None:
        blocks.append(flushed)
    return blocks


def _parse_line_numbers(stdout: str) -> set[int]:
    matched: set[int] = set()
    for line in stdout.splitlines():
        m = LINE_PREFIX_RE.match(line.strip())
        if m:
            matched.add(int(m.group(1)))
    return matched


def _find_lines_python(srt_path: Path, pattern: str) -> set[int]:
    try:
        compiled = re.compile(pattern, re.IGNORECASE)
    except re.error:
        compiled = re.compile(re.escape(pattern), re.IGNORECASE)

    content = srt_path.read_text(encoding="utf-8", errors="replace")
    return {
        idx
        for idx, line in enumerate(content.splitlines(), start=1)
        if compiled.search(line)
    }


def find_matching_lines(srt_path: Path, pattern: str) -> set[int]:
    """Finds 1-based matching line numbers in an SRT file using rg, grep, or Python re.

    Example:
        >>> find_matching_lines(Path("nonexistent.srt"), "termo")
        set()
    """
    if not pattern.strip() or not srt_path.exists():
        return set()

    rg_bin = shutil.which("rg")
    if rg_bin:
        cmd = [rg_bin, "-n", "-i", "--no-heading", "--color=never", "-e", pattern, str(srt_path)]
        res = subprocess.run(cmd, capture_output=True, text=True, check=False)
        if res.returncode in (0, 1):
            return _parse_line_numbers(res.stdout)

    grep_bin = shutil.which("grep")
    if grep_bin:
        cmd = [grep_bin, "-n", "-i", "-E", "-e", pattern, str(srt_path)]
        res = subprocess.run(cmd, capture_output=True, text=True, check=False)
        if res.returncode in (0, 1):
            return _parse_line_numbers(res.stdout)

    return _find_lines_python(srt_path, pattern)


def _build_safe_regex_from_query(query: str) -> tuple[str, list[str]]:
    raw_tokens = [tok.strip() for tok in re.split(r"\s+", query.strip()) if tok.strip()]
    word_tokens = re.findall(r"[a-zA-ZÀ-ÿ0-9_$+-]{2,}", query)
    candidates = word_tokens if word_tokens else raw_tokens
    unique_terms = list(dict.fromkeys(candidates))
    if not unique_terms:
        return "", []
    escaped = [re.escape(term) for term in unique_terms]
    return "|".join(escaped), [t.lower() for t in unique_terms]


def _merge_windows(
    blocks: list[SrtBlock],
    hit_block_positions: list[int],
    terms: list[str],
    window: int,
    max_results: int,
) -> list[SrtMatchWindow]:
    if not hit_block_positions:
        return []

    total = len(blocks)
    intervals: list[tuple[int, int]] = []
    for pos in sorted(set(hit_block_positions)):
        lo = max(0, pos - window)
        hi = min(total - 1, pos + window)
        if intervals and lo <= intervals[-1][1] + 1:
            prev_lo, prev_hi = intervals[-1]
            intervals[-1] = (prev_lo, max(prev_hi, hi))
        else:
            intervals.append((lo, hi))

    scored: list[tuple[int, int, SrtMatchWindow]] = []
    for order, (lo, hi) in enumerate(intervals):
        slice_blocks = blocks[lo : hi + 1]
        combined_text = " ".join(b.text for b in slice_blocks)
        lower_text = combined_text.lower()
        score = sum(lower_text.count(term) for term in terms)
        win = SrtMatchWindow(
            start_time=slice_blocks[0].start_time,
            end_time=slice_blocks[-1].end_time,
            text=combined_text,
            matched_indices=tuple(b.index for b in slice_blocks),
        )
        scored.append((score, -order, win))

    scored.sort(key=lambda item: (item[0], item[1]), reverse=True)
    return [item[2] for item in scored[:max_results]]


def search_srt_file(
    srt_path: Path,
    query: str,
    *,
    window: int = 1,
    max_results: int = 6,
) -> list[SrtMatchWindow]:
    """Searches an SRT file via rg/grep/Python and returns timestamped context windows.

    Example:
        >>> search_srt_file(Path("missing.srt"), "budget")
        []
    """
    if not query or not query.strip() or not srt_path.exists():
        return []

    pattern, terms = _build_safe_regex_from_query(query)
    if not pattern:
        return []

    blocks = parse_srt_blocks(srt_path.read_text(encoding="utf-8", errors="replace"))
    if not blocks:
        return []

    matched_lines = find_matching_lines(srt_path, pattern)
    if not matched_lines:
        return []

    hit_positions = [
        pos
        for pos, block in enumerate(blocks)
        if any(block.start_line <= ln <= block.end_line for ln in matched_lines)
    ]
    return _merge_windows(blocks, hit_positions, terms, window, max_results)


def extract_search_keywords(user_message: str, max_keywords: int = 8) -> str:
    """Extracts meaningful search keywords and entities from a user chat prompt.

    Example:
        >>> extract_search_keywords("Adicione que o Gabriel falou sobre migração")
        'Gabriel migração'
    """
    words = re.findall(r"[a-zA-ZÀ-ÿ0-9_-]{3,}", user_message)
    prioritized: list[str] = []
    regular: list[str] = []

    for word in words:
        lower = word.lower()
        if lower in STOPWORDS:
            continue
        if word[0].isupper() or any(ch.isdigit() for ch in word):
            if word not in prioritized:
                prioritized.append(word)
        elif lower not in [w.lower() for w in regular] and lower not in [p.lower() for p in prioritized]:
            regular.append(word)

    selected = (prioritized + regular)[:max_keywords]
    return " ".join(selected)
