from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

from clerk.srt_search import (
    SrtBlock,
    SrtMatchWindow,
    extract_search_keywords,
    find_matching_lines,
    parse_srt_blocks,
    search_srt_file,
)

SAMPLE_SRT = """1
00:00:01,200 --> 00:00:04,500
Bom dia pessoal, hoje vamos revisar
o orçamento do projeto Selene.

2
00:00:04,800 --> 00:00:08,100
O Gabriel ficou responsável pela migração
do banco de dados até sexta-feira.

3
00:00:08,500 --> 00:00:12,000
A professora Mônica validou o cronograma
com a equipe de QA.

4
00:00:12,500 --> 00:00:16,000
Então fechamos a decisão de adiar o deploy
para a próxima sprint.
"""


def test_parse_srt_blocks_extracts_timestamps_lines_and_text() -> None:
    blocks = parse_srt_blocks(SAMPLE_SRT)
    assert len(blocks) == 4

    assert blocks[0] == SrtBlock(
        index=1,
        start_time="00:00:01",
        end_time="00:00:04",
        start_line=1,
        end_line=4,
        text="Bom dia pessoal, hoje vamos revisar o orçamento do projeto Selene.",
    )
    assert blocks[1].index == 2
    assert blocks[1].start_time == "00:00:04"
    assert blocks[1].end_time == "00:00:08"
    assert "Gabriel ficou responsável" in blocks[1].text


def test_parse_srt_blocks_handles_empty_and_malformed_input() -> None:
    assert parse_srt_blocks("") == []
    assert parse_srt_blocks("   \n\n  ") == []
    assert parse_srt_blocks("random text without timestamps") == []


def test_find_matching_lines_with_rg_or_default(tmp_path: Path) -> None:
    srt_file = tmp_path / "meeting.srt"
    srt_file.write_text(SAMPLE_SRT, encoding="utf-8")

    lines = find_matching_lines(srt_file, "Gabriel|orçamento")
    # Line 4 has "orçamento", Line 8 has "Gabriel"
    assert 4 in lines
    assert 8 in lines


def test_find_matching_lines_falls_back_to_grep_when_rg_missing(tmp_path: Path) -> None:
    srt_file = tmp_path / "meeting.srt"
    srt_file.write_text(SAMPLE_SRT, encoding="utf-8")

    def fake_which(cmd: str) -> str | None:
        if cmd == "rg":
            return None
        return "/usr/bin/grep"

    with patch("clerk.srt_search.shutil.which", side_effect=fake_which):
        lines = find_matching_lines(srt_file, "Mônica")
    assert 13 in lines


def test_find_matching_lines_falls_back_to_python_when_binaries_missing(tmp_path: Path) -> None:
    srt_file = tmp_path / "meeting.srt"
    srt_file.write_text(SAMPLE_SRT, encoding="utf-8")

    with patch("clerk.srt_search.shutil.which", return_value=None):
        lines = find_matching_lines(srt_file, "deploy")
    assert 18 in lines


def test_search_srt_file_handles_regex_special_chars_safely(tmp_path: Path) -> None:
    srt_file = tmp_path / "special.srt"
    srt_file.write_text(
        "1\n00:00:01,000 --> 00:00:03,000\n[music] (test) C++ budget $500?\n",
        encoding="utf-8",
    )

    matches = search_srt_file(srt_file, "[music] (test) $500?")
    assert len(matches) == 1
    assert "$500" in matches[0].text


def test_search_srt_file_returns_windowed_citations(tmp_path: Path) -> None:
    srt_file = tmp_path / "meeting.srt"
    srt_file.write_text(SAMPLE_SRT, encoding="utf-8")

    matches = search_srt_file(srt_file, "Gabriel banco", window=1, max_results=3)
    assert len(matches) >= 1
    top = matches[0]
    assert isinstance(top, SrtMatchWindow)
    # Block 2 matched, so window=1 includes blocks 1, 2, 3
    assert top.start_time == "00:00:01"
    assert top.end_time == "00:00:12"
    assert "Gabriel" in top.text
    assert "orçamento" in top.text
    assert "Mônica" in top.text
    citation = top.format_citation()
    assert citation.startswith("[00:00:01 -> 00:00:12]")


def test_search_srt_file_returns_empty_for_missing_file_or_empty_query(tmp_path: Path) -> None:
    srt_file = tmp_path / "meeting.srt"
    srt_file.write_text(SAMPLE_SRT, encoding="utf-8")

    assert search_srt_file(srt_file, "   ") == []
    assert search_srt_file(tmp_path / "nonexistent.srt", "Gabriel") == []


def test_extract_search_keywords_filters_stopwords_and_keeps_entities() -> None:
    msg = "Adicione na ata que o Gabriel ficou responsável pela migração do banco de dados"
    keywords = extract_search_keywords(msg)
    assert "Gabriel" in keywords
    assert "migração" in keywords
    assert "banco" in keywords
    assert "que" not in keywords.split()
