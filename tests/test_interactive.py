from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

from clerk.backends import ChatResponse, ToolCall, ToolNotSupportedError
from clerk.interactive import (
    CpuInteractiveStrategy,
    GpuInteractiveStrategy,
    get_interactive_strategy,
    merge_ata_sections,
    resolve_companion_srt,
    run_interactive_session,
)

SAMPLE_ATA = """## Contexto
* Reunião de alinhamento do projeto Selene.

## Pauta e discussões
* Discussão sobre migração do banco de dados.

## Decisões
* Adiar o deploy para a próxima sprint.

## Encaminhamentos
* Realizar migração do banco até sexta-feira.

## Pendências
* Nenhuma registrada."""

SAMPLE_SRT = """1
00:00:01,000 --> 00:00:05,000
O Gabriel explicou que o banco precisa migrar até sexta.

2
00:00:05,500 --> 00:00:09,000
A Mônica concordou em adiar o deploy para a sprint 14.
"""


class FakeInteractiveBackend:
    """Named fake LLMBackend for testing interactive strategies and sessions."""

    def __init__(self, responses: list[ChatResponse | Exception]) -> None:
        self.responses = list(responses)
        self.chat_calls: list[tuple[list[dict[str, Any]], list[dict[str, Any]] | None]] = []
        self.cleaned_up = False

    def generate(self, prompt: str) -> str:
        resp = self.chat([{"role": "user", "content": prompt}])
        return resp.content

    def chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
    ) -> ChatResponse:
        self.chat_calls.append((messages, tools))
        if not self.responses:
            return ChatResponse(content="Ok.")
        item = self.responses.pop(0)
        if isinstance(item, Exception):
            raise item
        return item

    def cleanup(self) -> None:
        self.cleaned_up = True


def test_merge_ata_sections_patches_only_modified_sections() -> None:
    llm_output = (
        "Atualizei os participantes e o responsável pelo banco:\n\n"
        "## Contexto\n"
        "- Reunião de alinhamento do projeto Selene com Gabriel (Tech Lead) e Mônica (PM).\n\n"
        "## Encaminhamentos\n"
        "- Gabriel: Realizar migração do banco até sexta-feira."
    )

    merged = merge_ata_sections(SAMPLE_ATA, llm_output, is_video=False)
    assert merged is not None
    assert "Gabriel (Tech Lead) e Mônica (PM)" in merged
    assert "Gabriel: Realizar migração do banco até sexta-feira." in merged
    # Untouched sections must remain intact
    assert "* Discussão sobre migração do banco de dados." in merged
    assert "* Adiar o deploy para a próxima sprint." in merged
    assert "## Pendências\n* Nenhuma registrada." in merged


def test_merge_ata_sections_returns_none_for_plain_qa_answer() -> None:
    llm_output = "De acordo com o SRT em [00:00:01 -> 00:00:05], o Gabriel ficou responsável pelo banco."
    assert merge_ata_sections(SAMPLE_ATA, llm_output, is_video=False) is None


def test_get_interactive_strategy_selects_cpu_or_gpu() -> None:
    assert isinstance(get_interactive_strategy("LiquidAI/lfm2.5-1.2b-instruct", is_gpu=False), CpuInteractiveStrategy)
    assert isinstance(get_interactive_strategy("gemma4:e2b", is_gpu=False), GpuInteractiveStrategy)
    assert isinstance(get_interactive_strategy("custom-model", is_gpu=True), GpuInteractiveStrategy)


def test_cpu_interactive_strategy_pre_searches_srt_and_updates_ata(tmp_path: Path) -> None:
    srt_file = tmp_path / "meeting_transcript.srt"
    srt_file.write_text(SAMPLE_SRT, encoding="utf-8")

    fake_backend = FakeInteractiveBackend(
        [
            ChatResponse(
                content="## Encaminhamentos\n- Gabriel: Realizar migração do banco até sexta-feira."
            )
        ]
    )
    strategy = CpuInteractiveStrategy()

    turn = strategy.process_turn(
        user_message="Quem falou do banco foi o Gabriel, atualize os encaminhamentos",
        current_ata=SAMPLE_ATA,
        history=[],
        srt_path=srt_file,
        backend=fake_backend,
        language="pt",
        is_video=False,
    )

    assert turn.updated_ata is not None
    assert "Gabriel: Realizar migração do banco" in turn.updated_ata
    assert len(turn.srt_matches) >= 1
    sent_messages, sent_tools = fake_backend.chat_calls[0]
    assert sent_tools is None
    assert "<<<SRT_EXCERPTS>>>" in sent_messages[-1]["content"]
    assert "00:00:01" in sent_messages[-1]["content"]


def test_gpu_interactive_strategy_executes_tool_call_and_returns_answer(tmp_path: Path) -> None:
    srt_file = tmp_path / "meeting_transcript.srt"
    srt_file.write_text(SAMPLE_SRT, encoding="utf-8")

    fake_backend = FakeInteractiveBackend(
        [
            ChatResponse(
                content="",
                tool_calls=(ToolCall(name="search_srt", arguments={"query": "Mônica sprint"}),),
            ),
            ChatResponse(
                content="A Mônica concordou em adiar o deploy para a sprint 14 ([00:00:05 -> 00:00:09])."
            ),
        ]
    )
    strategy = GpuInteractiveStrategy()

    turn = strategy.process_turn(
        user_message="O que a Mônica disse sobre a sprint?",
        current_ata=SAMPLE_ATA,
        history=[],
        srt_path=srt_file,
        backend=fake_backend,
        language="pt",
        is_video=False,
    )

    assert turn.updated_ata is None
    assert "sprint 14" in turn.reply_text
    assert len(turn.srt_matches) >= 1
    assert len(fake_backend.chat_calls) == 2
    assert fake_backend.chat_calls[0][1] is not None  # tools passed on first call
    second_call_messages = fake_backend.chat_calls[1][0]
    assert any(m.get("role") == "assistant" and m.get("tool_calls") for m in second_call_messages)
    assert any(m.get("role") == "tool" and m.get("name") == "search_srt" for m in second_call_messages)


def test_gpu_interactive_strategy_falls_back_when_tools_unsupported(tmp_path: Path) -> None:
    srt_file = tmp_path / "meeting_transcript.srt"
    srt_file.write_text(SAMPLE_SRT, encoding="utf-8")

    fake_backend = FakeInteractiveBackend(
        [
            ToolNotSupportedError("Model does not support tools"),
            ChatResponse(content="## Decisões\n- Adiar o deploy para a sprint 14 (validado por Mônica)."),
        ]
    )
    strategy = GpuInteractiveStrategy()

    turn = strategy.process_turn(
        user_message="Atualize a decisão com o número da sprint que a Mônica falou",
        current_ata=SAMPLE_ATA,
        history=[],
        srt_path=srt_file,
        backend=fake_backend,
        language="pt",
        is_video=False,
    )

    assert turn.updated_ata is not None
    assert "sprint 14" in turn.updated_ata
    assert len(fake_backend.chat_calls) == 2
    assert fake_backend.chat_calls[1][1] is None  # fallback call has no tools


def test_resolve_companion_srt_discovers_metadata_and_naming_conventions(tmp_path: Path) -> None:
    summary_path = tmp_path / "reuniao_ata.md"
    summary_path.write_text(SAMPLE_ATA, encoding="utf-8")

    # 1. None when nothing exists
    assert resolve_companion_srt(summary_path) is None

    # 2. <stem>.srt
    alt_srt = tmp_path / "reuniao.srt"
    alt_srt.write_text(SAMPLE_SRT, encoding="utf-8")
    assert resolve_companion_srt(summary_path) == alt_srt

    # 3. <stem>_transcript.srt takes precedence over <stem>.srt
    std_srt = tmp_path / "reuniao_transcript.srt"
    std_srt.write_text(SAMPLE_SRT, encoding="utf-8")
    assert resolve_companion_srt(summary_path) == std_srt

    # 4. Metadata JSON takes precedence
    meta_srt = tmp_path / "custom_from_meta.srt"
    meta_srt.write_text(SAMPLE_SRT, encoding="utf-8")
    meta_path = tmp_path / "reuniao_metadata.json"
    meta_path.write_text(
        json.dumps({"output_files": {"transcript_path": str(meta_srt)}}),
        encoding="utf-8",
    )
    assert resolve_companion_srt(summary_path) == meta_srt

    # 5. Explicit --srt overrides all
    explicit_srt = tmp_path / "explicit.srt"
    explicit_srt.write_text(SAMPLE_SRT, encoding="utf-8")
    assert resolve_companion_srt(summary_path, explicit_srt=explicit_srt) == explicit_srt


@patch("clerk.interactive.prompt_edit_confirmation")
@patch("clerk.interactive.questionary.text")
def test_run_interactive_session_commands_accept_undo_and_cleanup(
    mock_text: MagicMock,
    mock_confirm: MagicMock,
    tmp_path: Path,
) -> None:
    summary_path = tmp_path / "meeting_ata.md"
    summary_path.write_text(SAMPLE_ATA, encoding="utf-8")
    srt_path = tmp_path / "meeting_transcript.srt"
    srt_path.write_text(SAMPLE_SRT, encoding="utf-8")

    fake_backend = FakeInteractiveBackend(
        [
            ChatResponse(
                content="## Encaminhamentos\n- Gabriel: Realizar migração do banco até sexta-feira."
            )
        ]
    )

    # Sequence of user inputs:
    # 1. /search Gabriel
    # 2. /show
    # 3. instruction to update Encaminhamentos -> confirmed 'accept'
    # 4. /diff
    # 5. /undo -> reverts file on disk
    # 6. /help
    # 7. /exit
    mock_text.return_value.ask.side_effect = [
        "/search Gabriel",
        "/show",
        "Coloque o Gabriel como responsável no encaminhamento do banco",
        "/diff",
        "/undo",
        "/help",
        "/exit",
    ]
    mock_confirm.return_value = "accept"

    run_interactive_session(
        summary_path=summary_path,
        srt_path=srt_path,
        llm_model="LiquidAI/lfm2.5-1.2b-instruct",
        language="pt",
        is_video=False,
        backend=fake_backend,
    )

    assert fake_backend.cleaned_up is True
    # Because /undo was called after accepting the edit, disk content should be back to SAMPLE_ATA
    assert summary_path.read_text(encoding="utf-8").strip() == SAMPLE_ATA.strip()


@patch("clerk.interactive.prompt_edit_confirmation")
@patch("clerk.interactive.questionary.text")
def test_run_interactive_session_adjust_flow_persists_refined_edit(
    mock_text: MagicMock,
    mock_confirm: MagicMock,
    tmp_path: Path,
) -> None:
    summary_path = tmp_path / "meeting_ata.md"
    summary_path.write_text(SAMPLE_ATA, encoding="utf-8")

    fake_backend = FakeInteractiveBackend(
        [
            ChatResponse(content="## Encaminhamentos\n- Gabriel: Migrar banco."),
            ChatResponse(content="## Encaminhamentos\n- Gabriel (Tech Lead): Migrar banco até sexta."),
        ]
    )

    mock_text.return_value.ask.side_effect = [
        "Atribua ao Gabriel",
        "Inclua o cargo Tech Lead e o prazo sexta",
        "/exit",
    ]
    mock_confirm.side_effect = ["adjust", "accept"]

    run_interactive_session(
        summary_path=summary_path,
        srt_path=None,
        llm_model="LiquidAI/lfm2.5-1.2b-instruct",
        language="pt",
        is_video=False,
        backend=fake_backend,
    )

    saved = summary_path.read_text(encoding="utf-8")
    assert "Gabriel (Tech Lead): Migrar banco até sexta." in saved
    assert fake_backend.cleaned_up is True


@patch("clerk.interactive.prompt_edit_confirmation")
@patch("clerk.interactive.questionary.text")
def test_run_interactive_session_adjust_preserves_draft_when_llm_replies_without_headers(
    mock_text: MagicMock,
    mock_confirm: MagicMock,
    tmp_path: Path,
) -> None:
    summary_path = tmp_path / "meeting_ata.md"
    summary_path.write_text(SAMPLE_ATA, encoding="utf-8")

    fake_backend = FakeInteractiveBackend(
        [
            ChatResponse(content="## Encaminhamentos\n- Gabriel: Migrar banco."),
            ChatResponse(content="Não encontrei menção a horário exato no SRT."),
        ]
    )

    mock_text.return_value.ask.side_effect = [
        "Atribua ao Gabriel",
        "Qual foi o horário exato?",
        "/exit",
    ]
    # First confirm: 'adjust' -> LLM replies conversationally -> loop continues -> second confirm: 'accept'
    mock_confirm.side_effect = ["adjust", "accept"]

    run_interactive_session(
        summary_path=summary_path,
        srt_path=None,
        llm_model="LiquidAI/lfm2.5-1.2b-instruct",
        language="pt",
        is_video=False,
        backend=fake_backend,
    )

    saved = summary_path.read_text(encoding="utf-8")
    assert "Gabriel: Migrar banco." in saved
