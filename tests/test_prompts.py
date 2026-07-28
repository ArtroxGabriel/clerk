from __future__ import annotations

from clerk.prompts import (
    CpuPromptStrategy,
    GpuPromptStrategy,
    PromptManager,
    clean_llm_output,
    clean_srt_for_prompt,
    get_language_name,
    is_meaningful_transcript,
)


def test_clean_llm_output() -> None:
    raw = """<<<ITEMS>>>
- Item 1
- Item 2
<<<END ITEMS>>>"""
    assert clean_llm_output(raw) == "- Item 1\n- Item 2"

    raw_consolidado = """<<<consolidado>>>
- Item A
- Item B
<<<END CONSOLIDADO>>>"""
    assert clean_llm_output(raw_consolidado) == "- Item A\n- Item B"


def test_get_language_name() -> None:
    assert get_language_name("pt") == "Portuguese"
    assert get_language_name("en") == "English"
    assert get_language_name(None) == "Portuguese"
    assert get_language_name("custom") == "custom"


def test_clean_srt_for_prompt() -> None:
    raw_srt = """1
00:00:00,000 --> 00:00:02,500
Hello world, welcome to the test.

2
00:00:03,000 --> 00:00:05,000
This is a secondary line.
"""
    cleaned = clean_srt_for_prompt(raw_srt)
    assert "00:00:00" not in cleaned
    assert "1" not in cleaned.splitlines()
    assert "Hello world, welcome to the test." in cleaned
    assert "This is a secondary line." in cleaned


def test_is_meaningful_transcript() -> None:
    assert is_meaningful_transcript("Reunião produtiva com decisões importantes sobre o projeto.") is True
    assert is_meaningful_transcript("") is False
    assert is_meaningful_transcript("   ") is False
    assert is_meaningful_transcript("[Music]") is False
    assert is_meaningful_transcript("noise noise noise noise") is False
    assert is_meaningful_transcript("3 words noise") is False


def test_prompt_manager_and_strategies() -> None:
    cpu_strat = PromptManager.get_strategy(is_gpu_model=False)
    gpu_strat = PromptManager.get_strategy(is_gpu_model=True)

    assert isinstance(cpu_strat, CpuPromptStrategy)
    assert isinstance(gpu_strat, GpuPromptStrategy)

    meeting_prompt_cpu = cpu_strat.build_summary_prompt("test transcript", "Portuguese", is_video=False)
    assert "## Pontos principais" in meeting_prompt_cpu
    assert "<<<TRANSCRIPT>>>" in meeting_prompt_cpu
    assert "<<<END TRANSCRIPT>>>" in meeting_prompt_cpu

    video_prompt_gpu = gpu_strat.build_summary_prompt("test transcript", "English", is_video=True)
    assert "## Resumo geral" in video_prompt_gpu
    assert "content strategist" in video_prompt_gpu
    assert "<<<TRANSCRIPT>>>" in video_prompt_gpu
    assert "<<<END TRANSCRIPT>>>" in video_prompt_gpu


def test_custom_prompt_strategy() -> None:
    from clerk.prompts import CustomPromptStrategy

    custom_summary = "Custom summary for {language}: {transcript}"
    custom_consolidation = "Consolidate category {category} in {language}: {items}"
    strat = CustomPromptStrategy(custom_summary, custom_consolidation)

    p_summary = strat.build_summary_prompt("my transcript", "Portuguese", is_video=False)
    assert p_summary == "Custom summary for Portuguese: my transcript"

    p_cons = strat.build_consolidation_prompt("Pontos principais", "- item 1", "Portuguese")
    assert p_cons == "Consolidate category Pontos principais in Portuguese: - item 1"

    no_placeholder_strat = CustomPromptStrategy("Simple summary prompt for {language}")
    p_auto = no_placeholder_strat.build_summary_prompt("my transcript", "English", is_video=False)
    assert "Simple summary prompt for English" in p_auto
    assert "<<<TRANSCRIPT>>>\nmy transcript\n<<<END TRANSCRIPT>>>" in p_auto

