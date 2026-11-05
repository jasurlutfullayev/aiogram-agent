from aiogram_agent.router import split_text


def test_split_short():
    assert split_text("hi") == ["hi"]
    assert split_text("  ") == []


def test_split_on_paragraphs():
    text = ("a" * 30 + "\n\n") * 5
    chunks = split_text(text, limit=70)
    assert all(len(c) <= 70 for c in chunks)
    assert "".join(chunks).replace("\n", "") == "a" * 150


def test_split_hard_cut():
    chunks = split_text("x" * 250, limit=100)
    assert [len(c) for c in chunks] == [100, 100, 50]


def test_router_builds():
    from conftest import ScriptedProvider

    from aiogram_agent import Agent, HandoffManager, create_router

    router = create_router(Agent(ScriptedProvider(), handoff=HandoffManager(-1)))
    assert router.name == "aiogram_agent"
