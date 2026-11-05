from .base import LLMProvider, TextCallback

__all__ = ["LLMProvider", "TextCallback", "OpenAIProvider", "OpenAITranscriber", "AnthropicProvider"]


def __getattr__(name: str):  # lazy: don't import SDKs the user didn't install
    if name in ("OpenAIProvider", "OpenAITranscriber"):
        from . import openai

        return getattr(openai, name)
    if name == "AnthropicProvider":
        from .anthropic import AnthropicProvider

        return AnthropicProvider
    raise AttributeError(name)
