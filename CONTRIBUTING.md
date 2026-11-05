# Contributing

Thanks for helping! Issues and pull requests are welcome in English, Uzbek or Russian.

## Setup

```bash
git clone https://github.com/jasurlutfullayev/aiogram-agent
cd aiogram-agent
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev,all]"
pytest
ruff check .
```

## Guidelines

- Keep the core free of heavy dependencies; providers and backends go behind extras.
- Every change needs a test. Use `ScriptedProvider` from `tests/conftest.py` — tests must not call real LLM APIs.
- One feature per pull request; describe *why* in the PR text.
- New memory backends implement `get`, `append`, `clear` (see `aiogram_agent/memory.py`).

Look for issues labelled **good first issue** to get started.
