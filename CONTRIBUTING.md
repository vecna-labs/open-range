# Contributing to OpenRange

Thanks for contributing.

## Contributor License Agreement

Before we can merge your first pull request, you need to sign the
[Contributor License Agreement](./CLA.md). The CLA bot will comment on your
PR with instructions — you sign by posting a single comment, once, and the
bot remembers you for every future PR.

If you are contributing on behalf of an employer, please confirm with them
that you have authorization to sign before doing so. For corporate-wide
agreements, contact the maintainers.

## What To Work On

We welcome contributions of all sizes. A good place to start is the issue board:
pick up an open issue, ask a question, or suggest a direction before you begin.
Discussion in issues and on Discord should be low-friction; if something is
unclear, please ask.

Feedback is also useful even when it does not come with code. Bug reports,
documentation notes, design questions, and examples all help the project move
forward. Another strong way to contribute is by creating or improving a pack;
the current overview is in [the pack section of the OpenRange docs](docs/start_here.md#pack).

## Local Setup

OpenRange uses [`uv`](https://github.com/astral-sh/uv) for local development.

```bash
uv sync --group dev
```

Useful smoke checks:

```bash
uv run openrange --help
```

Strands dependencies are optional:

```bash
uv sync --extra strands
```

## Checks

Before opening a pull request, run the checks relevant to your change.

```bash
uv run ruff check .
uv run mypy src tests examples main.py
uv run coverage run -m pytest tests
uv run coverage report
```

For local iteration, prefer targeted tests first. If your change touches
runtime, dashboard, training, or evaluation, include the exact non-routine
verification commands you ran in the PR description.

## Commit Messages

Use [Conventional Commits](https://www.conventionalcommits.org/) for commit
messages. Prefer concise prefixes such as `feat:`, `fix:`, `docs:`, `test:`,
`refactor:`, and `chore:`.

## Pull Requests

Good pull requests are usually:

- scoped to one clear change
- explicit about what changed and why
- backed by tests when behavior changes
- accompanied by doc updates when public behavior or workflows change

Use the repository PR template in
[.github/PULL_REQUEST_TEMPLATE.md](.github/PULL_REQUEST_TEMPLATE.md).

Keep the `Testing` section short and factual:

- list only manual or non-routine verification that reviewers would not
  otherwise see in CI
- if there was no special verification beyond CI-covered lint/unit checks, say
  that plainly
- do not paste long terminal transcripts into the PR body

Use `Review Notes` only for reviewer focus areas, tradeoffs, risks, or follow-up
work.

## Project Context

If you need more background before changing core behavior, start with:

- [`docs/start_here.md`](docs/start_here.md)
- [`docs/dashboard.md`](docs/dashboard.md)
- [`.rules`](.rules) for repo-specific rules
