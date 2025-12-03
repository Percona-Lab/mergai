# MergAI

AI-assisted merge conflict helper with a small CLI. It captures conflict context, sends it to an agent (Gemini CLI by default), stores the proposed resolution as a git note, and can drive commits or PRs from that note.

**This is under active development; do not use on production code.**

## Prerequisites

- Python 3.10+ and git installed.
- Gemini CLI available on your `PATH` (used as the default agent). Setup authentication on your own.
- Optional: `GITHUB_TOKEN` or `GH_TOKEN` for commands that read/write PRs.
- Recommended: `git config merge.conflictstyle diff3` for better conflict context.
- Recommended: `git config notes.displayRef refs/notes/mergai-marker` for seeing notes availability.

## Install

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

Verify with `mergai --help`.

## Typical Flow

- Trigger or pause at a merge conflict in your repo and capture conflict context: `mergai create-conflict-context`, or
- Capture context with PR comments: `mergai pr --repo <repo> add-comments-to-context`
- Ask the agent to propose a fix: `mergai resolve` (add `-y/--yolo` to relax safety checks).
- Inspect what was saved: `mergai status`.
- Commit the resolved files using the stored solution: `mergai commit`.
- Open a PR - `mergai pr --repo owner/name create --base main`

## Useful Commands

- `mergai log` - list commits that carry MergAI notes.
- `mergai show <commit>` - print stored context/solution (use flags like `--context`, `--solution`, `--raw`).
- `mergai create-conflict-context` - cature conflict context
- `mergai prompt` - show the prompt that would be sent to the agent.
- `mergai resolve` - resolve the conflict with the saved context as an input
- `mergai drop` - remove saved note pieces (e.g., `--solution`, `--context`, `--all`).
- `mergai commit` - commit solution and add a note
- `mergai pr --repo owner/name add-comments-to-context` - capture comments from PR with positive feedback as a context
- `mergai pr --repo owner/name create --base main` - open a PR using the recorded solution.

## Configuration

- Prompts: `.mergai/invariants.md` is appended to prompts when present.
- Add `.mergai_state/` to your project's `.gitignore`

## Tips

- Keep the working tree clean before running `mergai resolve`; it checks that only files the agent touched are staged.
- Fetching MergAI notes:

```bash
git fetch origin refs/notes/mergai
git fetch origin refs/notes/mergai-marker
```
