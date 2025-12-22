from importlib.resources import files


def load_prompt(prompt_name: str) -> str:
    return (files("mergai.prompts") / prompt_name).read_text(encoding="utf-8")


def load_system_prompt() -> str:
    return load_prompt("system_prompt.md")


def load_pr_comments_prompt() -> str:
    return load_prompt("pr_comments.md")


def load_conflict_context_prompt() -> str:
    return load_prompt("conflict_context.md")
