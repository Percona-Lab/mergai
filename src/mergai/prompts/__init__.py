from importlib.resources import files

from jinja2 import Environment, select_autoescape

# Plain-text Jinja environment for rendering prompt files with project-specific
# terms (e.g. ``fork_term``). keep_trailing_newline preserves the trailing
# newline the prompts rely on. Autoescape stays OFF: prompts are plain
# text/markdown fed to an AI agent, not HTML served to a browser, and the
# prompt bodies contain `<`, `>`, `&`, and quotes (e.g. `Foo::bar(int, Context&)`,
# `<commit>`, JSON) that HTML-escaping would corrupt. select_autoescape with
# default_for_string=False keeps escaping disabled for these from_string
# templates while avoiding a bare `autoescape=False` constant.
_PROMPT_ENV = Environment(
    keep_trailing_newline=True,
    autoescape=select_autoescape(default_for_string=False, default=False),
)


def load_prompt(prompt_name: str) -> str:
    return (files("mergai.prompts") / prompt_name).read_text(encoding="utf-8")


def _render_prompt(prompt_name: str, context: dict | None) -> str:
    """Load a prompt; when ``context`` is given, render it as a Jinja template.

    Used for prompts that interpolate project-specific terms (``fork_term`` /
    ``upstream_term``); plain prompts pass ``context=None`` and are returned
    verbatim.
    """
    text = load_prompt(prompt_name)
    if not context:
        return text
    return _PROMPT_ENV.from_string(text).render(**context)


def load_system_prompt_resolve() -> str:
    return load_prompt("system_prompt_resolve.md")


def load_system_prompt_describe() -> str:
    return load_prompt("system_prompt_describe.md")


def load_system_prompt_describe_verify() -> str:
    return load_prompt("system_prompt_describe_verify.md")


def load_system_prompt_ci_fix(context: dict | None = None) -> str:
    return _render_prompt("system_prompt_ci_fix.md", context)


def load_pr_comments_prompt() -> str:
    return load_prompt("pr_comments.md")


def load_conflict_context_prompt() -> str:
    return load_prompt("conflict_context.md")


def load_ci_fix_context_prompt() -> str:
    return load_prompt("ci_fix_context.md")


def load_merge_context_for_ci_fix_prompt(context: dict | None = None) -> str:
    return _render_prompt("merge_context_for_ci_fix.md", context)


def load_user_comment_prompt() -> str:
    return load_prompt("user_comment.md")
