"""Template loading and rendering utilities for mergai.

This module provides functions to load and render Jinja2 templates
from the mergai.templates package.
"""

import json
from typing import Any, Callable, Dict, Optional
from jinja2 import Environment, PackageLoader, TemplateNotFound

from .output import OutputFormat


def _create_jinja_env() -> Environment:
    """Create and configure the Jinja2 environment.

    Returns:
        Configured Jinja2 Environment with custom filters.
    """
    env = Environment(
        loader=PackageLoader("mergai", "templates"),
        # Don't use trim_blocks/lstrip_blocks - we handle whitespace manually
        # in templates to ensure proper table formatting
        trim_blocks=False,
        lstrip_blocks=False,
        keep_trailing_newline=True,
    )

    # Add custom filters
    env.filters["short_sha"] = lambda sha: sha[:11] if sha else ""
    env.filters["to_json"] = lambda obj, **kwargs: json.dumps(
        obj, default=str, **kwargs
    )

    return env


# Global Jinja2 environment (lazy initialization)
_jinja_env: Optional[Environment] = None


def get_jinja_env() -> Environment:
    """Get the global Jinja2 environment (creates it if needed).

    Returns:
        The configured Jinja2 Environment.
    """
    global _jinja_env
    if _jinja_env is None:
        _jinja_env = _create_jinja_env()
    return _jinja_env


def load_template(format: str, name: str):
    """Load a template file for the given format and name.

    Args:
        format: The output format ('text', 'markdown', or 'json').
        name: The template name (without .jinja2 extension).

    Returns:
        The loaded Jinja2 Template object.

    Raises:
        TemplateNotFound: If the template file doesn't exist.
    """
    env = get_jinja_env()
    template_path = f"{format}/{name}.jinja2"
    return env.get_template(template_path)


def render_template(
    format: str,
    name: str,
    format_sha: Optional[Callable[[str], str]] = None,
    **context: Any,
) -> str:
    """Render a template with the given context.

    Args:
        format: The output format ('text', 'markdown').
        name: The template name (without .jinja2 extension).
        format_sha: Optional function to format SHA strings.
        **context: Template context variables.

    Returns:
        The rendered template string.

    Raises:
        TemplateNotFound: If the template file doesn't exist.
    """
    template = load_template(format, name)

    # Add default format_sha if not provided
    if format_sha is None:
        format_sha = lambda sha, use_short=False: sha[:11] if use_short else sha

    return template.render(format_sha=format_sha, **context)


def render_to_format(
    format: str,
    template_name: str,
    data: Any,
    format_sha: Optional[Callable[[str], str]] = None,
    pretty: bool = False,
    **extra_context: Any,
) -> str:
    """Render data to the specified output format.

    For JSON format, serializes the data directly.
    For text/markdown, uses templates.

    Args:
        format: The output format ('text', 'markdown', 'json').
        template_name: The template name for text/markdown formats.
        data: The data to render (dict or object with to_dict method).
        format_sha: Optional function to format SHA strings.
        pretty: For JSON, whether to pretty-print.
        **extra_context: Additional context variables for templates.

    Returns:
        The formatted output string.
    """
    if format == OutputFormat.JSON.value or format == "json":
        # Convert to dict if needed
        if hasattr(data, "to_dict"):
            data = data.to_dict()
        return json.dumps(data, default=str, indent=2 if pretty else None) + "\n"

    # For text/markdown, use templates
    # Prepare context - if data has to_dict, use that; otherwise pass as-is
    if hasattr(data, "to_dict"):
        # For template rendering, we often need the object itself, not just dict
        context = {template_name: data, **extra_context}
    else:
        context = {template_name: data, **extra_context}

    return render_template(format, template_name, format_sha=format_sha, **context)
