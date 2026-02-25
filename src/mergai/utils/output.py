"""Output format utilities for mergai CLI commands.

This module provides a unified way to handle output formats across commands.
"""

from enum import Enum
from typing import Callable, Optional
import click


class OutputFormat(str, Enum):
    """Supported output formats for CLI commands."""

    TEXT = "text"
    MARKDOWN = "markdown"
    JSON = "json"


def format_option(
    default: OutputFormat = OutputFormat.TEXT,
    include_json: bool = True,
) -> Callable:
    """Create a Click option decorator for output format selection.

    Provides:
    - --format with choices: text, markdown, json (or text, markdown if include_json=False)
    - --md / --markdown aliases for markdown format
    - --json alias for json format (if include_json=True)

    Args:
        default: The default output format.
        include_json: Whether to include JSON as a format option.

    Returns:
        A decorator function that adds format options to a Click command.

    Example:
        @click.command()
        @format_option()
        def my_command(format: str):
            ...

        @click.command()
        @format_option(default=OutputFormat.MARKDOWN, include_json=False)
        def prompt_command(format: str):
            ...
    """

    def decorator(func: Callable) -> Callable:
        # Build choices based on include_json
        if include_json:
            choices = [
                OutputFormat.TEXT.value,
                OutputFormat.MARKDOWN.value,
                OutputFormat.JSON.value,
            ]
        else:
            choices = [OutputFormat.TEXT.value, OutputFormat.MARKDOWN.value]

        # Add --format option
        func = click.option(
            "--format",
            "format",
            type=click.Choice(choices),
            default=default.value,
            help=f"Output format (default: {default.value}).",
            show_default=False,
        )(func)

        # Add --md / --markdown aliases
        def set_markdown(ctx, param, value):
            if value:
                ctx.params["format"] = OutputFormat.MARKDOWN.value
            return value

        func = click.option(
            "--md",
            "--markdown",
            "markdown_flag",
            is_flag=True,
            is_eager=True,
            expose_value=False,
            callback=set_markdown,
            help="Output in markdown format (alias for --format markdown).",
        )(func)

        # Add --json alias (only if JSON is included)
        if include_json:

            def set_json(ctx, param, value):
                if value:
                    ctx.params["format"] = OutputFormat.JSON.value
                return value

            func = click.option(
                "--json",
                "json_flag",
                is_flag=True,
                is_eager=True,
                expose_value=False,
                callback=set_json,
                help="Output in JSON format (alias for --format json).",
            )(func)

        return func

    return decorator
