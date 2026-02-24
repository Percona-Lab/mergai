"""Config command for setting up mergai environment.

This module provides the config command for configuring:
- Git settings (merge.conflictstyle, notes.displayRef)
- Shell completion (bash, zsh, fish)
"""

import click
from typing import Optional, Tuple, Dict, List
from pathlib import Path

from ..config import MergaiConfig, InitConfig


# Constant for notes display ref - not configurable, only whether to set it
NOTES_DISPLAY_REF = "refs/notes/mergai-marker"


def get_git_config_value(repo, key: str, global_scope: bool) -> Optional[str]:
    """Get current value of a git config key.

    Args:
        repo: GitPython Repo instance.
        key: Git config key to read.
        global_scope: If True, read from global config.

    Returns:
        Current value or None if not set.
    """
    try:
        scope_flag = "--global" if global_scope else "--local"
        return repo.git.config(scope_flag, "--get", key)
    except Exception:
        return None


def set_git_config_value(repo, key: str, value: str, global_scope: bool) -> None:
    """Set a git config value.

    Args:
        repo: GitPython Repo instance.
        key: Git config key to set.
        value: Value to set.
        global_scope: If True, set in global config.
    """
    scope_flag = "--global" if global_scope else "--local"
    repo.git.config(scope_flag, key, value)


def configure_git(
    repo,
    config: InitConfig,
    global_scope: bool,
    dry_run: bool,
) -> Tuple[int, int]:
    """Configure git settings for mergai.

    Args:
        repo: GitPython Repo instance.
        config: InitConfig with git settings.
        global_scope: If True, configure globally.
        dry_run: If True, only show what would be done.

    Returns:
        Tuple of (configured_count, skipped_count).
    """
    scope_name = "global" if global_scope else "local"
    click.echo(f"Configuring git ({scope_name} scope):")

    configured = 0
    skipped = 0

    # Configure merge.conflictstyle
    key = "merge.conflictstyle"
    desired_value = config.git.conflictstyle
    current_value = get_git_config_value(repo, key, global_scope)

    if current_value == desired_value:
        click.echo(f"  - {key} = {desired_value} (already set)")
        skipped += 1
    else:
        status_suffix = f" (was: {current_value})" if current_value else ""
        if dry_run:
            click.echo(f"  Would set: {key} = {desired_value}{status_suffix}")
        else:
            set_git_config_value(repo, key, desired_value, global_scope)
            click.echo(
                click.style("  + ", fg="green")
                + f"{key} = {desired_value}{status_suffix}"
            )
            configured += 1

    # Configure notes.displayRef if display is enabled
    if config.git.notes.display:
        key = "notes.displayRef"
        desired_value = NOTES_DISPLAY_REF
        current_value = get_git_config_value(repo, key, global_scope)

        if current_value == desired_value:
            click.echo(f"  - {key} = {desired_value} (already set)")
            skipped += 1
        else:
            status_suffix = f" (was: {current_value})" if current_value else ""
            if dry_run:
                click.echo(f"  Would set: {key} = {desired_value}{status_suffix}")
            else:
                set_git_config_value(repo, key, desired_value, global_scope)
                click.echo(
                    click.style("  + ", fg="green")
                    + f"{key} = {desired_value}{status_suffix}"
                )
                configured += 1

    return configured, skipped


def get_completion_script_content(shell: str) -> str:
    """Get the completion script content for the specified shell.

    Args:
        shell: Shell type ("bash", "zsh", or "fish").

    Returns:
        Completion script content.
    """
    # Click provides built-in shell completion via environment variable
    env_var = "_MERGAI_COMPLETE"

    if shell == "bash":
        return f'eval "$({env_var}=bash_source mergai)"'
    elif shell == "zsh":
        return f'eval "$({env_var}=zsh_source mergai)"'
    elif shell == "fish":
        return f"{env_var}=fish_source mergai | source"
    else:
        return f'eval "$({env_var}={shell}_source mergai)"'


def get_shell_rc_file(shell: str) -> str:
    """Get the typical RC file path for the specified shell.

    Args:
        shell: Shell type.

    Returns:
        Path to the shell's RC file.
    """
    home = Path.home()
    if shell == "bash":
        # Check for .bash_profile on macOS, .bashrc on Linux
        bash_profile = home / ".bash_profile"
        if bash_profile.exists():
            return str(bash_profile)
        return str(home / ".bashrc")
    elif shell == "zsh":
        return str(home / ".zshrc")
    elif shell == "fish":
        return str(home / ".config" / "fish" / "config.fish")
    else:
        return str(home / f".{shell}rc")


def setup_completion(
    config: InitConfig,
    shell_override: Optional[str],
    dry_run: bool,
) -> None:
    """Set up shell completion for mergai.

    Args:
        config: InitConfig with completion settings.
        shell_override: Override shell from CLI (or None to use config).
        dry_run: If True, only show what would be done.
    """
    shell = shell_override or config.completion.shell
    script_line = get_completion_script_content(shell)
    rc_file = get_shell_rc_file(shell)

    click.echo(f"\nShell completion ({shell}):")

    if dry_run:
        click.echo(f"  Would generate {shell} completion setup.")
        click.echo(f"  RC file: {rc_file}")
        return

    click.echo(f"  To enable completion, add this line to {rc_file}:")
    click.echo("")
    click.echo(click.style(f"    {script_line}", fg="cyan"))
    click.echo("")
    click.echo("  Then restart your shell or run:")
    click.echo(f"    source {rc_file}")


def check_git_status(
    repo, config: InitConfig, global_scope: bool
) -> Dict[str, Tuple[Optional[str], str, bool]]:
    """Check current git configuration status.

    Args:
        repo: GitPython Repo instance.
        config: InitConfig with desired settings.
        global_scope: If True, check global config.

    Returns:
        Dict mapping config key to (current_value, desired_value, is_configured).
    """
    result = {}

    # Check merge.conflictstyle
    key = "merge.conflictstyle"
    desired_value = config.git.conflictstyle
    current_value = get_git_config_value(repo, key, global_scope)
    result[key] = (current_value, desired_value, current_value == desired_value)

    # Check notes.displayRef if display is enabled
    if config.git.notes.display:
        key = "notes.displayRef"
        desired_value = NOTES_DISPLAY_REF
        current_value = get_git_config_value(repo, key, global_scope)
        result[key] = (current_value, desired_value, current_value == desired_value)

    return result


def show_status(repo, config: InitConfig, global_scope: bool) -> None:
    """Show current configuration status.

    Args:
        repo: GitPython Repo instance.
        config: InitConfig with desired settings.
        global_scope: If True, check global config.
    """
    scope_name = "global" if global_scope else "local"
    click.echo(f"Configuration status ({scope_name} scope):\n")

    status = check_git_status(repo, config, global_scope)
    all_configured = True

    click.echo("Git settings:")
    for key, (current, desired, is_configured) in status.items():
        if is_configured:
            click.echo(click.style("  [OK] ", fg="green") + f"{key} = {current}")
        elif current is None:
            click.echo(
                click.style("  [--] ", fg="yellow")
                + f"{key} (not set, recommended: {desired})"
            )
            all_configured = False
        else:
            click.echo(
                click.style("  [!!] ", fg="red")
                + f"{key} = {current} (recommended: {desired})"
            )

    # Show notes marker text (from config file, not git config)
    click.echo("")
    click.echo("Notes settings (from config file):")
    click.echo(f'  marker_text: "{config.git.notes.marker_text}"')

    click.echo("")
    if all_configured:
        click.echo(
            click.style("All git settings are configured correctly.", fg="green")
        )
    else:
        click.echo("Run 'mergai config' to configure missing settings.")


@click.command("config")
@click.pass_obj
@click.option(
    "--global",
    "global_scope",
    is_flag=True,
    default=False,
    help="Apply git config globally (user-wide) instead of locally (repository).",
)
@click.option(
    "--dry-run",
    "dry_run",
    is_flag=True,
    default=False,
    help="Show what would be configured without making changes.",
)
@click.option(
    "--git/--no-git",
    "configure_git_flag",
    default=True,
    help="Configure git settings (merge.conflictstyle, notes.displayRef).",
)
@click.option(
    "--completion",
    "configure_completion",
    is_flag=True,
    default=False,
    help="Show shell completion setup instructions.",
)
@click.option(
    "--completion-shell",
    "completion_shell",
    type=click.Choice(["bash", "zsh", "fish"]),
    default=None,
    help="Shell type for completion (default: from config or 'bash').",
)
@click.option(
    "--status",
    "show_status_flag",
    is_flag=True,
    default=False,
    help="Show current configuration status without making changes.",
)
def config(
    app,
    global_scope: bool,
    dry_run: bool,
    configure_git_flag: bool,
    configure_completion: bool,
    completion_shell: Optional[str],
    show_status_flag: bool,
):
    """Configure mergai environment and settings.

    Sets up git configuration and shell completion for optimal mergai usage.

    \b
    Git settings configured:
    - merge.conflictstyle = diff3 (better conflict context)
    - notes.displayRef = refs/notes/mergai-marker (show notes in git log)

    \b
    Settings can be customized in .mergai/config.yml:
        config:
          git:
            conflictstyle: diff3
            notes:
              display: true
              marker_text: "MergAI note available"
          completion:
            shell: bash

    \b
    Examples:
        mergai config                    # Configure git settings locally
        mergai config --global           # Configure git settings globally (user-wide)
        mergai config --dry-run          # Preview changes without applying
        mergai config --status           # Check current configuration status
        mergai config --completion       # Show shell completion setup
        mergai config --completion --completion-shell zsh  # Zsh completion setup
    """
    # Get config section from the loaded config
    config_settings = app.config.config

    # Show status and exit if requested
    if show_status_flag:
        show_status(app.repo, config_settings, global_scope)
        return

    if dry_run:
        click.echo("Dry run mode - no changes will be made.\n")

    configured_count = 0
    skipped_count = 0

    # Configure git settings
    if configure_git_flag:
        conf, skip = configure_git(app.repo, config_settings, global_scope, dry_run)
        configured_count += conf
        skipped_count += skip

    # Show completion setup
    if configure_completion:
        setup_completion(config_settings, completion_shell, dry_run)

    # Summary
    click.echo("")
    if dry_run:
        click.echo("Dry run complete. Run without --dry-run to apply changes.")
    elif configured_count > 0:
        click.echo(click.style("Done! ", fg="green") + "mergai is ready to use.")
    elif skipped_count > 0:
        click.echo(
            click.style("Done! ", fg="green") + "All settings already configured."
        )
    else:
        click.echo("Done!")
