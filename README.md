# mergai

A CLI tool for AI-assisted merge conflict resolution, designed for maintaining long-running forks that regularly sync with upstream repositories.

mergai automates the merge workflow by:
- Prioritizing which upstream commits to merge next based on configurable strategies (conflicts, large commits, important files)
- Capturing conflict context (diffs, conflict markers, commit history) and passing it to an AI agent
- Storing all merge metadata and AI solutions as git notes, allowing them to travel with commits
- Managing branches and PRs for both clean merges and conflict resolution workflows

Currently, the tool uses Gemini CLI as the AI agent.

**This is under active development; use with caution.**

## Prerequisites

- Python 3.10+ and git installed.
- Gemini CLI available on your `PATH` (used as the default agent). Setup authentication on your own.
- Optional: `GITHUB_TOKEN` or `GH_TOKEN` for commands that read/write PRs.
- Run `mergai config` to configure git settings (conflictstyle, notes display).
- Run `mergai fork init` to initialize the upstream remote from config.


## Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

Verify with `mergai --help`.

## Configuration

mergai is configured via `.mergai/config.yml` in your repository root.

Additionally, `.mergai/invariants.md` is appended to AI prompts when present. Use it to provide project-specific instructions or constraints for the AI agent.

### Config Sections

| Section | Description |
|---------|-------------|
| `fork` | Upstream repository settings and merge-pick prioritization strategies |
| `resolve` | AI agent type and retry attempts |
| `branch` | Branch naming format with token substitution |
| `commit` | Commit message footer |
| `pr` | PR title formats and labels for main/solution PRs |
| `prompt` | Commit fields included in AI prompts |
| `finalize` | Post-merge finalization mode (squash/keep/fast-forward) |
| `config` | Git settings (conflictstyle, notes display) |

### Example Configuration

```yaml
fork:
  upstream_url: https://github.com/upstream-org/upstream-repo.git
  upstream_branch: main
  upstream_remote: upstream
  merge_picks:
    most_recent_fallback: true
    # Strategies evaluated in order; first match determines priority
    strategies:
      - conflict: true
      - huge_commit: "(num_of_files >= 100 or num_of_lines >= 5000 or num_of_dirs > 10)"
      - branching_point: true
      - important_files:
          - src/core/critical_module.cpp
          - src/api/public_api.h

resolve:
  agent: gemini-cli:gemini-2.5-pro
  max_attempts: 3

branch:
  name_format: "mergai/%(target_branch)-%(target_branch_short_sha)-%(merge_commit_short_sha)/%(type)"

commit:
  footer: "Note: commit created by mergai"

pr:
  main:
    title_format: "Merge '%(merge_commit_short_sha)' into '%(target_branch)'"
    labels:
      - "ci-skip-format"
  solution:
    title_format: "Resolve conflicts for merge '%(merge_commit_short_sha)' into '%(target_branch)'"
    labels:
      - "ci-skip-format"

prompt:
  # Valid values: hexsha, short_sha, author, authored_date, summary, message, parents
  commit_fields:
    - hexsha

finalize:
  # Available modes: squash, keep, fast-forward
  mode: fast-forward

config:
  git:
    conflictstyle: diff3
    notes:
      display: true
      marker_text: "mergai note available, use `mergai show <commit>` to view it."
```

### Merge-Pick Strategies

The `fork.merge_picks.strategies` list defines how `mergai fork merge-pick` prioritizes unmerged commits. Strategies are evaluated in order; the first matching strategy determines a commit's priority.

| Strategy | Description |
|----------|-------------|
| `conflict: true` | Prioritizes the first commit that would cause merge conflicts |
| `huge_commit: "<expr>"` | Prioritizes commits matching a size expression. Variables: `num_of_files`, `num_of_lines`, `lines_added`, `lines_deleted`, `num_of_dirs` |
| `branching_point: true` | Prioritizes commits that are branching points in upstream (have multiple children) |
| `important_files: [...]` | Prioritizes commits that modify any of the listed file paths |

Set `most_recent_fallback: true` to select the most recent unmerged commit when no strategy matches.

### Branch Naming Format

The `branch.name_format` setting controls how mergai names branches. Available tokens:

| Token | Description | Required |
|-------|-------------|----------|
| `%(target_branch)` | Target branch name | yes |
| `%(target_branch_sha)` | Full SHA of target branch (40 chars) | one of these |
| `%(target_branch_short_sha)` | Short SHA of target branch (11 chars) | one of these |
| `%(merge_commit_sha)` | Full SHA of merge commit (40 chars) | one of these |
| `%(merge_commit_short_sha)` | Short SHA of merge commit (11 chars) | one of these |
| `%(type)` | Branch type: `main`, `conflict`, or `solution` | no |

The format must contain `%(target_branch)`, at least one of the target branch SHA tokens, and at least one of the merge commit SHA tokens.

Example: `mergai/%(target_branch)-%(target_branch_short_sha)-%(merge_commit_short_sha)/%(type)` produces `mergai/release-8.0-abc12345678-def98765432/solution`.

### Notes Storage

mergai stores context and solutions as git notes attached to commits. This allows metadata to travel with commits without modifying commit history.

**Note references:**
- `refs/notes/mergai` - main note storage containing context, solutions, and merge info
- `refs/notes/mergai-marker` - lightweight markers for quick identification in `git log`

**Note contents and when they are created:**

| Field | Description | Created by |
|-------|-------------|------------|
| `merge_info` | Merge metadata (target branch, merge commit SHA, target branch SHA) | `mergai context init` |
| `merge_context` | Auto-merged files info, merge strategy | `mergai merge` (on success or conflict) |
| `conflict_context` | Conflicting files with diffs and conflict markers | `mergai merge` (on conflict only) |
| `solutions` | AI or human resolutions with file changes | `mergai resolve`, `mergai commit sync` |

**Commands:**
- `mergai notes update` - Fetch and merge notes from remote
- `mergai notes push` - Push notes to remote
- `mergai show <commit>` - View note contents
- `mergai context drop` - Remove note pieces

Notes are automatically attached when using `mergai commit` subcommands. Use `mergai context init` to rebuild local state from existing notes.

## Commands

| Command | Description |
|---------|-------------|
| `mergai config` | Configure git settings (conflictstyle, notes display) |
| `mergai fork merge-pick` | Get prioritized commits from upstream based on configured strategies |
| `mergai fork fetch` | Fetch upstream repository |
| `mergai context init` | Initialize merge context with commit SHA and target branch |
| `mergai notes update` | Fetch and merge notes from remote |
| `mergai notes push` | Push mergai notes to remote |
| `mergai merge` | Attempt merge of the context commit |
| `mergai resolve` | Run AI conflict resolution (`-y` for yolo mode) |
| `mergai commit` | Commit with note attached (subcommands: `merge`, `conflict`, `solution`, `sync`) |
| `mergai branch create` | Create working branch (types: `main`, `conflict`, `solution`) |
| `mergai branch push` | Push branch to origin |
| `mergai finalize` | Finalize merge after solution PR is merged |
| `mergai pr create` | Create pull request |
| `mergai pr update` | Update PR body with current note data |
| `mergai log` | List commits with mergai notes |
| `mergai show` | Show stored context/solution for a commit |
| `mergai status` | Show current state |
| `mergai prompt` | Show the prompt that would be sent to the agent |
| `mergai context drop` | Remove note pieces (`--solution`, `--context`, `--all`) |

## Typical Workflow

### Clean Merge (no conflicts)

```bash
mergai fork merge-pick --next          # Get next prioritized commit from upstream
mergai context init <commit>           # Initialize merge context with commit SHA
mergai notes update                    # Fetch and merge notes from remote
mergai merge                           # Attempt merge (succeeds if no conflicts)
mergai commit merge                    # Create merge commit with note attached
mergai branch create main              # Create main working branch
mergai branch push main                # Push main branch to origin
mergai notes push                      # Push mergai notes to remote
mergai pr --repo <repo> create main    # Open PR for the merge
```

### Conflict Resolution

```bash
mergai fork merge-pick --next          # Get next prioritized commit from upstream
mergai context init <commit>           # Initialize merge context with commit SHA
mergai notes update                    # Fetch and merge notes from remote
mergai merge                           # Attempt merge (exits 1 on conflicts)
mergai commit conflict                 # Commit conflict state with markers
mergai branch create conflict          # Create branch with conflict markers
mergai branch create solution          # Create branch for AI solution
mergai resolve -y                      # Run AI conflict resolution (yolo mode)
mergai commit solution                 # Commit resolved files with note
mergai branch push solution            # Push solution branch
mergai branch push conflict            # Push conflict branch
mergai notes push                      # Push mergai notes to remote
mergai pr --repo <repo> create solution # Open PR for the solution
```

### Post-Merge Finalization

After the solution PR is merged into the conflict branch:

```bash
mergai notes update                    # Fetch and merge notes from remote
mergai context init                    # Rebuild context from commit notes
mergai branch create main              # Create main working branch
mergai commit sync                     # Sync manual commits to notes
mergai finalize                        # Finalize into final merge commit
mergai branch push main                # Push main branch to origin
mergai notes push                      # Push mergai notes to remote
mergai pr --repo <repo> create main    # Open PR for the finalized merge
```

## Manual Usage

### Examining Commits with Notes

```bash
mergai log                             # List commits that carry mergai notes
mergai show <commit>                   # Print stored context and solution
mergai show <commit> --context         # Show only the conflict context
mergai show <commit> --solution        # Show only the solution
mergai show <commit> --raw             # Show raw note data
```

### Manual Conflict Resolution

```bash
mergai prompt                          # Show the prompt that would be sent to the agent
mergai resolve                         # Run AI resolution (add -y for yolo mode)
mergai commit solution                 # Commit AI solution and attach note
mergai status                          # Inspect current state
```

### For manual fixes after AI resolution (or instead of it):

```bash
git add <files>                        # Stage manually resolved files
git commit -m "Fix remaining conflicts" # Create regular git commit
mergai commit sync                     # Sync human commits to notes
mergai pr --repo <repo> update solution # Update PR body with new solutions
```
