# AI Agent Guide for mergai

This document provides context and guidelines for AI coding assistants working on the mergai project.

## Project Overview

mergai is a CLI tool for AI-assisted merge conflict resolution, designed for maintaining long-running forks that regularly sync with upstream repositories.

**Key features:**
- Prioritizes which upstream commits to merge next based on configurable strategies
- Captures conflict context (diffs, conflict markers, commit history) and passes it to an AI agent
- Stores all merge metadata and AI solutions as git notes
- Manages branches and PRs for both clean merges and conflict resolution workflows

## Architecture

### Directory Structure

```
src/mergai/
├── agents/           # AI agent implementations (GeminiCLI, ClaudeCLI, OpenCode)
│   ├── base.py       # Agent and CliAgent base classes
│   ├── factory.py    # REGISTRY mapping agent types to classes
│   ├── error.py      # AgentError and AgentResult classes
│   └── gemini_cli.py # Reference agent implementation
├── commands/         # Click CLI command handlers
├── prompts/          # Prompt templates for AI agents
├── templates/        # Jinja2 templates for PR bodies
├── utils/            # Utility functions
├── app.py            # Main application logic
├── config.py         # Configuration loading and validation
├── models.py         # Data models (MergaiNote, MergeInfo, etc.)
└── prompt_builder.py # Prompt construction from note data
```

### Key Components

#### Agent System

The agent system uses a plugin-based architecture:

1. **Base classes** (`agents/base.py`):
   - `Agent`: Abstract base with `run()` method
   - `CliAgent`: For CLI-based agents (extends Agent)

2. **Factory** (`agents/factory.py`):
   - `REGISTRY`: Dict mapping agent type strings to classes
   - `create_agent(agent_type, model, yolo, debug)`: Factory function

3. **Reference implementation** (`agents/gemini_cli.py`):
   - `GeminiCLIAgent`: Full implementation showing the expected pattern

#### Configuration

- Config file location: `.mergai/config.yml` in target repository
- Agent configuration: `resolve.agent: <type>:<model>`
  - Example: `gemini-cli:gemini-2.5-pro`
  - Example: `claude-cli:claude-sonnet-4-20250514`

#### Notes System

mergai uses git notes to store merge metadata:
- `refs/notes/mergai` - Main note storage
- `refs/notes/mergai-marker` - Lightweight markers for git log

## Code Style Requirements

- **Python version**: 3.10+
- **Type hints**: Required for all functions
- **Formatter**: Black (line length 88)
- **Linter**: Ruff (pycodestyle, Pyflakes, isort, flake8-bugbear, flake8-comprehensions, pyupgrade, flake8-simplify)
- **Type checker**: mypy

### Commands to Run Before Committing

```bash
# Format code
black src/

# Check formatting without modifying
black --check --diff src/

# Run linter
ruff check src/

# Auto-fix linting issues
ruff check --fix src/

# Run type checker
mypy src/mergai --ignore-missing-imports

# Check for unused dependencies
deptry src/
```

## Adding a New Agent

### Step 1: Create the Agent File

Create `src/mergai/agents/{agent_name}.py`:

```python
from pathlib import Path
from .base import CliAgent
from .error import AgentError, AgentErrorType, AgentResult

class NewAgentCLI(CliAgent):
    def __init__(self, model: str, yolo: bool, debug: bool = False):
        super().__init__(model)
        self.session_id: str | None = None
        self.yolo = yolo
        self.debug = debug

    def get_version(self) -> str:
        """Return the CLI tool version string."""
        # Run: agent --version
        pass

    def build_args(self, prompt: str) -> list:
        """Build command-line arguments for the agent."""
        pass

    def run_prompt(self, prompt: str) -> dict:
        """Execute the prompt and parse streaming JSON output."""
        pass

    def run(self, prompt: str, response_file: Path | None = None) -> AgentResult:
        """Main entry point. Returns AgentResult with result or error."""
        pass

    def read_session(self, session_id: str) -> dict | None:
        """Read session data from agent's storage location."""
        pass

    def parse_stats(self, session_data: dict) -> dict:
        """Extract token usage statistics from session data."""
        pass

    def get_session_data(self) -> dict | None:
        """Return session data from the last run."""
        pass

    def get_session_id(self) -> str | None:
        """Return session ID from the last run."""
        pass
```

### Step 2: Register in Factory

Update `src/mergai/agents/factory.py`:

```python
from .new_agent import NewAgentCLI

REGISTRY: dict[str, type[Agent]] = {
    "gemini-cli": GeminiCLIAgent,
    "new-agent": NewAgentCLI,  # Add new agent
}
```

### Step 3: Update Documentation

Add configuration example to `README.md`:

```yaml
resolve:
  agent: new-agent:model-name
  max_attempts: 3
```

## Error Handling

### Error Types

Use `AgentError` from `agents/error.py`:

```python
from .error import AgentError, AgentErrorType

# For CLI execution failures
raise AgentError(AgentErrorType.AGENT_EXECUTION, "error message")

# For response parsing failures
raise AgentError(AgentErrorType.PARSING_RESULT, "error message")
```

### Agent Result

Always return `AgentResult` from the `run()` method:

```python
from .error import AgentResult

# On success
return AgentResult(result={"response": ..., "stats": ..., "agent_info": ...})

# On error
return AgentResult(error=AgentError(AgentErrorType.AGENT_EXECUTION, "message"))
```

## Testing

### Manual Testing with Real Conflicts

1. Set up a test repository with known merge conflicts
2. Configure the agent in `.mergai/config.yml`
3. Run `mergai resolve -y` (yolo mode)
4. Verify:
   - Agent executes successfully
   - Response is properly parsed
   - Session stats are captured in notes
   - Conflict resolution is reasonable

### Development Installation

```bash
# Create virtual environment
python -m venv .venv
source .venv/bin/activate

# Install with dev dependencies
pip install -e ".[dev]"

# Set up pre-commit hooks
pre-commit install
```

## Key Files Reference

| File | Purpose |
|------|---------|
| `src/mergai/agents/base.py` | Agent base classes (`Agent`, `CliAgent`) |
| `src/mergai/agents/factory.py` | Agent registry and factory function |
| `src/mergai/agents/error.py` | `AgentError`, `AgentErrorType`, `AgentResult` |
| `src/mergai/agents/gemini_cli.py` | Reference agent implementation |
| `src/mergai/prompt_builder.py` | Builds prompts from MergaiNote data |
| `src/mergai/config.py` | Configuration loading and validation |
| `src/mergai/models.py` | Data models (`MergaiNote`, `MergeInfo`, etc.) |
| `src/mergai/app.py` | Main application logic |
| `pyproject.toml` | Project metadata, dependencies, tool configs |

## Common Patterns

### Subprocess Execution

```python
import subprocess
import os

proc = subprocess.Popen(
    args,
    stdout=subprocess.PIPE,
    stderr=subprocess.PIPE,
    text=True,
    env=os.environ.copy(),
)

for line in proc.stdout:
    event = json.loads(line)  # Parse streaming JSON
    # Handle event...

rc = proc.wait()
if rc != 0:
    raise AgentError(AgentErrorType.AGENT_EXECUTION, f"exited with code {rc}")
```

### JSON Streaming Output

Agents should output JSON events, one per line:
- `{"type": "init", "session_id": "..."}` - Session started
- `{"type": "message", "role": "assistant", "content": "..."}` - Agent response
- `{"type": "result", "status": "success", "timestamp": "..."}` - Completion

### Session Data Structure

```python
{
    "response": {...},      # Parsed JSON response from agent
    "stats": {              # Token usage statistics
        "models": {
            "model-name": {
                "tokens": {"input": N, "output": M}
            }
        }
    },
    "agent_info": {
        "agent_type": "agent-name",
        "version": "x.y.z"
    },
    "session": {...}        # Raw session data (optional)
}
```
