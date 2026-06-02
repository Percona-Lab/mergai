## CI Fix Context

The `CI Fix Context` JSON below describes the failing CI run.

### Fields

- `workflow_name`: Name of the failing GitHub Actions workflow.
- `run_id`: The workflow run ID.
- `pr_number`: Pull request number associated with the run.
- `summary`: One-line description of the failure (e.g. "clang-tidy reported 3 findings in 2 files", or "clang-tidy failed before producing SARIF; using log of job ...").
- `files_affected`: Repository-relative paths implicated by the failure. May be empty when the source is a build log rather than a structured findings list.
- `details`: The actual failure data. Either:
  - A Markdown list of static-analysis findings, one per line, in the form `- [<level>] <rule_id> at <file>:<line>` followed by an indented message line; or
  - An excerpt from the failing job's log when the workflow died before producing a structured report (e.g. a Bazel loading-phase error). The excerpt is anchored on the failing step and may include a head + tail with an omission marker if the section was too large to keep whole.

Use `details` to identify what actually broke, then edit the relevant files in the working tree to fix it. The set of files you edit must match what you list under `resolved` / `modified` in your response.
