## Review Context

The `Review Context` JSON below lists the unresolved review threads on the
pull request that need your attention. Threads that were resolved, marked
outdated, opted out, or already answered by automation have been filtered
out - every thread here is one you must classify as `addressed` or
`unaddressed`.

### Fields

Each entry is keyed by a **thread id** (an opaque string). Use that exact
key in your response's `addressed` / `unaddressed` maps.

- `path`: Repository-relative file the thread is anchored to (may be
  null for a thread not tied to a specific line).
- `line`: Line number the comment refers to (may be null when the line
  is outdated or the comment is file-level).
- `diff_hunk`: The diff hunk GitHub shows alongside the comment - the
  exact code the reviewer was looking at. Use it to locate the relevant
  code, then open the current file to see its present state (the hunk may
  be slightly stale).
- `comments`: The full conversation on the thread, oldest first. Each has
  `author`, `created_at`, and `body`. Read all of them before deciding.

Use this context to locate what each reviewer asked for, then edit the
relevant files in the working tree. The set of files you edit must match
what you list under `resolved` / `modified` in your response.
