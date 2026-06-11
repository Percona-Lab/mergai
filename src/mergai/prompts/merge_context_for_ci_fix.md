## Merge Context

The `Merge Context` JSON below describes the merge that produced this
branch. Use it as the starting point for diagnosing the CI failure.

### Fields

- `merge_info`: target branch and merge commit SHA — what was merged
  into what. `merge_info.merge_commit_sha` is the {{ upstream_term }} commit just
  brought in; `merge_info.target_branch` is the {{ fork_term }} branch.
- `merge_context`: bookkeeping for a clean (non-conflicting) merge —
  list of {{ upstream_term }} commits brought in, the strategy used, and any
  files explicitly tracked.
- `conflict_context`: only present when the {{ upstream_term }} merge had
  conflicts. Lists the conflicted files, conflict type per file
  (`both modified` / `both added` / etc.), the diffs that caused the
  conflict, and the {{ upstream_term }} commits per file that introduced the
  changes.
- `solutions`: prior agent / human solutions recorded on this branch.
  Each entry has a `type`:
    - `conflict_resolution` — how the {{ upstream_term }} merge conflicts were
      resolved. `response.resolved` lists the files the resolver
      edited; `response.summary` explains the approach.
    - `ci_fix` — a previous CI-fix attempt on this branch. Each
      carries a `request` dict with the workflow / run_id that
      triggered it.
  Skim these to know what's already been done; you can also run
  `mergai show <commit>` to view the full note attached to any
  specific commit.

### How to use it

1. The CI failure happened on top of this state. The most likely
   reasons it's broken: an {{ upstream_term }} change you'll find in
   `merge_context.merged_commits`, a {{ fork_term }}-specific assumption that
   no longer holds, or a flaw in the conflict-resolution `solutions`.
2. Cross-reference the failure's affected files with
   `conflict_context.files` and `solutions[*].response.resolved` —
   if there's an overlap, the prior conflict resolution is a
   strong suspect.
3. Use `git log --oneline -20`, `git show <sha>`, and
   `mergai show <commit>` to dig in.
