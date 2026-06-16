## Merge Context

The `Merge Context` JSON below describes the merge that produced this
branch. Use it as background for the code under review: it tells you what
was merged and how prior conflicts / fixes were resolved, so you can relate
a review comment to the change it refers to.

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
    - `ci_fix` — a previous CI-fix attempt on this branch.
    - `review_fix` — a previous review-comment fix on this branch.
  Skim these to know what's already been done; you can also run
  `mergai show <commit>` to view the full note attached to any
  specific commit.

### How to use it

1. Read the review comment first - it is the task. The merge context is
   only background to help you understand *why* the code looks the way it
   does.
2. If a comment concerns code touched by the merge, cross-reference it with
   `merge_context.merged_commits`, `conflict_context.files`, and
   `solutions[*].response.resolved` to see what changed and how it was
   resolved.
3. Use `git log --oneline -20`, `git show <sha>`, and
   `mergai show <commit>` to dig in.
