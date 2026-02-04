## Merge context

Always make sure you understood the changes made by the upstream.
Don't go into too much details if the merged commit is not relevant for the fork.
Always check if there were any files auto merged by git. If so make sure the changes are reviewed.

### Note format

```json
  "merge_context": {
    "merge_commit": "hash of merge commit",
    "merged_commits": [
      "hash of merged commit 1",
      "hash of merged commit 1",
    ],
    "important_files_modified": [
        "important file modified 1",
        "important file modified 2",
    ],
    "timestamp": "the timestamp"
  }

```