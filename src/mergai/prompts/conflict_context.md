## Conflict Context

- ALWAYS make sure the conflict markers are COMPLETELY removed from the final code.
- If any of guidelines cannot be followed, do not attempt to resolve the conflict.
- Instead, leave the conflict markers intact and provide a clear explanation of the issue, highlighting why it cannot be resolved automatically.

```diff
<<<<<<< HEAD
||||||| BASE
=======
>>>>>>>
```

- their commits: commits between base and theirs for each file


### Note format

- `ours_commit`, `theirs_commit`, `base_commit` - commit details (see format below)
- `files` - list of files with merge conflict
- `conflict_types` - dict[key: path, value: conflict type], possible values: 
  - "both modified"
  - "both added"
  - "deleted by them"
  - "added by us"
  - "deleted by us"
  - "added by them"
  - "unknown"
- `diffs` - dict[key: path, value: hunk]:
  - hunks indicating the conflicting code
  - the diffs can be compressed, huge blocks will have `(... X more deleted/added lines...)`
  - the conflicted filed contains the conflict markers indicating the sections of code that are in conflict
- `their_commits` - dict[key: path, value: list of commits]:
  - list of commits which modified given file

- Conflict markers:

```diff
<<<<<<< HEAD
||||||| BASE
=======
>>>>>>>
```

- `commit` format:

The commit fields contain expanded commit information with configurable fields.
Default fields include: hexsha, authored_date, summary, and author.

```json
"commit": {
    "hexsha": "full commit SHA (40 chars)",
    "author": {
        "name": "author name",
        "email": "author@email.com"
    },
    "authored_date": 1234567890,
    "summary": "first line of commit message"
}
```

Optional fields (can be enabled via config):
- `short_sha`: shortened SHA (11 chars)
- `message`: full commit message
- `parents`: list of parent commit SHAs

- `conflict_context` structure:

```json
"conflict_context": {
    "ours_commit": { /* commit object */ },
    "theirs_commit": { /* commit object */ },
    "base_commit": { /* commit object */ },
    "files": ["path/to/file1.py", "path/to/file2.py"],
    "conflict_types": {
        "path/to/file1.py": "both modified",
        "path/to/file2.py": "deleted by them"
    },
    "diffs": {
        "path/to/file1.py": "diff content..."
    },
    "their_commits": {
        "path/to/file1.py": [
            { /* commit object */ },
            { /* commit object */ }
        ]
    }
}
```