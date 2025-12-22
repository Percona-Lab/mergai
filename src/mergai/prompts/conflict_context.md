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

- `ours_commit`, `theirs_commit`, `base_commit` - commits details,
- `files` - list of files with merge conflict,
- `conflict_types` - dict[key: path, value: conflict type], possible values: 
  - "both modified"
  - "both added"
  - "deleted by them"
  - "added by us"
  - "deleted by us"
  - "added by them"
  - "unknown"
- `diffs` - dict[key: path, value: hunk]:
  - hunks indicating the conflicting code,
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

```json
"commit": {
    "hexsha": "commit SHA",
    "author": {
        "name": "name",
        "email": "name@email.com",
    },
    "authored_date": "timestamp of the commit",
    "summary": "commit summary",
    "message": "commit message",
    "parents": ["parent SHA"]
}
```

- `conflict_context` structure:

```json
"conflict_context": {
    "ours_commit": {},
    "theirs_commit": {},
    "base_commit": {},
    "files": [],
    "conflict_types": {},
    "diffs": {},
    "their_commits": {}
}
```