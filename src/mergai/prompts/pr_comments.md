
## PR comments

- The comments refer to files modified in the previous attempt to fix the merge conflict,
- All comments **MUST** be adressed,
- All modified files should be included in the `resolved` field (see `Output format`)
- In case of comments which were not adressed, add appropriate explanation in:
  - the `unresolved` field, per each file, or
  - in the `summary`.
- Make **ONLY** changes related to the comments
- See `Note format` for details of the format
- See `Note data` for the comments data
- There are two types of comments:
  - review comments - refer to specific line in a file for a given commit
  - normal comment - general comment

### Note format

```json
"pr_comments": {
    "<id1 of review comment>": {
        "commit_id": "<commit SHA>",
        "user": "user which made a comment",
        "created_at": "date of creating the comment",
        "body": "content of the comment",
        "path": "relative path to file which comment refers to",
        "line": "line number, or end line if start_line is set",
        "start_line": "(optional) start line if comment refers to multiple lines",
        "line_str": "line in format N in case of single line comment or N1-N2 in case of multiple lines"
    },
    "<id2 of normal comment>": {
        "user": "user which made a comment",
        "created_at": "date of creating the comment",
        "body": "content of the comment",
    },
}
```
