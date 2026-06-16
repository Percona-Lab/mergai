"""Tests for the aggregated CI-fix PR comment (A2).

Covers the summary renderer (``_render_ci_notification_summary``) as pure
logic, and the ``mergai ci comment post`` aggregation - many recorded
attempts collapse to a single PR comment per target PR.
"""

import mergai.commands.ci as ci
from mergai.ci.comments import _render_ci_notification_summary


def _entry(workflow, outcome, *, run_id, pr_number=10, commit_sha=None, summary=""):
    return {
        "workflow": workflow,
        "outcome": outcome,
        "run_id": run_id,
        "pr_number": pr_number,
        "commit_sha": commit_sha,
        "response": {"summary": summary},
        "posted_at": None,
        "posted_comment_url": None,
    }


# --- renderer -------------------------------------------------------------


def test_summary_renders_each_check_status_one_after_another():
    entries = [
        _entry("format", "fixed", run_id="1", commit_sha="c0600df8fade1234"),
        _entry("build-and-test", "already_resolved", run_id="2"),
    ]
    body = _render_ci_notification_summary(entries)
    # No title, no table - just each check's status, one after another.
    assert "### mergai CI auto-fix summary" not in body
    assert "| Check |" not in body
    assert "The `format` check was fixed in commit `c0600df8fade`" in body
    assert "No fix needed for the `build-and-test` check" in body
    # The format status comes before the build-and-test one.
    assert body.index("`format`") < body.index("`build-and-test`")


def test_summary_single_entry_is_just_that_status():
    body = _render_ci_notification_summary(
        [_entry("clang-tidy", "unfixable", run_id="9", summary="needs a human")]
    )
    assert "| Check |" not in body
    assert "could not be auto-fixed" in body
    assert "needs a human" in body


# --- comment_post aggregation ---------------------------------------------


class _FakeComment:
    html_url = "https://github.com/o/r/pull/10#issuecomment-1"


class _FakePull:
    def __init__(self, recorder):
        self._recorder = recorder

    def create_issue_comment(self, body):
        self._recorder.append(body)
        return _FakeComment()


class _FakeRepo:
    def __init__(self):
        self.posted_bodies: list[str] = []
        self.get_pull_calls: list[int] = []

    def get_pull(self, number):
        self.get_pull_calls.append(number)
        return _FakePull(self.posted_bodies)


class _FakeNote:
    def __init__(self, comments):
        self.ci_comments = comments
        self.marked: list[tuple[str, str, str | None]] = []

    def pending_ci_comments(self):
        return [c for c in self.ci_comments if c.get("posted_at") is None]

    def get_ci_comment_for_run(self, run_id):
        return next(
            (c for c in self.ci_comments if str(c.get("run_id")) == str(run_id)),
            None,
        )

    def mark_ci_comment_posted(self, run_id, *, posted_at, comment_url):
        self.marked.append((run_id, posted_at, comment_url))
        c = self.get_ci_comment_for_run(run_id)
        if c is not None:
            c["posted_at"] = posted_at
            c["posted_comment_url"] = comment_url
        return c is not None


class _FakeApp:
    def __init__(self, note, repo):
        self.note = note
        self.gh_repo = repo
        self.gh = object()
        self.has_note = True
        self.saved = 0

    def save_note(self, note):
        self.saved += 1


def _run_post(app, **kw):
    opts = {"target": "all", "dry_run": False, "force": False, "review_pr": None}
    opts.update(kw)
    # `.callback` is wrapped by `@click.pass_obj` (needs an active click
    # context); `.__wrapped__` is the bare command function taking `app`.
    ci.comment_post.callback.__wrapped__(app, **opts)


def test_two_pending_entries_post_one_aggregated_comment():
    note = _FakeNote(
        [
            _entry("format", "fixed", run_id="1", commit_sha="abcdef123456"),
            _entry("build-and-test", "already_resolved", run_id="2"),
        ]
    )
    repo = _FakeRepo()
    app = _FakeApp(note, repo)

    _run_post(app)

    # Exactly one PR comment, containing both rows and both detail lines.
    assert len(repo.posted_bodies) == 1
    body = repo.posted_bodies[0]
    assert "`format`" in body and "`build-and-test`" in body
    assert "fixed in commit `abcdef123456`" in body
    assert "No fix needed for the `build-and-test`" in body
    # Both entries marked posted with the same URL.
    assert {m[0] for m in note.marked} == {"1", "2"}
    assert {m[2] for m in note.marked} == {_FakeComment.html_url}
    assert app.saved == 1


def test_dry_run_posts_nothing():
    note = _FakeNote([_entry("format", "fixed", run_id="1", commit_sha="abcdef123456")])
    repo = _FakeRepo()
    app = _FakeApp(note, repo)

    _run_post(app, dry_run=True)

    assert repo.posted_bodies == []
    assert note.marked == []
    assert app.saved == 0


def test_already_posted_excluded_without_force():
    note = _FakeNote(
        [
            _entry("format", "fixed", run_id="1", commit_sha="abcdef123456"),
            {
                **_entry("build-and-test", "already_resolved", run_id="2"),
                "posted_at": "2026-01-01T00:00:00Z",
            },
        ]
    )
    repo = _FakeRepo()
    app = _FakeApp(note, repo)

    _run_post(app)

    body = repo.posted_bodies[0]
    assert "`format`" in body
    assert "`build-and-test`" not in body
    assert {m[0] for m in note.marked} == {"1"}


def test_entries_on_different_prs_post_per_pr():
    note = _FakeNote(
        [
            _entry(
                "format", "fixed", run_id="1", pr_number=10, commit_sha="aaa111bbb222"
            ),
            _entry(
                "build-and-test",
                "fixed",
                run_id="2",
                pr_number=20,
                commit_sha="ccc333ddd444",
            ),
        ]
    )
    repo = _FakeRepo()
    app = _FakeApp(note, repo)

    _run_post(app)

    assert len(repo.posted_bodies) == 2
    assert set(repo.get_pull_calls) == {10, 20}
