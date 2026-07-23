import json
from pathlib import Path


class StateStore:
    DEFAULT_DIR = ".cache/mergai"
    NOTE_FILE = "note.json"
    PICK_FILE = "merge_pick.json"
    JSON_INDENT = 2

    def __init__(self, dir: str):
        self.path = Path(dir) / self.DEFAULT_DIR
        self.path.mkdir(parents=True, exist_ok=True)

    def note_path(self) -> Path:
        return self.path / self.NOTE_FILE

    def note_exists(self) -> bool:
        return self.note_path().exists()

    def remove_note(self):
        if self.note_exists():
            self.note_path().unlink()

    def save_note(self, note: dict):
        with open(self.note_path(), "w") as f:
            json.dump(note, f, indent=self.JSON_INDENT)

    def load_note(self) -> dict:
        with open(self.note_path()) as f:
            result: dict = json.load(f)
            return result

    # --- Merge-pick handoff ---
    #
    # `mergai fork merge-pick --record` writes the chosen pick's metadata here;
    # `mergai context init` reads it and attaches it to the note as merge_pick.
    # This keeps the pick's sha/summary inside mergai rather than round-tripping
    # them through the shell between the two CLI calls.

    def pick_path(self) -> Path:
        return self.path / self.PICK_FILE

    def pick_exists(self) -> bool:
        return self.pick_path().exists()

    def remove_pick(self):
        if self.pick_exists():
            self.pick_path().unlink()

    def save_pick(self, pick: dict):
        with open(self.pick_path(), "w") as f:
            json.dump(pick, f, indent=self.JSON_INDENT)

    def load_pick(self) -> dict:
        with open(self.pick_path()) as f:
            result: dict = json.load(f)
            return result
