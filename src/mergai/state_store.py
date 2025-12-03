from pathlib import Path
import json


class StateStore:
    DEFAULT_DIR = ".mergai_state"
    NOTE_FILE = "note.json"
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
        with open(self.note_path(), "r") as f:
            return json.load(f)
