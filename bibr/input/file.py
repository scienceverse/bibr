from dataclasses import dataclass
from pathlib import Path


@dataclass
class InputFormat:
    file_extension: str
    detected_mime_type: str
    file_type: str


@dataclass
class InputFile:
    path: str | Path
    file_hash: str | None = None
    file_name: str | None = None
    input_format: InputFormat | None = None
    is_supported: bool = False
    is_valid: bool = False
    is_corrupted: bool = False
    is_encrypted: bool = False
    native_artifact: object | None = None

    def __post_init__(self):
        self.path = Path(self.path)
        self.file_name = self.path.name
        self.file_extension = self.path.suffix
