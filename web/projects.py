"""ProjectStore — filesystem-backed CRUD for projects and their uploaded files.

Layout (under DATA_ROOT, default ./data):
    projects/{id}/meta.json     {id, name, created_at}
    projects/{id}/files/*       raw uploaded files (the source of truth for the file list)
    lancedb/{id}/               isolated LanceDB for this project

The file list is read directly from the files/ directory (no drift with meta.json).
Only files with supported suffixes are exposed.
"""

import json
import shutil
import uuid
from datetime import datetime, timezone
from pathlib import Path

from src.vectordb.client import invalidate_db_cache
from src.vectordb.indexer import SUPPORTED_SUFFIXES, resolve_index_settings


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class ProjectStore:
    """Filesystem-backed store for projects and uploaded files."""

    def __init__(self, root: str | Path = "data"):
        self.root = Path(root)
        self.projects_dir = self.root / "projects"
        self.lancedb_dir = self.root / "lancedb"
        self.projects_dir.mkdir(parents=True, exist_ok=True)
        self.lancedb_dir.mkdir(parents=True, exist_ok=True)

    # ── paths ──

    def _project_dir(self, pid: str) -> Path:
        return self.projects_dir / pid

    def _meta_path(self, pid: str) -> Path:
        return self._project_dir(pid) / "meta.json"

    def files_dir(self, pid: str) -> Path:
        return self._project_dir(pid) / "files"

    def db_path(self, pid: str) -> str:
        return str(self.lancedb_dir / pid)

    # ── meta ──

    def _read_meta(self, pid: str) -> dict:
        return json.loads(self._meta_path(pid).read_text(encoding="utf-8"))

    def _write_meta(self, pid: str, meta: dict) -> None:
        self._meta_path(pid).write_text(
            json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    # ── projects ──

    def list_projects(self) -> list[dict]:
        """All projects, newest first."""
        projects = []
        for d in self.projects_dir.iterdir():
            if d.is_dir() and (d / "meta.json").exists():
                try:
                    projects.append(self._read_meta(d.name))
                except Exception:
                    continue
        projects.sort(key=lambda m: m.get("created_at", ""), reverse=True)
        return projects

    def get(self, pid: str) -> dict | None:
        if self._meta_path(pid).exists():
            return self._read_meta(pid)
        return None

    def create(self, name: str) -> dict:
        pid = uuid.uuid4().hex
        self.files_dir(pid).mkdir(parents=True, exist_ok=True)
        meta = {"id": pid, "name": name.strip() or "Untitled", "created_at": _now_iso()}
        self._write_meta(pid, meta)
        return meta

    def rename(self, pid: str, name: str) -> dict:
        meta = self._read_meta(pid)
        meta["name"] = name.strip() or meta["name"]
        self._write_meta(pid, meta)
        return meta

    def delete(self, pid: str) -> None:
        invalidate_db_cache(self.db_path(pid))
        shutil.rmtree(self._project_dir(pid), ignore_errors=True)
        shutil.rmtree(Path(self.db_path(pid)), ignore_errors=True)

    # ── per-project indexing settings ──

    def get_index_settings(self, pid: str) -> dict:
        """Indexing hyperparameters for this project, defaults filled in.

        Projects without saved settings get the global vdb_settings values.
        """
        meta = self._read_meta(pid)
        return resolve_index_settings(meta.get("index_settings"))

    def set_index_settings(self, pid: str, settings: dict) -> dict:
        meta = self._read_meta(pid)
        meta["index_settings"] = resolve_index_settings(settings)
        self._write_meta(pid, meta)
        return meta["index_settings"]

    # ── files (directory is the source of truth) ──

    def list_files(self, pid: str) -> list[dict]:
        """Uploaded files for a project: [{name, size}], sorted by name."""
        fdir = self.files_dir(pid)
        if not fdir.exists():
            return []
        files = [
            {"name": f.name, "size": f.stat().st_size}
            for f in fdir.iterdir()
            if f.is_file() and f.suffix.lower() in SUPPORTED_SUFFIXES
        ]
        files.sort(key=lambda f: f["name"].lower())
        return files

    def file_exists(self, pid: str, filename: str) -> bool:
        """Whether a file with this name is already uploaded to the project."""
        return (self.files_dir(pid) / Path(filename).name).exists()

    def add_file(self, pid: str, filename: str, content: bytes) -> None:
        """Save an uploaded file. Raises ValueError on unsupported suffix."""
        name = Path(filename).name  # strip any path components
        if Path(name).suffix.lower() not in SUPPORTED_SUFFIXES:
            raise ValueError(
                f"Unsupported file type: {name}. "
                f"Supported: {', '.join(sorted(SUPPORTED_SUFFIXES))}"
            )
        self.files_dir(pid).mkdir(parents=True, exist_ok=True)
        (self.files_dir(pid) / name).write_bytes(content)

    def rename_file(self, pid: str, old: str, new: str) -> None:
        new_name = Path(new).name
        if Path(new_name).suffix.lower() not in SUPPORTED_SUFFIXES:
            raise ValueError(f"Unsupported file type: {new_name}")
        src = self.files_dir(pid) / Path(old).name
        dst = self.files_dir(pid) / new_name
        src.rename(dst)

    def delete_file(self, pid: str, name: str) -> None:
        target = self.files_dir(pid) / Path(name).name
        target.unlink(missing_ok=True)
