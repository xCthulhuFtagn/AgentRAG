"""Tests for the web ProjectStore (filesystem-backed CRUD)."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from web.projects import ProjectStore


@pytest.fixture
def store(tmp_path):
    return ProjectStore(root=tmp_path)


def test_create_and_list(store):
    meta = store.create("Alpha")
    assert meta["name"] == "Alpha"
    assert "id" in meta
    projects = store.list_projects()
    assert len(projects) == 1
    assert projects[0]["id"] == meta["id"]


def test_rename(store):
    pid = store.create("Old")["id"]
    store.rename(pid, "New")
    assert store.get(pid)["name"] == "New"


def test_delete_removes_files_and_db(store):
    pid = store.create("Temp")["id"]
    store.add_file(pid, "note.txt", b"hello world")
    assert store.files_dir(pid).exists()
    store.delete(pid)
    assert store.get(pid) is None
    assert not store.files_dir(pid).exists()


def test_add_list_delete_file(store):
    pid = store.create("Docs")["id"]
    store.add_file(pid, "a.txt", b"alpha")
    store.add_file(pid, "b.md", b"# beta")
    names = [f["name"] for f in store.list_files(pid)]
    assert names == ["a.txt", "b.md"]
    store.delete_file(pid, "a.txt")
    assert [f["name"] for f in store.list_files(pid)] == ["b.md"]


def test_rename_file(store):
    pid = store.create("Docs")["id"]
    store.add_file(pid, "old.txt", b"data")
    store.rename_file(pid, "old.txt", "new.txt")
    assert [f["name"] for f in store.list_files(pid)] == ["new.txt"]


def test_unsupported_file_rejected(store):
    pid = store.create("Docs")["id"]
    with pytest.raises(ValueError):
        store.add_file(pid, "evil.exe", b"\x00")


def test_isolation_separate_db_paths(store):
    a = store.create("A")["id"]
    b = store.create("B")["id"]
    assert store.db_path(a) != store.db_path(b)
    assert a in store.db_path(a)
