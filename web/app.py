"""NiceGUI web app for Agentic RAG.

Left column: projects (green theme) + files panel for the open project.
Right column: chat. Editing a project's files freezes the chat (blue + tremble)
while the project's vector DB reindexes.

Run: python -m web.app
"""

import html
import uuid
from pathlib import Path

from nicegui import ui, events

from src.logging_setup import setup_logging
from src.vectordb.descriptions import load_descriptions
from src.vectordb.indexer import SUPPORTED_SUFFIXES, table_for_file
from src.vectordb.tools import fetch_all_chunks, merge_chunk_texts
from web import runtime
from web.chat import run_chat
from web.indexing import reindex_project, update_project_index

setup_logging()  # node decisions → console, same logs as the CLI

STORE = runtime.STORE
CSS = (Path(__file__).parent / "static" / "style.css").read_text(encoding="utf-8")

ACCEPT = ".pdf,.docx,.pptx,.txt,.md"


def _snowflakes_html(n: int = 14) -> str:
    """Build a snow overlay — n flakes with varied position/size/speed."""
    flakes = []
    for i in range(n):
        left = (i * 67 + 5) % 100
        duration = 4 + (i % 5)            # 4–8s
        delay = round((i % 7) * 0.6, 1)   # 0–3.6s
        size = round(0.7 + (i % 4) * 0.35, 2)  # 0.7–1.75rem
        char = "❄" if i % 2 == 0 else "❅"
        flakes.append(
            f"<span class='snowflake' style='left:{left}%;"
            f"animation-duration:{duration}s;animation-delay:{delay}s;"
            f"font-size:{size}rem;'>{char}</span>"
        )
    return "".join(flakes)


def index():
    """Root page — runs per client connection (per-client state via closures)."""
    ui.colors(primary="#16a34a")
    ui.add_css(CSS)
    # Drop NiceGUI's default content padding/gap so the h-screen layout fills the
    # viewport exactly (otherwise the page overflows and the chat input scrolls
    # off the bottom).
    ui.query(".nicegui-content").classes("p-0 gap-0")

    # Per-client UI state.
    # ctx["edit"]: None, or {"pid": str, "files": [staged-entry, ...]} while editing files.
    # A staged entry: {name, origin: "disk"|"new", orig_name, content: bytes|None, deleted: bool}.
    ctx = {"open_pid": None, "chat_pid": None, "messages": [], "edit": None}

    # ── refreshable views ──

    def _project_card(p):
        pid = p["id"]
        frozen = runtime.is_frozen(pid)
        active = pid == ctx["open_pid"]
        card_cls = "project-card w-full p-2" + (" active" if active else "")
        # Name: green (button green) on white cards, white on the selected card.
        name_color = "text-white" if active else "text-green-600"
        with ui.card().classes(card_cls):
            with ui.row().classes("w-full items-center no-wrap"):
                ui.label(p["name"]).classes(
                    f"font-medium grow cursor-pointer {name_color}"
                ).on("click", lambda _=None, x=pid: open_project(x))
                with ui.button(icon="more_vert").props("flat dense round"):
                    with ui.menu():
                        ui.menu_item(
                            "Indexing settings",
                            lambda x=pid: index_settings_dialog(x),
                        )
                        ui.menu_item(
                            "RAG process settings",
                            lambda x=pid: rag_process_settings_dialog(x),
                        )
                        ui.menu_item("Open", lambda x=pid: open_project(x))
                        ui.menu_item(
                            "Rename",
                            lambda x=pid, n=p["name"]: rename_project_dialog(x, n),
                        )
                        ui.menu_item("Delete", lambda x=pid: delete_project_dialog(x))
            btn = (
                ui.button("Open in chat", on_click=lambda _=None, x=pid: open_in_chat(x))
                .props("dense")
                .classes("w-full")
            )
            # Color via Quasar props (reliably beats Quasar's own bg-primary):
            # idle = green button / white text; selected = white button / green text.
            if frozen:
                # Openable so you can watch the frozen chat; marked, not usable.
                btn.props("flat").classes("frozen")
                ui.label("Reindexing…").classes("text-xs text-blue-600")
            elif active:
                btn.props("color=white text-color=primary")
            else:
                btn.props("color=primary")

    @ui.refreshable
    def projects_list():
        projects = STORE.list_projects()
        if not projects:
            ui.label("No projects yet").classes("text-gray-400 text-sm")
            return
        # Selected project floats to the top — so clicking it (which resets the
        # scroll on refresh) lands on it instead of jumping away.
        active_pid = ctx["open_pid"]
        projects.sort(key=lambda p: p["id"] != active_pid)
        # Scrollable — show ~2.5 projects so the files panel stays reachable.
        with ui.scroll_area().classes("w-full").style("height: 230px"):
            with ui.column().classes("w-full gap-2"):
                for p in projects:
                    _project_card(p)

    @ui.refreshable
    def staged_files_list():
        # Separate refreshable so on_upload can redraw the list WITHOUT rebuilding
        # the ui.upload widget above it — rebuilding it mid-transfer tears down the
        # connections of other in-flight files (ClientDisconnect → silently lost).
        visible = [s for s in ctx["edit"]["files"] if not s["deleted"]]
        with ui.scroll_area().classes("w-full").style("height: 150px"):
            with ui.column().classes("w-full gap-1"):
                if not visible:
                    ui.label("No files").classes("text-gray-400 text-sm")
                for s in visible:
                    tag = " (new)" if s["origin"] == "new" else ""
                    with ui.row().classes("w-full items-center no-wrap"):
                        ui.icon("description").classes("text-green-600")
                        ui.label(f"{s['name']}{tag}").classes("grow text-sm")
                        ui.button(
                            icon="edit",
                            on_click=lambda _=None, x=s: stage_rename_dialog(x),
                        ).props("flat dense round")
                        ui.button(
                            icon="delete",
                            on_click=lambda _=None, x=s: stage_delete(x),
                        ).props("flat dense round color=red")

    @ui.refreshable
    def files_panel():
        pid = ctx["open_pid"]
        if not pid:
            return
        meta = STORE.get(pid)
        if not meta:
            return
        frozen = runtime.is_frozen(pid)
        editing = bool(ctx["edit"]) and ctx["edit"]["pid"] == pid
        ui.label(f"Files — {meta['name']}").classes("font-bold text-green-800")

        if not editing:
            # ── View mode: scrollable list (~4 files) + always-visible Edit ──
            files = STORE.list_files(pid)
            with ui.scroll_area().classes("w-full").style("height: 150px"):
                with ui.column().classes("w-full gap-1"):
                    if not files:
                        ui.label("No files uploaded").classes("text-gray-400 text-sm")
                    progress = runtime.get_progress(pid)
                    for f in files:
                        with ui.row().classes("w-full items-center no-wrap"):
                            ui.icon("description").classes("text-green-600")
                            # min-w-0 + truncate: an unbreakable long filename
                            # must shrink (ellipsis), not push the row wide.
                            ui.label(f"{f['name']}  ({f['size']} B)").classes(
                                "grow text-sm min-w-0 truncate"
                            ).tooltip(f"{f['name']} ({f['size']} B)")
                            # status: None = pending, True = indexed, False = failed.
                            status = progress.get(f["name"])
                            if status is False:
                                ui.icon("error").classes("text-red-500").tooltip(
                                    "Indexing failed — this file is not searchable"
                                )
                            elif frozen and status is None:
                                ui.spinner(size="sm").classes("text-green-600")
                            view_btn = ui.button(
                                icon="preview",
                                on_click=lambda _=None, x=pid, n=f["name"]:
                                    parsed_text_dialog(x, n),
                            ).props("flat dense round")
                            view_btn.tooltip("View parsed text (as indexed)")
                            if frozen:
                                # Mid-reindex the table may be dropped/rebuilt.
                                view_btn.props("disable")
            edit_btn = ui.button(
                "Edit files", icon="edit", on_click=lambda _=None, x=pid: enter_edit(x)
            ).props("flat dense").classes("w-full")
            if frozen:
                edit_btn.props("disable")
            return

        # ── Edit mode: upload + scrollable staged list + always-visible actions ──
        ui.label("Editing — changes apply on Done").classes("text-xs text-blue-700")
        # hide-uploader-files: the uploader's own file rows are redundant —
        # staged files are listed below with rename/delete actions.
        ui.upload(
            on_upload=lambda e: stage_upload(e),
            auto_upload=True,
            multiple=True,
        ).props(f'accept="{ACCEPT}"').classes("w-full hide-uploader-files")

        staged_files_list()

        with ui.row().classes("w-full no-wrap gap-2"):
            ui.button("Cancel", on_click=lambda _=None: cancel_edit()).props("flat")
            ui.button(
                "Done & update index", icon="check",
                on_click=lambda _=None: commit_edit(),
            ).props("color=primary").classes("grow")

    @ui.refreshable
    def messages_view():
        for m in ctx["messages"]:
            render_message(m)

    @ui.refreshable
    def chat_panel():
        pid = ctx["chat_pid"]
        frozen = bool(pid) and runtime.is_frozen(pid)
        base = (
            "chat-panel w-full max-w-xl h-full flex flex-col gap-2 "
            "p-3 rounded-lg border min-w-0"
        )
        with ui.column().classes(base + (" frozen" if frozen else "")):
            if frozen:
                ui.html(_snowflakes_html()).classes("snowflakes")
            if not pid:
                ui.label("Open a project in chat to start").classes(
                    "text-gray-500 text-lg m-auto"
                )
                return
            meta = STORE.get(pid)
            ui.label(f"Chat — {meta['name'] if meta else pid}").classes(
                "text-lg font-bold text-green-700"
            )
            if frozen:
                with ui.row().classes("items-center gap-2"):
                    ui.spinner(size="sm")
                    ui.label("Reindexing… chat frozen").classes("text-blue-700")
            with ui.scroll_area().classes("grow w-full min-w-0"):
                with ui.column().classes("w-full gap-2 min-w-0"):
                    messages_view()
            with ui.row().classes("w-full no-wrap items-center"):
                inp = (
                    ui.input(placeholder="Ask a question…")
                    .classes("grow")
                    .props("outlined dense")
                )
                send_btn = ui.button(
                    icon="send", on_click=lambda _=None: send_message(inp)
                )
                inp.on("keydown.enter", lambda _=None: send_message(inp))
                if frozen:
                    inp.disable()
                    send_btn.disable()

    def render_message(m):
        if m["role"] == "user":
            ui.html(
                f"<div class='chat-bubble-user'>{html.escape(m['text'])}</div>"
            ).classes("w-full flex justify-end")
        else:
            with ui.column().classes("w-full gap-1 min-w-0"):
                total_in = total_out = 0
                for t in m.get("trace", []):
                    agent = html.escape(str(t.get("agent", "")))
                    decision = html.escape(str(t.get("decision", "")))
                    ti = int(t.get("input_tokens", 0) or 0)
                    to = int(t.get("output_tokens", 0) or 0)
                    total_in += ti
                    total_out += to
                    tok_line = (
                        f"tokens · in {ti:,} / out {to:,} · Σ {ti + to:,}"
                        if (ti or to)
                        else "tokens · —"
                    )
                    info = str(t.get("info", "")).strip()
                    info_html = (
                        f"<div class='trace-step-info'>{html.escape(info).replace(chr(10), '<br>')}</div>"
                        if info
                        else ""
                    )
                    ui.html(
                        "<div class='trace-step'>"
                        f"<div class='trace-step-main'>🔵 {agent}: {decision}</div>"
                        f"{info_html}"
                        f"<div class='trace-step-tok'>{tok_line}</div>"
                        "</div>"
                    )
                if (total_in or total_out) and m.get("trace"):
                    ui.html(
                        "<div class='trace-total'>"
                        f"Σ tokens — in {total_in:,} / out {total_out:,} "
                        f"· total {total_in + total_out:,}</div>"
                    )
                if m.get("text"):
                    ui.markdown(m["text"]).classes("chat-bubble-assistant")
                elif not m.get("trace"):
                    ui.spinner(size="sm")

    # ── actions ──

    def open_project(pid):
        # Switching to a different project discards an in-progress edit session.
        if ctx["edit"] and ctx["edit"]["pid"] != pid:
            ctx["edit"] = None
        ctx["open_pid"] = pid
        projects_list.refresh()
        files_panel.refresh()

    def open_in_chat(pid):
        # Openable even while reindexing — the chat shows frozen (input disabled),
        # so you can switch away and back and still see the freeze.
        if ctx["chat_pid"] != pid:
            ctx["chat_pid"] = pid
            ctx["messages"] = []
        open_project(pid)  # also select the project (highlight card, show its files)
        chat_panel.refresh()

    async def new_project():
        with ui.dialog() as dialog, ui.card():
            ui.label("New project").classes("font-bold")
            name = ui.input("Project name").props("autofocus")
            name.on("keydown.enter", lambda: dialog.submit(name.value))
            with ui.row():
                ui.button("Cancel", on_click=lambda: dialog.submit(None)).props("flat")
                ui.button("Create", on_click=lambda: dialog.submit(name.value))
        result = await dialog
        if result:
            STORE.create(result)
            projects_list.refresh()

    async def rename_project_dialog(pid, current):
        with ui.dialog() as dialog, ui.card():
            ui.label("Rename project").classes("font-bold")
            name = ui.input("New name", value=current).props("autofocus")
            with ui.row():
                ui.button("Cancel", on_click=lambda: dialog.submit(None)).props("flat")
                ui.button("Save", on_click=lambda: dialog.submit(name.value))
        result = await dialog
        if result:
            STORE.rename(pid, result)
            projects_list.refresh()
            files_panel.refresh()
            chat_panel.refresh()

    async def delete_project_dialog(pid):
        with ui.dialog() as dialog, ui.card():
            ui.label("Delete this project?").classes("font-bold")
            ui.label("Removes its files and vector index.").classes(
                "text-sm text-gray-500"
            )
            with ui.row():
                ui.button("Cancel", on_click=lambda: dialog.submit(False)).props("flat")
                ui.button(
                    "Delete", on_click=lambda: dialog.submit(True)
                ).props("color=red")
        if await dialog:
            STORE.delete(pid)
            runtime.clear_progress(pid)
            if ctx["edit"] and ctx["edit"]["pid"] == pid:
                ctx["edit"] = None
            if ctx["open_pid"] == pid:
                ctx["open_pid"] = None
            if ctx["chat_pid"] == pid:
                ctx["chat_pid"] = None
                ctx["messages"] = []
            projects_list.refresh()
            files_panel.refresh()
            chat_panel.refresh()

    async def index_settings_dialog(pid):
        meta = STORE.get(pid)
        if not meta:
            return
        current = STORE.get_index_settings(pid)
        with ui.dialog() as dialog, ui.card().classes("settings-card w-96"):
            ui.label("Indexing settings").classes("font-bold text-green-800")
            ui.label(f"Project — {meta['name']}").classes("text-sm text-gray-500")

            ui.label("Chunking & descriptions — applying triggers a full reindex").classes(
                "text-xs font-medium text-green-700 mt-1"
            )
            chunk_size = ui.number(
                "Chunk size (chars)",
                value=current["chunk_size"], min=100, max=8000, step=50, precision=0,
            ).classes("w-full")
            chunk_overlap = ui.number(
                "Chunk overlap (chars)",
                value=current["chunk_overlap"], min=0, max=4000, step=10, precision=0,
            ).classes("w-full")
            descriptions = ui.switch(
                "LLM file descriptions (used for search routing)",
                value=current["descriptions_enabled"],
            )
            describe_chars = ui.number(
                "Description excerpt (chars sent to the LLM)",
                value=current["describe_max_chars"],
                min=500, max=50000, step=500, precision=0,
            ).classes("w-full")
            describe_chars.bind_enabled_from(descriptions, "value")

            def collect() -> dict:
                # Spread current so the project's search-time overrides (the
                # Retrieval settings) survive an index-time save untouched.
                return {
                    **current,
                    "chunk_size": int(chunk_size.value or current["chunk_size"]),
                    "chunk_overlap": int(chunk_overlap.value or 0),
                    "descriptions_enabled": bool(descriptions.value),
                    "describe_max_chars": int(
                        describe_chars.value or current["describe_max_chars"]
                    ),
                }

            def apply():
                vals = collect()
                if vals["chunk_overlap"] >= vals["chunk_size"]:
                    ui.notify(
                        "Chunk overlap must be smaller than chunk size",
                        color="negative",
                    )
                    return
                dialog.submit(vals)

            with ui.row().classes("w-full justify-end items-center"):
                ui.button("Close", on_click=lambda: dialog.submit(None)).props("flat")
                apply_btn = (
                    ui.button("Apply — full reindex", on_click=apply)
                    .props("color=red")
                )
            # Hidden until the first change — closing an untouched dialog is a
            # plain cancel; reverting every field hides it again. Every knob
            # here is index-time, so applying always rebuilds the index.
            apply_btn.set_visibility(False)

            def on_change(_=None):
                apply_btn.set_visibility(collect() != current)

            for el in (chunk_size, chunk_overlap, descriptions, describe_chars):
                el.on_value_change(on_change)

        result = await dialog
        if not result or result == current:
            return
        STORE.set_index_settings(pid, result)
        ui.notify("Settings saved — rebuilding the index…", color="positive")
        await trigger_reindex(pid)

    async def rag_process_settings_dialog(pid):
        meta = STORE.get(pid)
        if not meta:
            return
        current = STORE.get_index_settings(pid)
        with ui.dialog() as dialog, ui.card().classes("settings-card w-96"):
            ui.label("RAG process settings").classes("font-bold text-green-800")
            ui.label(f"Project — {meta['name']}").classes("text-sm text-gray-500")
            ui.label(
                "All apply from the next search — no reindex."
            ).classes("text-xs text-gray-500")

            ui.label("Search & neighbor stitching").classes(
                "text-xs font-medium text-green-700 mt-2"
            )
            search_top_k = ui.number(
                "k blocks (nearest chunks fetched per search)",
                value=current["search_top_k"], min=1, max=50, step=1, precision=0,
            ).classes("w-full")
            expand_padding = ui.number(
                "Stitch padding (chunks pulled around each hit)",
                value=current["expand_padding"], min=0, max=10, step=1, precision=0,
            ).classes("w-full")
            bridge_gap = ui.number(
                "Merge gap (max chunks between windows to bridge)",
                value=current["bridge_gap"], min=0, max=20, step=1, precision=0,
            ).classes("w-full")

            hybrid_search = ui.switch(
                "Hybrid search (BM25 + vector, fused via RRF)",
                value=current["hybrid_search_enabled"],
            )
            ui.label(
                "Recovers exact terms, names and abbreviations that pure "
                "vector similarity can miss. Needs a full-text index built at "
                "index time — for a project indexed before this feature (or "
                "with it previously off), run a full reindex to build it."
            ).classes("text-xs text-gray-600")

            ui.label("Reranking").classes(
                "text-xs font-medium text-green-700 mt-2"
            )
            ui.label(
                "LLM per-chunk relevance assessment — when enabled, the judge sees "
                "a per-search topic-hit trend («прирост по теме») powered by LLM "
                "relevance scores instead of keyword matching."
            ).classes("text-xs text-gray-600")

            reranking = ui.switch(
                "Enable LLM reranking (per-chunk relevance checks)",
                value=current["reranking_enabled"],
            )

            remove_irrelevant = ui.switch(
                "Remove chunks assessed as irrelevant from search results",
                value=current["reranking_remove_irrelevant"],
            )
            remove_irrelevant.bind_enabled_from(reranking, "value")

            def _on_reranking_change(e):
                # When reranking is turned off, force removal to False —
                # a disabled switch must not silently stay ON.
                if not e.value:
                    remove_irrelevant.set_value(False)

            reranking.on_value_change(_on_reranking_change)

            ui.label("Iteration budget").classes(
                "text-xs font-medium text-green-700 mt-2"
            )
            ui.label(
                "Maximum search-and-judge iterations — how many times the pipeline "
                "can loop back to search for missing information before giving up."
            ).classes("text-xs text-gray-600")
            max_iter = ui.number(
                "Max iterations",
                value=current["max_iterations"],
                min=1, max=10, step=1, precision=0,
            ).classes("w-24")

            def collect() -> dict:
                # Spread current so an index-time knob touched elsewhere isn't
                # reset here; this dialog only owns the search-time keys.
                return {
                    **current,
                    "search_top_k": int(
                        current["search_top_k"]
                        if search_top_k.value is None else search_top_k.value
                    ),
                    "expand_padding": int(
                        current["expand_padding"]
                        if expand_padding.value is None else expand_padding.value
                    ),
                    "bridge_gap": int(
                        current["bridge_gap"]
                        if bridge_gap.value is None else bridge_gap.value
                    ),
                    "max_iterations": int(max_iter.value or current["max_iterations"]),
                    "reranking_enabled": bool(reranking.value),
                    "reranking_remove_irrelevant": bool(remove_irrelevant.value),
                    "hybrid_search_enabled": bool(hybrid_search.value),
                }

            def apply():
                dialog.submit(collect())

            with ui.row().classes("w-full justify-end items-center"):
                ui.button("Close", on_click=lambda: dialog.submit(None)).props("flat")
                apply_btn = ui.button("Apply", on_click=apply)
            apply_btn.set_visibility(False)

            def on_change(_=None):
                vals = collect()
                apply_btn.set_visibility(vals != current)

            for el in (search_top_k, expand_padding, bridge_gap, hybrid_search,
                       max_iter, reranking, remove_irrelevant):
                el.on_value_change(on_change)

        result = await dialog
        if not result or result == current:
            return
        STORE.set_index_settings(pid, result)
        ui.notify("RAG process settings saved — applied from the next search", color="positive")

    async def parsed_text_dialog(pid, name):
        # Read straight from the project's LanceDB table — this is exactly the
        # text the retriever searches (extracted → cleaned → chunked at index
        # time), not a fresh re-parse of the source file.
        db_path = STORE.db_path(pid)
        descriptions = load_descriptions(db_path)
        table = table_for_file(
            name,
            descriptions=descriptions,
            # Sibling names let the resolver refuse a stem-colliding legacy
            # table instead of rendering another file's content.
            siblings=[f["name"] for f in STORE.list_files(pid)],
        )
        try:
            rows = await fetch_all_chunks(table, db_path) if table else []
        except Exception:
            rows = []
        if not rows:
            ui.notify(
                f"'{name}' has no viewable indexed text — indexing failed, is "
                "still running, extracted nothing, or the file's table can't "
                "be resolved unambiguously (a reindex fixes the latter)",
                color="warning",
            )
            return
        entry = descriptions.get(table, {})
        description = entry.get("description", "")
        # Prefer the overlap the chunks were ACTUALLY cut with (stored in the
        # sidecar at index time); legacy sidecars → current setting. 0 is a
        # legitimate stored value, so test for None, not falsiness.
        overlap = entry.get("chunk_overlap")
        if overlap is None:
            overlap = STORE.get_index_settings(pid)["chunk_overlap"]
        merged = merge_chunk_texts([r["text"] for r in rows], overlap)

        def _chunk_block(i: int, r: dict) -> str:
            seq = i if r["seq"] is None else r["seq"]
            return (
                "<div class='parsed-chunk'>"
                f"<div class='parsed-chunk-tag'>chunk {seq} · {len(r['text'])} chars</div>"
                f"<div class='parsed-text'>{html.escape(r['text'])}</div>"
                "</div>"
            )

        chunks_html = "".join(_chunk_block(i, r) for i, r in enumerate(rows))
        merged_html = f"<div class='parsed-text'>{html.escape(merged)}</div>"

        with ui.dialog() as dialog, ui.card().classes("settings-card parsed-card"):
            ui.label(f"Parsed text — {name}").classes("font-bold text-green-800")
            ui.label(
                f"Collection '{table}' · {len(rows)} chunks · {len(merged):,} chars"
            ).classes("text-sm text-gray-500")
            if description:
                ui.label(description).classes("text-xs text-gray-600")
            view = ui.toggle(
                {"chunks": "Chunks (as stored)", "merged": "Continuous text"},
                value="chunks",
            ).props("dense no-caps")
            with ui.scroll_area().classes("w-full grow"):
                content = ui.html(chunks_html)
            view.on_value_change(
                lambda e: content.set_content(
                    merged_html if e.value == "merged" else chunks_html
                )
            )
            with ui.row().classes("w-full justify-end"):
                ui.button("Close", on_click=lambda: dialog.submit(None)).props("flat")
        await dialog
        # A closed dialog is only hidden, never removed — for the other (tiny)
        # dialogs that's harmless, but here both HTML payloads are the whole
        # document, so drop the element instead of leaking one per preview.
        dialog.clear()
        dialog.delete()

    async def trigger_reindex(pid):
        # Show frozen UI immediately, then reindex, then unfreeze.
        runtime.set_status(pid, "reindexing")
        projects_list.refresh()
        files_panel.refresh()
        chat_panel.refresh()
        await reindex_project(pid)
        projects_list.refresh()
        files_panel.refresh()
        chat_panel.refresh()

    async def trigger_update(pid, added, removed):
        # Incremental variant of trigger_reindex — only the file delta is indexed.
        runtime.set_status(pid, "reindexing")
        projects_list.refresh()
        files_panel.refresh()
        chat_panel.refresh()
        await update_project_index(pid, added, removed)
        projects_list.refresh()
        files_panel.refresh()
        chat_panel.refresh()

    async def confirm_replace(name) -> bool:
        with ui.dialog() as dialog, ui.card():
            ui.label(f"File '{name}' already exists.").classes("font-bold")
            ui.label("Replace the existing file?").classes("text-sm text-gray-500")
            with ui.row():
                ui.button(
                    "Keep existing", on_click=lambda: dialog.submit(False)
                ).props("flat")
                ui.button(
                    "Replace", on_click=lambda: dialog.submit(True)
                ).props("color=primary")
        return bool(await dialog)

    # ── Edit-mode file staging (disk + reindex happen only on "Done") ──

    def enter_edit(pid):
        # Snapshot the current on-disk files as the starting staged set.
        staged = [
            {"name": f["name"], "origin": "disk", "orig_name": f["name"],
             "content": None, "deleted": False}
            for f in STORE.list_files(pid)
        ]
        ctx["edit"] = {"pid": pid, "files": staged}
        files_panel.refresh()

    def cancel_edit():
        # Discard all staged changes — disk untouched.
        ctx["edit"] = None
        files_panel.refresh()

    def _visible_names() -> set[str]:
        return {s["name"] for s in ctx["edit"]["files"] if not s["deleted"]}

    async def stage_upload(e: events.UploadEventArguments):
        name = Path(e.file.name).name
        if Path(name).suffix.lower() not in SUPPORTED_SUFFIXES:
            ui.notify(f"Unsupported file type: {name}", color="negative")
            return
        # Same-name within the staged set → ask before replacing.
        clash = next(
            (s for s in ctx["edit"]["files"] if not s["deleted"] and s["name"] == name),
            None,
        )
        if clash is not None:
            if not await confirm_replace(name):
                ui.notify(f"Kept existing {name}", color="warning")
                return
            clash["deleted"] = True
        content = await e.file.read()
        ctx["edit"]["files"].append(
            {"name": name, "origin": "new", "orig_name": None,
             "content": content, "deleted": False}
        )
        # Refresh only the list — NOT files_panel — so the ui.upload widget
        # survives and other files in this batch keep transferring.
        staged_files_list.refresh()

    async def stage_rename_dialog(entry):
        with ui.dialog() as dialog, ui.card():
            ui.label("Rename file").classes("font-bold")
            inp = ui.input("New filename", value=entry["name"]).props("autofocus")
            with ui.row():
                ui.button("Cancel", on_click=lambda: dialog.submit(None)).props("flat")
                ui.button("Save", on_click=lambda: dialog.submit(inp.value))
        result = await dialog
        if not result:
            return
        new_name = Path(result).name
        if new_name == entry["name"]:
            return
        if Path(new_name).suffix.lower() not in SUPPORTED_SUFFIXES:
            ui.notify(f"Unsupported file type: {new_name}", color="negative")
            return
        if new_name in _visible_names():
            ui.notify(f"A file named '{new_name}' already exists", color="negative")
            return
        entry["name"] = new_name
        staged_files_list.refresh()

    def stage_delete(entry):
        if entry["origin"] == "new":
            ctx["edit"]["files"].remove(entry)  # nothing on disk yet
        else:
            entry["deleted"] = True             # tombstone — removed on commit
        staged_files_list.refresh()

    async def commit_edit():
        pid = ctx["edit"]["pid"]
        # Compute the delta vs. disk and apply it directly via ProjectStore's
        # per-file operations (delete/rename/add) — only the files actually
        # changed are touched, unlike reading every untouched file into memory
        # and rewriting the whole directory (the old replace_all_files: a
        # failure partway through it lost every file, not just the changed
        # ones).
        added: list[str] = []              # new uploads, replacements, rename targets
        removed: list[str] = []            # deletions, rename sources
        deletions: list[str] = []          # orig_name to delete
        renames: list[tuple[str, str]] = []    # (orig_name, new_name)
        uploads: list[tuple[str, bytes]] = []  # (name, content)

        final_names: set[str] = set()
        for s in ctx["edit"]["files"]:
            if s["deleted"]:
                if s["origin"] == "disk":
                    deletions.append(s["orig_name"])
                    removed.append(s["orig_name"])
                continue
            name = s["name"]
            if name in final_names:
                ui.notify(f"Duplicate filename: {name}", color="negative")
                return
            final_names.add(name)
            if s["origin"] == "new":
                uploads.append((name, s["content"]))
                added.append(name)
            elif name != s["orig_name"]:  # renamed → new table name
                renames.append((s["orig_name"], name))
                removed.append(s["orig_name"])
                added.append(name)

        if not added and not removed:
            ctx["edit"] = None
            files_panel.refresh()
            ui.notify("No changes")
            return

        try:
            for orig_name in deletions:
                STORE.delete_file(pid, orig_name)
            # Renames go through a unique temporary name first: a direct
            # old->new rename silently overwrites `new` on a plain filesystem
            # rename, and `new` may itself be the source of ANOTHER rename in
            # this same batch that hasn't run yet (e.g. x.txt->y.txt staged
            # alongside y.txt->z.txt) — the temp hop makes the two-phase
            # application order-independent. Keep the original suffix so
            # rename_file's extension check (which only allows SUPPORTED_
            # SUFFIXES) still passes on the intermediate name.
            temp_targets: list[tuple[str, str]] = []
            for orig_name, new_name in renames:
                temp_name = f".__rename_tmp_{uuid.uuid4().hex}{Path(orig_name).suffix}"
                STORE.rename_file(pid, orig_name, temp_name)
                temp_targets.append((temp_name, new_name))
            for temp_name, new_name in temp_targets:
                STORE.rename_file(pid, temp_name, new_name)
            for name, content in uploads:
                STORE.add_file(pid, name, content)
        except (ValueError, OSError) as ex:
            ui.notify(str(ex), color="negative")
            return
        ctx["edit"] = None
        # Notify before trigger_update: it refreshes files_panel, which deletes
        # this handler's slot — any ui.* call after that would crash.
        ui.notify(
            f"Files saved — indexing {len(added)} changed file(s)…"
            if added else "Files saved — updating index…",
            color="positive",
        )
        await trigger_update(pid, added, removed)

    async def send_message(inp):
        pid = ctx["chat_pid"]
        if not pid or runtime.is_frozen(pid):
            return
        q = (inp.value or "").strip()
        if not q:
            return
        inp.value = ""
        ctx["messages"].append({"role": "user", "text": q})
        assistant = {"role": "assistant", "text": "", "trace": []}
        ctx["messages"].append(assistant)
        messages_view.refresh()
        async for kind, payload in run_chat(pid, q):
            if kind == "trace":
                assistant["trace"].append(payload)
            elif kind == "answer":
                assistant["text"] = payload
            messages_view.refresh()

    # ── layout ──

    with ui.row().classes("w-full h-screen no-wrap gap-0"):
        with ui.column().classes("app-sidebar w-1/3 h-full p-3 gap-2 overflow-auto"):
            ui.label("Projects").classes("text-xl font-bold text-green-800")
            ui.button("＋ New project", on_click=new_project).props("color=primary")
            projects_list()
            ui.separator()
            files_panel()
        with ui.column().classes("app-sidebar w-2/3 h-full p-3 items-center"):
            chat_panel()

    # Keep freeze visuals in sync with background reindexing, regardless of
    # navigation. Refresh only when the relevant projects' frozen state changes
    # (so it never interferes with live chat streaming).
    ctx["_frozen_snap"] = None

    def _sync_freeze():
        snap = (
            runtime.is_frozen(ctx["chat_pid"]) if ctx["chat_pid"] else False,
            runtime.is_frozen(ctx["open_pid"]) if ctx["open_pid"] else False,
            # per-file progress of the open project → refresh as each file finishes
            len(runtime.get_progress(ctx["open_pid"])) if ctx["open_pid"] else 0,
        )
        if snap != ctx["_frozen_snap"]:
            ctx["_frozen_snap"] = snap
            projects_list.refresh()
            files_panel.refresh()
            chat_panel.refresh()

    ui.timer(0.4, _sync_freeze)


def main():
    ui.run(root=index, title="Agentic RAG", port=8080, reload=False, show=False)


# NiceGUI requires this guard (covers its uvicorn subprocess).
if __name__ in {"__main__", "__mp_main__"}:
    main()
