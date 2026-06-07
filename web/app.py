"""NiceGUI web app for Agentic RAG.

Left column: projects (green theme) + files panel for the open project.
Right column: chat. Editing a project's files freezes the chat (blue + tremble)
while the project's vector DB reindexes.

Run: python -m web.app
"""

import html
from pathlib import Path

from nicegui import ui, events

from web import runtime
from web.chat import run_chat
from web.indexing import reindex_project

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

    # Per-client UI state.
    ctx = {"open_pid": None, "chat_pid": None, "messages": []}

    # ── refreshable views ──

    @ui.refreshable
    def projects_list():
        projects = STORE.list_projects()
        if not projects:
            ui.label("No projects yet").classes("text-gray-400 text-sm")
            return
        for p in projects:
            pid = p["id"]
            frozen = runtime.is_frozen(pid)
            active = pid == ctx["open_pid"]
            card_cls = "project-card w-full p-2" + (" active" if active else "")
            with ui.card().classes(card_cls):
                with ui.row().classes("w-full items-center no-wrap"):
                    (
                        ui.label(p["name"])
                        .classes("font-medium grow cursor-pointer")
                        .on("click", lambda _=None, x=pid: open_project(x))
                    )
                    with ui.button(icon="more_vert").props("flat dense round"):
                        with ui.menu():
                            ui.menu_item("Open", lambda x=pid: open_project(x))
                            ui.menu_item(
                                "Rename",
                                lambda x=pid, n=p["name"]: rename_project_dialog(x, n),
                            )
                            ui.menu_item(
                                "Delete", lambda x=pid: delete_project_dialog(x)
                            )
                btn = (
                    ui.button(
                        "Open in chat", on_click=lambda _=None, x=pid: open_in_chat(x)
                    )
                    .props("flat dense")
                    .classes("w-full")
                )
                if frozen:
                    btn.props("disable")
                    btn.classes("frozen")
                    ui.label("Reindexing…").classes("text-xs text-blue-600")

    @ui.refreshable
    def files_panel():
        pid = ctx["open_pid"]
        if not pid:
            return
        meta = STORE.get(pid)
        if not meta:
            return
        frozen = runtime.is_frozen(pid)
        ui.label(f"Files — {meta['name']}").classes("font-bold text-green-700")
        up = (
            ui.upload(
                on_upload=lambda e, x=pid: handle_upload(x, e),
                auto_upload=True,
                multiple=True,
            )
            .props(f'accept="{ACCEPT}"')
            .classes("w-full")
        )
        if frozen:
            up.disable()
        files = STORE.list_files(pid)
        if not files:
            ui.label("No files uploaded").classes("text-gray-400 text-sm")
            return
        for f in files:
            with ui.row().classes("w-full items-center no-wrap"):
                ui.icon("description").classes("text-green-600")
                ui.label(f"{f['name']}  ({f['size']} B)").classes("grow text-sm")
                rn = ui.button(
                    icon="edit",
                    on_click=lambda _=None, x=pid, n=f["name"]: rename_file_dialog(x, n),
                ).props("flat dense round")
                dl = ui.button(
                    icon="delete",
                    on_click=lambda _=None, x=pid, n=f["name"]: delete_file_action(x, n),
                ).props("flat dense round color=red")
                if frozen:
                    rn.disable()
                    dl.disable()

    @ui.refreshable
    def messages_view():
        for m in ctx["messages"]:
            render_message(m)

    @ui.refreshable
    def chat_panel():
        pid = ctx["chat_pid"]
        frozen = bool(pid) and runtime.is_frozen(pid)
        base = "w-full h-full flex flex-col gap-2 p-3 rounded-lg border"
        with ui.column().classes(base + (" frozen" if frozen else "")):
            if frozen:
                ui.html(_snowflakes_html()).classes("snowflakes")
            if not pid:
                ui.label("Open a project in chat to start").classes("text-gray-400")
                return
            meta = STORE.get(pid)
            ui.label(f"Chat — {meta['name'] if meta else pid}").classes(
                "text-lg font-bold text-green-700"
            )
            if frozen:
                with ui.row().classes("items-center gap-2"):
                    ui.spinner(size="sm")
                    ui.label("Reindexing… chat frozen").classes("text-blue-700")
            with ui.scroll_area().classes("grow w-full"):
                with ui.column().classes("w-full gap-2"):
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
            with ui.column().classes("w-full gap-1"):
                for t in m.get("trace", []):
                    agent = html.escape(str(t.get("agent", "")))
                    decision = html.escape(str(t.get("decision", "")))
                    ui.html(
                        f"<div class='trace-step'>🔵 {agent}: {decision}</div>"
                    )
                if m.get("text"):
                    ui.markdown(m["text"]).classes("chat-bubble-assistant")
                elif not m.get("trace"):
                    ui.spinner(size="sm")

    # ── actions ──

    def open_project(pid):
        ctx["open_pid"] = pid
        projects_list.refresh()
        files_panel.refresh()

    def open_in_chat(pid):
        if runtime.is_frozen(pid):
            ui.notify("Project is reindexing — chat is frozen", color="warning")
            return
        if ctx["chat_pid"] != pid:
            ctx["chat_pid"] = pid
            ctx["messages"] = []
        chat_panel.refresh()

    async def new_project():
        with ui.dialog() as dialog, ui.card():
            ui.label("New project").classes("font-bold")
            name = ui.input("Project name").props("autofocus")
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
            if ctx["open_pid"] == pid:
                ctx["open_pid"] = None
            if ctx["chat_pid"] == pid:
                ctx["chat_pid"] = None
                ctx["messages"] = []
            projects_list.refresh()
            files_panel.refresh()
            chat_panel.refresh()

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

    async def handle_upload(pid, e: events.UploadEventArguments):
        # NiceGUI 3.x: e.file is a FileUpload; .name + async .read().
        name = e.file.name
        # Same-name upload → ask before overwriting; "no" = don't save.
        if STORE.file_exists(pid, name):
            if not await confirm_replace(name):
                ui.notify(f"Kept existing {name}", color="warning")
                return
        try:
            content = await e.file.read()
            STORE.add_file(pid, name, content)
        except ValueError as ex:
            ui.notify(str(ex), color="negative")
            return
        await trigger_reindex(pid)
        ui.notify(f"Indexed {name}", color="positive")

    async def rename_file_dialog(pid, old):
        with ui.dialog() as dialog, ui.card():
            ui.label("Rename file").classes("font-bold")
            name = ui.input("New filename", value=old).props("autofocus")
            with ui.row():
                ui.button("Cancel", on_click=lambda: dialog.submit(None)).props("flat")
                ui.button("Save", on_click=lambda: dialog.submit(name.value))
        result = await dialog
        if result and result != old:
            try:
                STORE.rename_file(pid, old, result)
            except (ValueError, OSError) as ex:
                ui.notify(str(ex), color="negative")
                return
            await trigger_reindex(pid)

    async def delete_file_action(pid, name):
        STORE.delete_file(pid, name)
        await trigger_reindex(pid)

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
        with ui.column().classes("w-1/3 h-full p-3 gap-2 overflow-auto").style(
            "background:#fafafa"
        ):
            ui.label("Projects").classes("text-xl font-bold text-green-700")
            ui.button("＋ New project", on_click=new_project).props("color=primary")
            projects_list()
            ui.separator()
            files_panel()
        with ui.column().classes("w-2/3 h-full p-3"):
            chat_panel()


def main():
    ui.run(root=index, title="Agentic RAG", port=8080, reload=False, show=False)


# NiceGUI requires this guard (covers its uvicorn subprocess).
if __name__ in {"__main__", "__mp_main__"}:
    main()
