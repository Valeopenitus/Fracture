from __future__ import annotations

import os
import re
import json
import time
import shutil
import difflib
import py_compile
import subprocess
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from tkinter import font as tkfont


def pick_font(root, candidates, size=10, weight="normal"):
    """
    Returns a tk Font using the first installed family found in `candidates`.
    Falls back to TkDefaultFont if nothing matches.
    """
    available = set(tkfont.families(root))

    for fam in candidates:
        if fam in available:
            return tkfont.Font(root=root, family=fam, size=size, weight=weight)

    # Safe default
    return tkfont.nametofont("TkDefaultFont").copy()

# --- Canonical 3-line section header format ---
# 1) divider line: starts with # then spaces then >= min_len of same char (e.g. =====)
# 2) section line: "# SECTION: <CATEGORY> :: <NAME> [FLAGS]"
# 3) divider line again (same as line 1)

TOOL_NAME = "fracture"
TOOL_VERSION = "0.1.0-stable"

MIN_DIVIDER_LEN = 20
ALLOWED_FLAGS = {"PROTECTED", "INTERNAL", "EXPERIMENTAL", "DEPRECATED"}

# --- Optional patterns for future auto-split features (safe globals) ---
_TOP_DEFCLASS = re.compile(r"^(def|class)\s+([A-Za-z_][A-Za-z0-9_]*)\b")


# ==================================================================
# SECTION: CORE :: Paths and directories [PROTECTED]
# ==================================================================

def tool_root() -> str:
    return os.path.dirname(os.path.abspath(__file__))


def ensure_dirs() -> dict:
    root = tool_root()
    data_dir = os.path.join(root, "data")
    backups_dir = os.path.join(root, "backups")
    logs_dir = os.path.join(root, "logs")

    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(backups_dir, exist_ok=True)
    os.makedirs(logs_dir, exist_ok=True)

    paths = {
        "root": root,
        "data_dir": data_dir,
        "backups_dir": backups_dir,
        "logs_dir": logs_dir,
        "settings_path": os.path.join(data_dir, "settings.json"),
        "recent_path": os.path.join(data_dir, "recent_files.json"),
        "ledger_path": os.path.join(data_dir, "patch_ledger.jsonl"),
    }

    if not os.path.exists(paths["ledger_path"]):
        with open(paths["ledger_path"], "a", encoding="utf-8") as _:
            pass

    return paths


# ==================================================================
# SECTION: CORE :: Settings, recents, and ledger I/O
# ==================================================================

def load_json(path: str, default):
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return default


def save_json(path: str, obj) -> None:
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
    os.replace(tmp, path)


def append_ledger(ledger_path: str, event: dict) -> None:
    line = json.dumps(event, ensure_ascii=False)
    with open(ledger_path, "a", encoding="utf-8") as f:
        f.write(line + "\n")
        f.flush()
        os.fsync(f.fileno())


def iter_ledger_reverse(ledger_path: str):
    try:
        with open(ledger_path, "r", encoding="utf-8") as f:
            lines = f.read().splitlines()
        for line in reversed(lines):
            if not line.strip():
                continue
            try:
                yield json.loads(line)
            except Exception:
                continue
    except Exception:
        return


def get_last_successful_patch_event(ledger_path: str) -> dict | None:
    for ev in iter_ledger_reverse(ledger_path):
        if ev.get("action") == "apply_patch" and ev.get("status") == "success":
            backup_path = (ev.get("backup") or {}).get("backup_path")
            target_path = (ev.get("target") or {}).get("file_path")
            if backup_path and os.path.exists(backup_path) and target_path:
                return ev
    return None


# ==================================================================
# SECTION: FRACTURE :: Section detection and slicing
# ==================================================================

class SectionInfo:
    def __init__(
        self,
        category: str,
        name: str,
        flags: set[str],
        start_line: int,
        header_line: int,
        body_start_line: int
    ):
        self.category = category
        self.name = name
        self.flags = flags
        self.start_line = start_line          # 1-based
        self.header_line = header_line        # 1-based
        self.body_start_line = body_start_line  # 1-based
        self.protected = "PROTECTED" in flags

    @property
    def identity(self) -> str:
        return f"{self.category} :: {self.name}"


def _is_divider_line(line: str) -> tuple[bool, str]:
    s = line.rstrip("\n")
    if not s.lstrip().startswith("#"):
        return (False, "")
    after_hash = s.lstrip()[1:].strip()
    if len(after_hash) < MIN_DIVIDER_LEN:
        return (False, "")
    ch = after_hash[0]
    if ch not in "=~-_":
        return (False, "")
    if all(c == ch for c in after_hash):
        return (True, ch)
    return (False, "")


def _parse_section_line(line: str) -> tuple[str, str, set[str]] | None:
    s = line.strip()
    if not s.startswith("#"):
        return None
    s = s[1:].strip()
    if not s.startswith("SECTION:"):
        return None

    rest = s[len("SECTION:"):].strip()
    flags: set[str] = set()

    if rest.endswith("]") and "[" in rest:
        left = rest.rfind("[")
        flags_text = rest[left + 1:-1]
        rest = rest[:left].rstrip()
        for tok in flags_text.split(","):
            t = tok.strip().upper()
            if t:
                flags.add(t)

    if " :: " not in rest:
        return None

    category, name = rest.split(" :: ", 1)
    category = category.strip()
    name = name.strip()
    if not category or not name:
        return None

    return category, name, flags


def detect_sections(text: str) -> tuple[list[SectionInfo], list[str]]:
    errors: list[str] = []
    lines = text.splitlines(keepends=True)
    sections: list[SectionInfo] = []

    i = 0
    n = len(lines)

    while i < n - 2:
        is_div1, ch1 = _is_divider_line(lines[i])
        if not is_div1:
            i += 1
            continue

        parsed = _parse_section_line(lines[i + 1])
        is_div2, ch2 = _is_divider_line(lines[i + 2])

        if parsed and is_div2 and ch1 == ch2:
            category, name, flags = parsed
            start_line = i + 1
            header_line = i + 2
            body_start_line = i + 4  # 1-based
            sections.append(
                SectionInfo(category, name, flags, start_line, header_line, body_start_line)
            )
            i += 3
        else:
            i += 1

    seen: dict[str, int] = {}
    for s in sections:
        seen[s.identity] = seen.get(s.identity, 0) + 1
    dups = [k for k, v in seen.items() if v > 1]
    if dups:
        errors.append("Duplicate section identities found: " + ", ".join(dups))

    return sections, errors


def compute_section_body_range(
    lines: list[str],
    sections_sorted: list[SectionInfo],
    selected: SectionInfo
) -> tuple[int, int, int, int]:
    ordered = sorted(sections_sorted, key=lambda s: s.start_line)

    idx = None
    for i, s in enumerate(ordered):
        if s is selected:
            idx = i
            break
    if idx is None:
        raise ValueError("Selected section not found.")

    body_start_line_1b = selected.body_start_line
    start_idx = max(0, body_start_line_1b - 1)

    if idx + 1 < len(ordered):
        next_sec = ordered[idx + 1]
        body_end_line_1b = next_sec.start_line - 1
        end_idx = max(start_idx, body_end_line_1b)
    else:
        body_end_line_1b = len(lines)
        end_idx = len(lines)

    return start_idx, end_idx, body_start_line_1b, body_end_line_1b


def _find_top_level_blocks(lines: list[str]) -> list[dict]:
    def is_real_code(l: str) -> bool:
        s = l.strip()
        return bool(s) and not s.startswith("#")

    starts0 = []
    for i, line in enumerate(lines):
        m = _TOP_DEFCLASS.match(line)
        if m:
            starts0.append((i, m.group(1), m.group(2)))

    if starts0:
        starts = starts0
    else:
        base_indent = None
        for l in lines:
            if is_real_code(l):
                base_indent = l[:len(l) - len(l.lstrip())]
                break
        if base_indent is None:
            return []

        rx = re.compile(rf"^{re.escape(base_indent)}(def|class)\s+([A-Za-z_]\w*)")
        starts = []
        for i, line in enumerate(lines):
            m = rx.match(line)
            if m:
                starts.append((i, m.group(1), m.group(2)))

    blocks = []
    for idx, (start_i, kind, name) in enumerate(starts):
        end_i = starts[idx + 1][0] if idx + 1 < len(starts) else len(lines)
        blocks.append({
            "kind": kind,
            "name": name,
            "start": start_i,
            "end": end_i,
            "lines": lines[start_i:end_i],
        })

    return blocks


# ==================================================================
# SECTION: FRACTURE :: Improved auto-split [PROTECTED]
# ==================================================================

def auto_split_section_body(
    *,
    category: str,
    section_name: str,
    body_text: str,
    min_chunk_lines: int = 20,
    max_sections: int = 8,
) -> tuple[str, int]:
    lines = body_text.splitlines(keepends=True)

    blocks = _find_top_level_blocks(lines)
    if not blocks:
        return body_text, 0

    out: list[str] = []
    used_titles: set[str] = set()
    created = 0

    def uniq(base: str) -> str:
        title = base
        i = 2
        while title in used_titles:
            title = f"{base} #{i}"
            i += 1
        used_titles.add(title)
        return title

    def emit(title: str, chunk_lines: list[str]) -> None:
        nonlocal created
        if created >= max_sections:
            return
        created += 1
        if chunk_lines:
            indent = chunk_lines[0][:len(chunk_lines[0]) - len(chunk_lines[0].lstrip())]
        else:
            indent = ""
        out.append(f"# {'=' * 70}\n# SECTION: {category} :: {uniq(title)}\n# {'=' * 70}\n")
        out.extend(chunk_lines)
        out.append("\n")

    prelude_lines = lines[:blocks[0]["start"]] if blocks else lines
    if any(line.strip() for line in prelude_lines):
        emit(f"{section_name} — prelude", prelude_lines)

    for block in blocks:
        title = f"{block['kind']} {block['name']}"
        emit(title, block["lines"])

        if created >= max_sections:
            remaining = lines[block["end"]:]
            out.extend(remaining)
            break

    if created < max_sections and blocks:
        trailing = lines[blocks[-1]["end"]:]
        if any(line.strip() for line in trailing):
            emit(f"{section_name} — trailing", trailing)


def auto_format_file_text_into_sections(
    *,
    file_path: str,
    text: str,
    category: str = "FILE",
    max_sections: int = 32,
) -> tuple[str, int]:
    """
    Formats a *single file* by INSERTING SECTION HEADERS into the same file text.

    This does NOT split into multiple files.
    It creates section boundaries based on top-level def/class blocks when possible.
    """
    lines = text.splitlines(keepends=True)
    blocks = _find_top_level_blocks(lines)
    if not blocks:
        return text, 0

    out: list[str] = []
    used_titles: set[str] = set()
    created = 0

    def uniq(base: str) -> str:
        title = base
        i = 2
        while title in used_titles:
            title = f"{base} #{i}"
            i += 1
        used_titles.add(title)
        return title

    def emit(title: str, chunk_lines: list[str]) -> None:
        nonlocal created
        if created >= max_sections:
            return
        created += 1
        out.append(f"# {'=' * 70}\n# SECTION: {category} :: {uniq(title)}\n# {'=' * 70}\n")
        out.extend(chunk_lines)
        if out and (not out[-1].endswith("\n")):
            out[-1] = out[-1] + "\n"
        out.append("\n")

    base_name = os.path.basename(file_path) if file_path else "file"

    prelude_lines = lines[:blocks[0]["start"]]
    if any(line.strip() for line in prelude_lines):
        emit(f"{base_name} — prelude", prelude_lines)

    for block in blocks:
        title = f"{block['kind']} {block['name']}"
        emit(title, block["lines"])
        if created >= max_sections:
            # append remaining without adding more sections
            out.extend(lines[block["end"]:])
            break

    if created < max_sections:
        trailing = lines[blocks[-1]["end"]:]
        if any(line.strip() for line in trailing):
            emit(f"{base_name} — trailing", trailing)

    return "".join(out), created


def apply_format_whole_file(
    *,
    file_path: str,
    new_text: str,
    paths: dict,
    run_py_compile: bool,
    activity_log_cb=None
) -> tuple[bool, str, str | None]:
    """
    Writes a fully formatted file (with SECTION headers inserted) with:
    - automatic backup
    - atomic replace
    - section post-validation
    - optional py_compile check
    """
    def log(msg: str):
        if activity_log_cb:
            activity_log_cb(msg)

    if not os.path.exists(file_path):
        return False, f"File not found: {file_path}", None

    backup_path = None
    event = {
        "ledger_version": "1.0",
        "event_id": f"evt_{time.strftime('%Y%m%d_%H%M%S')}_{os.getpid()}",
        "timestamp": time.time(),
        "action": "format_file",
        "status": "failed",
        "tool": {"name": TOOL_NAME, "tool_version": TOOL_VERSION},
        "target": {"file_path": file_path, "mode": "insert_section_headers"},
        "backup": {"created": False, "backup_path": None},
        "checks": {"post_validate_ok": None, "py_compile_ok": None},
        "message": ""
    }

    try:
        log("Reading file...")
        old_text = open(file_path, "r", encoding="utf-8").read()

        log("Creating backup...")
        backup_path = make_backup(paths["backups_dir"], file_path, "auto_format_whole_file")
        event["backup"]["created"] = True
        event["backup"]["backup_path"] = backup_path

        log("Writing formatted file (atomic replace)...")
        atomic_write_text(file_path, new_text)

        log("Post-validating sections...")
        post_text = open(file_path, "r", encoding="utf-8").read()
        _, post_errors = detect_sections(post_text)
        event["checks"]["post_validate_ok"] = (len(post_errors) == 0)

        if post_errors:
            log("Post-validate failed; rolling back...")
            rollback_from_backup(file_path, backup_path)
            event["status"] = "rolled_back"
            event["message"] = "Post-validation failed; rolled back. Errors: " + " | ".join(post_errors)
            append_ledger(paths["ledger_path"], event)
            return False, event["message"], backup_path

        if run_py_compile and file_path.lower().endswith(".py"):
            log("Python compile check...")
            try:
                py_compile.compile(file_path, doraise=True)
                event["checks"]["py_compile_ok"] = True
            except Exception as e:
                log("Compile failed; rolling back...")
                rollback_from_backup(file_path, backup_path)
                event["status"] = "rolled_back"
                event["checks"]["py_compile_ok"] = False
                event["message"] = f"Python compile failed; rolled back. {e}"
                append_ledger(paths["ledger_path"], event)
                return False, event["message"], backup_path

        event["status"] = "success"
        event["message"] = "File formatted successfully (SECTION headers inserted)."
        append_ledger(paths["ledger_path"], event)
        return True, event["message"], backup_path

    except Exception as e:
        try:
            if backup_path and os.path.exists(backup_path):
                rollback_from_backup(file_path, backup_path)
                event["status"] = "rolled_back"
                event["message"] = f"Exception occurred; rolled back. {e}"
            else:
                event["status"] = "failed"
                event["message"] = f"Exception occurred; no rollback available. {e}"
        finally:
            try:
                append_ledger(paths["ledger_path"], event)
            except Exception:
                pass
        return False, event["message"], backup_path


    return "".join(out), created


# ==================================================================
# SECTION: SAFETY :: Atomic write and backups [PROTECTED]
# ==================================================================

def atomic_write_text(target_path: str, new_text: str) -> None:
    if new_text and not new_text.endswith("\n"):
        new_text += "\n"

    target_dir = os.path.dirname(os.path.abspath(target_path))
    base = os.path.basename(target_path)

    pid = os.getpid()
    ts = time.strftime("%Y%m%d_%H%M%S")
    tmp_name = f"{base}.fracture_tmp_{pid}_{ts}"
    tmp_path = os.path.join(target_dir, tmp_name)

    with open(tmp_path, "w", encoding="utf-8", newline="\n") as f:
        f.write(new_text)
        f.flush()
        os.fsync(f.fileno())

    os.replace(tmp_path, target_path)


def make_backup(backups_dir: str, target_path: str, section_identity: str) -> str:
    os.makedirs(backups_dir, exist_ok=True)
    ts = time.strftime("%Y%m%d_%H%M%S")
    safe_file = target_path.replace(":", "").replace("\\", "_").replace("/", "_")
    safe_sec = section_identity.replace(" :: ", "__").replace(" ", "_")
    backup_name = f"{safe_file}__{safe_sec}__{ts}.bak"
    backup_path = os.path.join(backups_dir, backup_name)
    shutil.copy2(target_path, backup_path)
    return os.path.abspath(backup_path)


def rollback_from_backup(target_path: str, backup_path: str) -> None:
    target_dir = os.path.dirname(os.path.abspath(target_path))
    base = os.path.basename(target_path)
    pid = os.getpid()
    ts = time.strftime("%Y%m%d_%H%M%S")
    tmp_path = os.path.join(target_dir, f"{base}.fracture_rollback_tmp_{pid}_{ts}")

    shutil.copy2(backup_path, tmp_path)
    os.replace(tmp_path, target_path)


# ==================================================================
# SECTION: PATCHER :: Apply patch and undo [PROTECTED]
# ==================================================================

def preview_diff(old_text: str, new_text: str, path: str) -> str:
    old_lines = old_text.splitlines(keepends=True)
    new_lines = new_text.splitlines(keepends=True)
    return "".join(difflib.unified_diff(old_lines, new_lines, fromfile=path, tofile=path))


def apply_patch_replace_body(
    *,
    file_path: str,
    section_identity: str,
    replacement_body: str,
    paths: dict,
    run_py_compile: bool,
    activity_log_cb=None
) -> tuple[bool, str, str | None]:
    def log(msg: str):
        if activity_log_cb:
            activity_log_cb(msg)

    if not os.path.exists(file_path):
        return False, f"File not found: {file_path}", None

    backup_path = None
    event = {
        "ledger_version": "1.0",
        "event_id": f"evt_{time.strftime('%Y%m%d_%H%M%S')}_{os.getpid()}",
        "timestamp": time.time(),
        "action": "apply_patch",
        "status": "failed",
        "tool": {"name": TOOL_NAME, "tool_version": TOOL_VERSION},
        "target": {"file_path": file_path, "section_identity": section_identity, "mode": "replace_body"},
        "backup": {"created": False, "backup_path": None},
        "checks": {"pre_validate_ok": None, "post_validate_ok": None, "py_compile_ok": None},
        "message": ""
    }

    try:
        log("Reading file...")
        old_text = open(file_path, "r", encoding="utf-8").read()

        log("Detecting sections...")
        sections, errors = detect_sections(old_text)
        event["checks"]["pre_validate_ok"] = (len(errors) == 0)

        if errors:
            msg = "Pre-validate failed: " + " | ".join(errors)
            event["message"] = msg
            append_ledger(paths["ledger_path"], event)
            return False, msg, None

        matches = [s for s in sections if s.identity == section_identity]
        if len(matches) != 1:
            msg = f"Section selection failed (found {len(matches)} matches): {section_identity}"
            event["message"] = msg
            append_ledger(paths["ledger_path"], event)
            return False, msg, None

        section = matches[0]
        if section.protected:
            msg = f"Protected section; patch refused: {section_identity}"
            event["message"] = msg
            append_ledger(paths["ledger_path"], event)
            return False, msg, None

        lines = old_text.splitlines(keepends=True)
        start_idx, end_idx, body_start_line, body_end_line = compute_section_body_range(lines, sections, section)

        rep = replacement_body
        if rep and not rep.endswith("\n"):
            rep += "\n"
        rep_lines = rep.splitlines(keepends=True)

        new_lines = lines[:start_idx] + rep_lines + lines[end_idx:]
        new_text = "".join(new_lines)
        if new_text and not new_text.endswith("\n"):
            new_text += "\n"

        log("Creating backup...")
        backup_path = make_backup(paths["backups_dir"], file_path, section_identity)
        event["backup"]["created"] = True
        event["backup"]["backup_path"] = backup_path

        log("Writing patched file (atomic replace)...")
        atomic_write_text(file_path, new_text)

        log("Post-validating sections...")
        post_text = open(file_path, "r", encoding="utf-8").read()
        _, post_errors = detect_sections(post_text)
        event["checks"]["post_validate_ok"] = (len(post_errors) == 0)

        if post_errors:
            log("Post-validate failed; rolling back...")
            rollback_from_backup(file_path, backup_path)
            event["status"] = "rolled_back"
            event["message"] = "Post-validation failed; rolled back. Errors: " + " | ".join(post_errors)
            append_ledger(paths["ledger_path"], event)
            return False, event["message"], backup_path

        if run_py_compile and file_path.lower().endswith(".py"):
            log("Python compile check...")
            try:
                py_compile.compile(file_path, doraise=True)
                event["checks"]["py_compile_ok"] = True
            except Exception as e:
                log("Compile failed; rolling back...")
                rollback_from_backup(file_path, backup_path)
                event["status"] = "rolled_back"
                event["checks"]["py_compile_ok"] = False
                event["message"] = f"Python compile failed; rolled back. {e}"
                append_ledger(paths["ledger_path"], event)
                return False, event["message"], backup_path

        event["status"] = "success"
        event["message"] = f"Patch applied successfully. (Body lines {body_start_line}–{body_end_line})"
        append_ledger(paths["ledger_path"], event)
        return True, event["message"], backup_path

    except Exception as e:
        try:
            if backup_path and os.path.exists(backup_path):
                rollback_from_backup(file_path, backup_path)
                event["status"] = "rolled_back"
                event["message"] = f"Exception occurred; rolled back. {e}"
            else:
                event["status"] = "failed"
                event["message"] = f"Exception occurred; no rollback available. {e}"
        finally:
            try:
                append_ledger(paths["ledger_path"], event)
            except Exception:
                pass
        return False, event["message"], backup_path


def undo_last_patch(paths: dict, activity_log_cb=None) -> tuple[bool, str]:
    def log(msg: str):
        if activity_log_cb:
            activity_log_cb(msg)

    ev = get_last_successful_patch_event(paths["ledger_path"])
    if not ev:
        return False, "Nothing to undo."

    target_path = (ev.get("target") or {}).get("file_path")
    section_identity = (ev.get("target") or {}).get("section_identity")
    backup_path = (ev.get("backup") or {}).get("backup_path")

    if not backup_path or not os.path.exists(backup_path):
        return False, f"Undo unavailable: backup missing: {backup_path}"

    if not target_path:
        return False, "Undo unavailable: target_path missing in ledger."

    try:
        log("Undo: restoring backup (atomic replace)...")
        rollback_from_backup(target_path, backup_path)

        roll_event = {
            "ledger_version": "1.0",
            "event_id": f"evt_{time.strftime('%Y%m%d_%H%M%S')}_{os.getpid()}",
            "timestamp": time.time(),
            "action": "rollback",
            "status": "success",
            "tool": {"name": TOOL_NAME, "tool_version": TOOL_VERSION},
            "target": {"file_path": target_path, "section_identity": section_identity, "mode": "replace_body"},
            "backup": {"backup_path": backup_path},
            "message": "Rollback completed."
        }
        append_ledger(paths["ledger_path"], roll_event)
        return True, f"Rollback completed: {os.path.basename(target_path)}"
    except Exception as e:
        return False, f"Rollback failed: {e}"


# ==================================================================
# SECTION: CORE :: Recent files store [PROTECTED]
# ==================================================================

def load_recents(recent_path: str) -> list[dict]:
    data = load_json(recent_path, {"recent_version": "1.0", "items": []})
    items = data.get("items", [])
    cleaned = []
    seen = set()
    for it in items:
        p = it.get("path")
        if not p or p in seen:
            continue
        seen.add(p)
        it.setdefault("display_name", os.path.basename(p))
        it.setdefault("pinned", False)
        it.setdefault("last_opened_ts", 0.0)
        it.setdefault("last_patched_ts", 0.0)
        it["exists_last_check"] = os.path.exists(p)
        cleaned.append(it)
    return cleaned


def save_recents(recent_path: str, items: list[dict]) -> None:
    save_json(recent_path, {"recent_version": "1.0", "items": items})


def touch_recent(items: list[dict], path: str, patched: bool = False) -> list[dict]:
    now = time.time()
    found = None
    for it in items:
        if it["path"] == path:
            found = it
            break

    if not found:
        found = {
            "path": path,
            "display_name": os.path.basename(path),
            "pinned": False,
            "last_opened_ts": now,
            "last_patched_ts": 0.0,
            "exists_last_check": os.path.exists(path)
        }
        items.insert(0, found)
    else:
        found["last_opened_ts"] = now
        found["exists_last_check"] = os.path.exists(path)
        items.remove(found)
        items.insert(0, found)

    if patched:
        found["last_patched_ts"] = now

    return items


# ==================================================================
# SECTION: GUI :: Tkinter interface
# ==================================================================


# ==================================================================
# SECTION: GUI :: Context Menu (Copy/Paste/Select All) 
# ==================================================================

def install_context_menu(widget: tk.Widget) -> None:
    """
    Adds a right-click context menu (Cut/Copy/Paste/Select All) to a Tk/ttk widget.
    Supports: tk.Entry, ttk.Entry, tk.Text, ttk.Combobox (editable part).
    """

    menu = tk.Menu(widget, tearoff=0)
    try:
        menu.configure(
            bg="#202020",
            fg="#e6e6e6",
            activebackground="#343434",
            activeforeground="#e6e6e6",
            bd=0,
            relief="flat",
        )
    except Exception:
        pass
    menu.add_command(label="Cut", command=lambda: _ctx_cut(widget))
    menu.add_command(label="Copy", command=lambda: _ctx_copy(widget))
    menu.add_command(label="Paste", command=lambda: _ctx_paste(widget))
    menu.add_separator()
    menu.add_command(label="Select All", command=lambda: _ctx_select_all(widget))

    def popup(event):
        try:
            menu.tk_popup(event.x_root, event.y_root)
        finally:
            menu.grab_release()

    # Windows/Linux right click
    widget.bind("<Button-3>", popup, add="+")
    # macOS often uses Control-Click
    widget.bind("<Control-Button-1>", popup, add="+")  # Fixed: import sys was missing, but assuming it's there or not needed


def _ctx_cut(w: tk.Widget) -> None:
    try:
        w.event_generate("<<Cut>>")
    except Exception:
        pass

def _ctx_copy(w: tk.Widget) -> None:
    try:
        w.event_generate("<<Copy>>")
    except Exception:
        pass

def _ctx_paste(w: tk.Widget) -> None:
    try:
        w.event_generate("<<Paste>>")
    except Exception:
        pass

def _ctx_select_all(w: tk.Widget) -> None:
    # Text widgets
    if isinstance(w, tk.Text):
        w.tag_add("sel", "1.0", "end-1c")
        w.mark_set("insert", "1.0")
        w.see("insert")
        return

    # Entry-like widgets (Entry, ttk.Entry, Combobox)
    try:
        w.selection_range(0, "end")
        w.icursor("end")
    except Exception:
        pass

class FractureApp(tk.Tk):
    
    def __init__(self, paths: dict):
        super().__init__()
        self.paths = paths

        self.init_fonts()
        self._init_styles()

        self.settings = load_json(self.paths["settings_path"], {
            "settings_version": "1.0",
            "safety": {
                "first_patch_warning_dismissed": False,
                "py_compile_after_patch_default": True
            },
            "behavior": {"recent_limit": 10}
        })

        self._build_ui()
        self._refresh_recent_dropdown()
        self.log("Ready.")

        self.title(f"Fracture — {TOOL_VERSION}")
        self.geometry("1100x750")

        self.file_path: str | None = None
        self.sections: list[SectionInfo] = []
        self.section_map: dict[str, SectionInfo] = {}


# ==================================================
# SECTION: GUI :: Fonts
# ==================================================
    def init_fonts(self):
        self.ui_font = pick_font(
            self,
            ["Segoe UI Variable", "Segoe UI", "Tahoma", "Arial"],
            size=10
        )

        self.ui_font_bold = tkfont.Font(
            root=self,
            family=self.ui_font.actual("family"),
            size=9,
            weight="bold"
        )

        self.mono_font = pick_font(
            self,
            ["Cascadia Mono", "Consolas", "Cascadia Code", "Courier New"],
            size=10
        )

# ==================================================
# SECTION: GUI :: ttk Styles
# ==================================================
    def _init_styles(self):
        # Dark palette (match Barbalo Server Manager vibe)
        self.colors = {
            "bg": "#111111",
            "panel": "#1a1a1a",
            "panel2": "#202020",
            "border": "#2d2d2d",
            "text": "#e6e6e6",
            "muted": "#a9a9a9",
            "btn": "#2a2a2a",
            "btn_hover": "#343434",
            "select_bg": "#2b3b55",
            "select_fg": "#ffffff",
        }

        self.style = ttk.Style(self)
        self.recents = load_recents(self.paths["recent_path"])

        # Use a theme we can recolor reliably
        try:
            self.style.theme_use("clam")
        except Exception:
            pass

        # Window background
        self.configure(bg=self.colors["bg"])

        # ttk base
        self.style.configure(".", background=self.colors["bg"], foreground=self.colors["text"])
        self.style.configure("TFrame", background=self.colors["bg"])
        self.style.configure("TLabel", background=self.colors["bg"], foreground=self.colors["text"])

        # Group boxes / section headers
        self.style.configure("TLabelframe", background=self.colors["bg"])
        self.style.configure(
            "TLabelframe.Label",
            background=self.colors["bg"],
            foreground=self.colors["muted"],
            font=self.ui_font_bold,
        )

        # Buttons
        self.style.configure(
            "TButton",
            background=self.colors["btn"],
            foreground=self.colors["text"],
            padding=(10, 7),
            font=self.ui_font,
        )
        self.style.map(
            "TButton",
            background=[("active", self.colors["btn_hover"]), ("pressed", self.colors["panel2"])],
            foreground=[("disabled", "#6f6f6f")],
        )

        # Keep your existing named styles
        self.style.configure("Action.TButton", font=self.ui_font)
        self.style.configure("BoldAction.TButton", font=self.ui_font_bold)

        # Treeview (file list) styling
        try:
            self.style.configure(
                "Treeview",
                background=self.colors["panel"],
                fieldbackground=self.colors["panel"],
                foreground=self.colors["text"],
                rowheight=22,
                borderwidth=0,
            )
            self.style.map(
                "Treeview",
                background=[("selected", self.colors["select_bg"])],
                foreground=[("selected", self.colors["select_fg"])],
            )
        except Exception:
            pass

        # Line counters
        self.preview_lines_var = tk.StringVar(value="Lines: 0")
        self.replacement_lines_var = tk.StringVar(value="Lines: 0")

    def log(self, msg: str):
        ts = time.strftime("%H:%M:%S")
        self.activity.insert("end", f"[{ts}] {msg}\n")
        self.activity.see("end")
        self.status_var.set(msg)

    def _build_ui(self):
        # File row
        file_row = ttk.Frame(self)
        file_row.pack(fill="x", padx=8, pady=6)

        self.file_label_var = tk.StringVar(value="File: (none selected)")
        ttk.Label(file_row, textvariable=self.file_label_var).pack(side="left")

        ttk.Button(file_row, text="Open File...", command=self.on_open_file).pack(side="left", padx=(10, 0))

        # Recent files
        self.recent_var = tk.StringVar(value="")
        self.recent_combo = ttk.Combobox(file_row, textvariable=self.recent_var, state="readonly", width=60)
        self.recent_combo.pack(side="left", padx=6)
        self.recent_combo.bind("<<ComboboxSelected>>", self.on_recent_selected)

        # Section row
        sec_row = ttk.Frame(self)
        sec_row.pack(fill="x", padx=8, pady=(0, 6))

        ttk.Button(sec_row, text="Refresh Sections", command=self.on_refresh_sections).pack(side="left", padx=(0, 10))
        ttk.Label(sec_row, text="Section:").pack(side="left")

        self.section_var = tk.StringVar(value="")
        self.section_combo = ttk.Combobox(sec_row, textvariable=self.section_var, state="readonly", width=80)
        self.section_combo.pack(side="left", padx=6)
        self.section_combo.bind("<<ComboboxSelected>>", self.on_section_selected)

        self.lines_var = tk.StringVar(value="")
        ttk.Label(sec_row, textvariable=self.lines_var).pack(side="left", padx=(10, 0))

        panes = ttk.PanedWindow(self, orient="horizontal")
        panes.pack(fill="both", expand=True, padx=8, pady=6)

        left = ttk.Frame(panes)
        panes.add(left, weight=1)

        ttk.Label(left, text="Current Section Body (read-only preview)").pack(anchor="w")

        ttk.Label(left, textvariable=self.preview_lines_var, foreground=self.colors["muted"]).pack(anchor="w")

        find_row = ttk.Frame(left)
        find_row.pack(fill="x", pady=(2, 4))

        ttk.Label(find_row, text="Find:").pack(side="left")
        self.find_var = tk.StringVar(value="")
        self.find_entry = ttk.Entry(find_row, textvariable=self.find_var, width=30)
        self.find_entry.pack(side="left", padx=(6, 6))

        self.find_nocase = tk.BooleanVar(value=True)
        ttk.Checkbutton(find_row, text="Ignore case", variable=self.find_nocase).pack(side="left", padx=(0, 8))

        ttk.Button(find_row, text="Prev", command=self._preview_find_prev).pack(side="left")
        ttk.Button(find_row, text="Next", command=self._preview_find_next).pack(side="left", padx=(6, 0))

        self.find_status_var = tk.StringVar(value="")
        ttk.Label(find_row, textvariable=self.find_status_var).pack(side="left", padx=(10, 0))

        self.preview = tk.Text(left, wrap="none", height=20)
        self.preview.pack(fill="both", expand=True)
        self.preview.configure(
            bg=self.colors["panel"],
            fg=self.colors["text"],
            insertbackground=self.colors["text"],
            selectbackground=self.colors["select_bg"],
            selectforeground=self.colors["select_fg"],
            relief="solid",
            bd=1,
            highlightthickness=1,
            highlightbackground=self.colors["border"],
            highlightcolor=self.colors["border"],
            font=self.mono_font,
        )

        p_scroll_y = ttk.Scrollbar(left, orient="vertical", command=self.preview.yview)
        p_scroll_y.place(relx=1.0, rely=0.06, relheight=0.94, anchor="ne")
        self.preview.config(yscrollcommand=p_scroll_y.set)

        p_scroll_x = ttk.Scrollbar(left, orient="horizontal", command=self.preview.xview)
        p_scroll_x.pack(fill="x")
        self.preview.config(xscrollcommand=p_scroll_x.set)

        right = ttk.Frame(panes)
        panes.add(right, weight=1)

        ttk.Label(right, text="Replacement Body (will replace ONLY the body)").pack(anchor="w")

        ttk.Label(right, textvariable=self.replacement_lines_var, foreground=self.colors["muted"]).pack(anchor="w")

        self.replacement = tk.Text(right, wrap="none", height=20)
        self.replacement.pack(fill="both", expand=True)
        self.replacement.configure(
            bg=self.colors["panel"],
            fg=self.colors["text"],
            insertbackground=self.colors["text"],
            selectbackground=self.colors["select_bg"],
            selectforeground=self.colors["select_fg"],
            relief="solid",
            bd=1,
            highlightthickness=1,
            highlightbackground=self.colors["border"],
            highlightcolor=self.colors["border"],
            font=self.mono_font,
        )

        # Live update line count on typing
        self.replacement.bind("<KeyRelease>", lambda e: self._update_line_counters())
        self.replacement.bind("<ButtonRelease-1>", lambda e: self._update_line_counters())

        r_scroll_y = ttk.Scrollbar(right, orient="vertical", command=self.replacement.yview)
        r_scroll_y.place(relx=1.0, rely=0.06, relheight=0.94, anchor="ne")
        self.replacement.config(yscrollcommand=r_scroll_y.set)

        r_scroll_x = ttk.Scrollbar(right, orient="horizontal", command=self.replacement.xview)
        r_scroll_x.pack(fill="x")
        self.replacement.config(xscrollcommand=r_scroll_x.set)

        actions = ttk.Frame(self)
        actions.pack(fill="x", padx=8, pady=(0, 6))

        self.compile_var = tk.BooleanVar(value=bool(self.settings.get("safety", {}).get("py_compile_after_patch_default", True)))

        ttk.Button(actions, text="Preview Diff", command=self.on_preview_diff).pack(side="left", padx=(10, 0))
        ttk.Button(actions, text="Apply Patch", command=self.on_apply_patch).pack(side="left", padx=(10, 0))
        ttk.Button(actions, text="Undo Last Patch", command=self.on_undo).pack(side="left", padx=(10, 0))
        ttk.Button(actions, text="Auto-Split Section", command=self.on_auto_split_section).pack(side="left", padx=(10, 0))
        ttk.Button(actions, text="Create Backup Now", command=self.on_manual_backup).pack(side="left", padx=(10, 0))
        ttk.Checkbutton(actions, text="Validate .py (compile)", variable=self.compile_var).pack(side="right", padx=(0, 10))

# ==================================================================
# Clean Patch button - removes artifacts from pasted diffs
# ==================================================================

        ttk.Button(
            actions,
            text="Clean Patch Text",
            command=self.on_clean_patch
        ).pack(side="left", padx=(20, 0))

        ttk.Button(actions, text="Open Backups", command=self.on_open_backups).pack(side="right")
        ttk.Button(actions, text="Open Ledger", command=self.on_open_ledger).pack(side="right", padx=(0, 10))

        self.status_var = tk.StringVar(value="Ready")
        ttk.Label(self, textvariable=self.status_var).pack(fill="x", padx=8)

        ttk.Label(self, text="Activity Log").pack(anchor="w", padx=8)
        self.activity = tk.Text(self, height=8, wrap="word")
        self.activity.pack(fill="both", expand=False, padx=8, pady=(0, 8))
        self.activity.configure(
            state="normal",
            bg=self.colors["panel2"],
            fg=self.colors["text"],
            insertbackground=self.colors["text"],
            selectbackground=self.colors["select_bg"],
            selectforeground=self.colors["select_fg"],
            relief="solid",
            bd=1,
            highlightthickness=1,
            highlightbackground=self.colors["border"],
            highlightcolor=self.colors["border"],
            font=self.mono_font,
        )

        # Install context menus
        install_context_menu(self.preview)
        install_context_menu(self.replacement)
        install_context_menu(self.activity)
        install_context_menu(self.find_entry)

    def _update_line_counters(self):
        preview_text = self.preview.get("1.0", "end-1c")
        preview_lines = len(preview_text.splitlines()) if preview_text else 0
        self.preview_lines_var.set(f"Lines: {preview_lines}")

        replacement_text = self.replacement.get("1.0", "end-1c")
        replacement_lines = len(replacement_text.splitlines()) if replacement_text else 0
        self.replacement_lines_var.set(f"Lines: {replacement_lines}")

    def _refresh_recent_dropdown(self):
        display = []
        limit = self.settings.get("behavior", {}).get("recent_limit", 10)
        for it in self.recents[:limit]:
            p = it["path"]
            tag = "" if os.path.exists(p) else " (missing)"
            display.append(p + tag)
        self.recent_combo["values"] = display
        if display:
            self.recent_combo.current(0)

    def on_recent_selected(self, _evt=None):
        val = self.recent_var.get().replace(" (missing)", "")
        if val and os.path.exists(val):
            self.set_file(val)
        else:
            self.log("Recent file missing or invalid.")

    def on_open_file(self):
        start_dir = os.path.dirname(self.file_path) if self.file_path else tool_root()
        path = filedialog.askopenfilename(initialdir=start_dir, title="Select a file to patch")
        if not path:
            return
        self.set_file(path)

    def set_file(self, path: str):
        self.file_path = os.path.abspath(path)
        self.file_label_var.set(f"File: {self.file_path}")
        self.log(f"Selected file: {self.file_path}")

        self.recents = touch_recent(self.recents, self.file_path, patched=False)
        save_recents(self.paths["recent_path"], self.recents)
        self._refresh_recent_dropdown()

        self.sections = []
        self.section_map = {}
        self.section_combo["values"] = []
        self.section_var.set("")
        self.lines_var.set("")
        self._set_preview_text("")
        self.replacement.delete("1.0", "end")
        self.find_var.set("")
        self.find_status_var.set("")
        self._preview_find_clear()
        self.on_refresh_sections()

    def on_refresh_sections(self):
        if not self.file_path:
            self.log("No file selected.")
            return
        try:
            text = open(self.file_path, "r", encoding="utf-8").read()
        except Exception as e:
            self.log(f"Read failed: {e}")
            return

        sections, errors = detect_sections(text)
        self.sections = sections
        self.section_map = {s.identity: s for s in sections}

        if errors:
            self.log("Section validation errors: " + " | ".join(errors))

        if not sections:
            self.log("No SECTION headers detected in file.")
            self.section_combo["values"] = []
            return

        lines = text.splitlines(keepends=True)
        values = []
        for s in sorted(sections, key=lambda x: x.start_line):
            try:
                _, _, bs, be = compute_section_body_range(lines, sections, s)
                lock = "🔒 " if s.protected else ""
                values.append(f"{lock}{s.identity} ({bs}–{be})")
            except Exception:
                lock = "🔒 " if s.protected else ""
                values.append(f"{lock}{s.identity}")

        self.section_combo["values"] = values
        self.log(f"Detected {len(sections)} sections.")

    def on_section_selected(self, _evt=None):
        if not self.file_path:
            return
        sel = self.section_var.get()
        if not sel:
            return

        raw = sel
        if raw.startswith("🔒 "):
            raw = raw[2:]
        if raw.endswith(")"):
            pos = raw.rfind(" (")
            if pos != -1:
                raw = raw[:pos]
        identity = raw.strip()

        s = self.section_map.get(identity)
        if not s:
            self.log("Selected section not found in map. Refresh sections.")
            return

        try:
            text = open(self.file_path, "r", encoding="utf-8").read()
            lines = text.splitlines(keepends=True)
            start_idx, end_idx, bs, be = compute_section_body_range(lines, self.sections, s)
            body = "".join(lines[start_idx:end_idx])
            self.lines_var.set(f"Body lines: {bs}–{be}")
            self._set_preview_text(body)
            self._update_line_counters()
            if s.protected:
                self.log("Selected section is PROTECTED (patching refused).")
            else:
                self.log(f"Selected section: {identity}")
        except Exception as e:
            self.log(f"Preview failed: {e}")

    def _set_preview_text(self, body: str):
        self.preview.configure(state="normal")
        self.preview.delete("1.0", "end")
        self.preview.insert("1.0", body)
        self.preview.configure(state="disabled")
        self._update_line_counters()

# ==================================================================
# SECTION: GUI :: Find in Preview (Working) ----------
# ==================================================================

    def _preview_find_clear(self):
        try:
            self.preview.tag_remove("find_all", "1.0", "end")
            self.preview.tag_remove("find_hit", "1.0", "end")
            self.preview.tag_remove("sel", "1.0", "end")
        except Exception:
            pass

    def _preview_find_apply(self, pattern: str):
        self._preview_find_clear()
        if not pattern:
            return

        self.preview.configure(state="normal")

        start = "1.0"
        hits = 0
        while True:
            pos = self.preview.search(
                pattern,
                start,
                stopindex="end",
                nocase=self.find_nocase.get()
            )
            if not pos:
                break
            end = f"{pos}+{len(pattern)}c"
            self.preview.tag_add("find_all", pos, end)
            hits += 1
            start = end

        self.preview.tag_config("find_all", background="#fff3a0")
        self.preview.tag_config("find_hit", background="#ffd000")

        self._find_hits = []
        if hits:
            ranges = self.preview.tag_ranges("find_all")
            for i in range(0, len(ranges), 2):
                self._find_hits.append((str(ranges[i]), str(ranges[i + 1])))

            self._find_index = 0
            self._preview_find_goto_index(self._find_index)
            self.find_status_var.set(f"{hits} match(es)")
        else:
            self.find_status_var.set("No matches")

        self.preview.configure(state="disabled")

    def _preview_find_goto_index(self, idx: int):
        if not getattr(self, "_find_hits", None):
            return
        if not self._find_hits:
            return

        idx = max(0, min(idx, len(self._find_hits) - 1))
        self._find_index = idx

        self.preview.configure(state="normal")
        self.preview.tag_remove("find_hit", "1.0", "end")

        start, end = self._find_hits[idx]
        self.preview.tag_add("find_hit", start, end)
        self.preview.see(start)

        self.preview.tag_remove("sel", "1.0", "end")
        self.preview.tag_add("sel", start, end)

        self.preview.configure(state="disabled")

    def _preview_find_next(self, _evt=None):
        pattern = self.find_var.get()
        if not pattern:
            return "break"

        if getattr(self, "_find_last", None) != (pattern, self.find_nocase.get()):
            self._find_last = (pattern, self.find_nocase.get())
            self._preview_find_apply(pattern)
            return "break"

        if getattr(self, "_find_hits", None) and self._find_hits:
            self._preview_find_goto_index((self._find_index + 1) % len(self._find_hits))
        return "break"

    def _preview_find_prev(self, _evt=None):
        pattern = self.find_var.get()
        if not pattern:
            return "break"

        if getattr(self, "_find_last", None) != (pattern, self.find_nocase.get()):
            self._find_last = (pattern, self.find_nocase.get())
            self._preview_find_apply(pattern)
            return "break"

        if getattr(self, "_find_hits", None) and self._find_hits:
            self._preview_find_goto_index((self._find_index - 1) % len(self._find_hits))
        return "break"

# ==================================================================
# SECTION: GUI :: Clean Patch Text 
# ==================================================================

    def on_clean_patch(self):
        text = self.replacement.get("1.0", "end-1c")
        if not text.strip():
            self.log("No text to clean.")
            return

        cleaned = self._clean_patch_text(text)
        self.replacement.delete("1.0", "end")
        self.replacement.insert("1.0", cleaned)
        self._update_line_counters()
        self.log("Patch text cleaned (diff artifacts removed).")

    def _clean_patch_text(self, text: str) -> str:
        lines = text.splitlines()
        cleaned = []
        in_hunk = False
        for line in lines:
            stripped = line.strip()
            if stripped.startswith('diff ') or stripped.startswith('index '):
                continue
            if stripped.startswith('--- ') or stripped.startswith('+++ '):
                continue
            if stripped.startswith('@@ '):
                in_hunk = True
                continue
            if in_hunk:
                if line.startswith('+'):
                    cleaned.append(line[1:])
                elif line.startswith(' '):
                    cleaned.append(line[1:])
                elif line.startswith('-'):
                    continue  # Ignore removals
                else:
                    cleaned.append(line)  # Fallback for unprefixed lines
            else:
                cleaned.append(line)  # Non-diff content
        return '\n'.join(cleaned).rstrip('\n')

# ==================================================================
# SECTION: GUI :: Manual Backup 
# ==================================================================

    def on_manual_backup(self):
        if not self.file_path:
            self.log("No file selected.")
            return

        section_id = "whole_file"
        if self.section_var.get():
            raw = self.section_var.get()
            if raw.startswith("🔒 "):
                raw = raw[2:]
            if raw.endswith(")"):
                pos = raw.rfind(" (")
                if pos != -1:
                    raw = raw[:pos]
            section_id = raw.strip()

        try:
            backup_path = make_backup(self.paths["backups_dir"], self.file_path, section_id)
            self.log(f"Manual backup created: {os.path.basename(backup_path)}")
            if messagebox.askyesno("Backup Created", f"Backup saved.\nOpen backups folder?"):
                self.on_open_backups()
        except Exception as e:
            self.log(f"Backup failed: {e}")

# ==================================================================
# SECTION: GUI :: Actions 
# ==================================================================


    def on_auto_split_section(self):
        if not self.file_path:
            self.log("No file selected.")
            return

        try:
            old_text = open(self.file_path, "r", encoding="utf-8").read()
        except Exception as e:
            self.log(f"Read failed: {e}")
            return

        sections, errors = detect_sections(old_text)
        if errors:
            self.log("Cannot auto-split: " + " | ".join(errors))
            return

        # If the file has NO section headers yet, auto-split means:
        # INSERT SECTION HEADERS into the same file (no multi-file splitting).
        if not sections:
            new_text, created_count = auto_format_file_text_into_sections(
                file_path=self.file_path,
                text=old_text,
                category="FILE",
            )

            if created_count == 0 or new_text == old_text:
                self.log("Auto-format: no usable boundaries found.")
                return

            diff = preview_diff(old_text, new_text, self.file_path)

            win = tk.Toplevel(self)
            win.title("Auto-Format Diff Preview (Insert SECTION headers)")
            win.geometry("900x600")

            txt = tk.Text(win, wrap="none")
            txt.pack(fill="both", expand=True)
            txt.insert("1.0", diff if diff.strip() else "(No diff produced.)")
            txt.configure(state="disabled")

            def apply_now():
                win.destroy()
                ok, msg, _ = apply_format_whole_file(
                    file_path=self.file_path,
                    new_text=new_text,
                    paths=self.paths,
                    run_py_compile=bool(self.compile_var.get()),
                    activity_log_cb=self.log,
                )
                self.log(msg)
                if ok:
                    self.on_refresh_sections()

            btns = ttk.Frame(win)
            btns.pack(fill="x", padx=8, pady=8)
            ttk.Button(btns, text="Cancel", command=win.destroy).pack(side="right")
            ttk.Button(btns, text=f"Apply Auto-Format ({created_count})", command=apply_now).pack(side="right", padx=(0, 10))

            self.log(f"Auto-format preview ready: {created_count} new sections inserted.")
            return

        # Normal behavior: auto-split the SELECTED section body into subsections.
        sel = self.section_var.get()
        if not sel:
            self.log("No section selected.")
            return

        raw = sel
        if raw.startswith("🔒 "):
            raw = raw[2:]
        if raw.endswith(")"):
            pos = raw.rfind(" (")
            if pos != -1:
                raw = raw[:pos]
        identity = raw.strip()

        s = self.section_map.get(identity)
        if not s:
            self.log("Selected section not found. Refresh sections.")
            return
        if s.protected:
            self.log("Selected section is PROTECTED (cannot auto-split).")
            return

        matches = [sec for sec in sections if sec.identity == identity]
        if len(matches) != 1:
            self.log(f"Cannot auto-split: section matches={len(matches)}")
            return

        sec = matches[0]
        lines = old_text.splitlines(keepends=True)
        start_idx, end_idx, _, _ = compute_section_body_range(lines, sections, sec)
        body = "".join(lines[start_idx:end_idx])

        new_body, created_count = auto_split_section_body(
            category=sec.category,
            section_name=sec.name,
            body_text=body,
        )

        if created_count == 0 or new_body == body:
            self.log("Auto-split: no usable boundaries found.")
            return

        new_lines = lines[:start_idx] + new_body.splitlines(keepends=True) + lines[end_idx:]
        new_text = "".join(new_lines)
        diff = preview_diff(old_text, new_text, self.file_path)

        win = tk.Toplevel(self)
        win.title("Auto-Split Diff Preview")
        win.geometry("900x600")

        txt = tk.Text(win, wrap="none")
        txt.pack(fill="both", expand=True)
        txt.insert("1.0", diff if diff.strip() else "(No diff produced.)")
        txt.configure(state="disabled")

        def apply_now():
            win.destroy()
            ok, msg, _ = apply_patch_replace_body(
                file_path=self.file_path,
                section_identity=identity,
                replacement_body=new_body,
                paths=self.paths,
                run_py_compile=bool(self.compile_var.get()),
                activity_log_cb=self.log,
            )
            self.log(msg)
            if ok:
                self.on_refresh_sections()
                self.section_var.set(sel)
                self.on_section_selected()

        btns = ttk.Frame(win)
        btns.pack(fill="x", padx=8, pady=8)
        ttk.Button(btns, text="Cancel", command=win.destroy).pack(side="right")
        ttk.Button(btns, text=f"Apply Auto-Split ({created_count})", command=apply_now).pack(side="right", padx=(0, 10))

        self.log(f"Auto-split preview ready: {created_count} new subsections.")

    def on_preview_diff(self):
        if not self.file_path:
            self.log("No file selected.")
            return

        sel = self.section_var.get()
        if not sel:
            self.log("No section selected.")
            return

        raw = sel
        if raw.startswith("🔒 "):
            raw = raw[2:]
        if raw.endswith(")"):
            pos = raw.rfind(" (")
            if pos != -1:
                raw = raw[:pos]
        identity = raw.strip()

        replacement_body = self.replacement.get("1.0", "end-1c")
        if not replacement_body.strip():
            self.log("Replacement is empty.")
            return

        try:
            old_text = open(self.file_path, "r", encoding="utf-8").read()
            sections, errors = detect_sections(old_text)
            if errors:
                self.log("Cannot preview diff: " + " | ".join(errors))
                return

            matches = [s for s in sections if s.identity == identity]
            if len(matches) != 1:
                self.log(f"Cannot preview diff: section matches={len(matches)}")
                return

            sec = matches[0]
            lines = old_text.splitlines(keepends=True)
            start_idx, end_idx, _, _ = compute_section_body_range(lines, sections, sec)

            rep = replacement_body
            if rep and not rep.endswith("\n"):
                rep += "\n"
            rep_lines = rep.splitlines(keepends=True)

            new_lines = lines[:start_idx] + rep_lines + lines[end_idx:]
            new_text = "".join(new_lines)
            if new_text and not new_text.endswith("\n"):
                new_text += "\n"

            diff = preview_diff(old_text, new_text, self.file_path)

            win = tk.Toplevel(self)
            win.title("Diff Preview")
            win.geometry("900x600")
            txt = tk.Text(win, wrap="none")
            txt.pack(fill="both", expand=True)
            txt.insert("1.0", diff if diff.strip() else "(No diff produced — replacement may be identical.)")
            txt.configure(state="disabled")
            self.log("Diff preview opened.")
        except Exception as e:
            self.log(f"Diff preview failed: {e}")

    def _first_patch_dialog(self) -> bool:
        win = tk.Toplevel(self)
        win.title("First Patch Safety")
        win.geometry("520x260")
        win.grab_set()

        txt = (
            "Fracture will create an automatic backup before patching.\n\n"
            "If your project is not under version control, consider copying the project folder as an extra safety net.\n\n"
            "Continue?"
        )
        ttk.Label(win, text=txt, justify="left", wraplength=480).pack(padx=15, pady=15, anchor="w")

        var = tk.BooleanVar(value=False)
        ttk.Checkbutton(win, text="Don’t show this again", variable=var).pack(padx=15, pady=(0, 10), anchor="w")

        result = {"ok": False}

        def on_continue():
            result["ok"] = True
            if var.get():
                self.settings.setdefault("safety", {})["first_patch_warning_dismissed"] = True
                save_json(self.paths["settings_path"], self.settings)
            win.destroy()

        def on_cancel():
            result["ok"] = False
            win.destroy()

        btns = ttk.Frame(win)
        btns.pack(fill="x", padx=15, pady=10)
        ttk.Button(btns, text="Cancel", command=on_cancel).pack(side="right")
        ttk.Button(btns, text="Continue", command=on_continue).pack(side="right", padx=(0, 10))

        self.wait_window(win)
        return result["ok"]

    def on_apply_patch(self):
        if not self.file_path:
            self.log("No file selected.")
            return
        sel = self.section_var.get()
        if not sel:
            self.log("No section selected.")
            return

        raw = sel
        if raw.startswith("🔒 "):
            raw = raw[2:]
        if raw.endswith(")"):
            pos = raw.rfind(" (")
            if pos != -1:
                raw = raw[:pos]
        identity = raw.strip()

        if not bool(self.settings.get("safety", {}).get("first_patch_warning_dismissed", False)):
            if not self._first_patch_dialog():
                self.log("Patch canceled.")
                return

        replacement_body = self.replacement.get("1.0", "end-1c")
        if not replacement_body.strip():
            self.log("Replacement is empty.")
            return

        if not messagebox.askokcancel(
            "Confirm Patch",
            f"File:\n{self.file_path}\n\nSection:\n{identity}\n\nThis will REPLACE ONLY the body of the selected section.\n\nProceed?"
        ):
            self.log("Patch canceled.")
            return

        ok, msg, _backup_path = apply_patch_replace_body(
            file_path=self.file_path,
            section_identity=identity,
            replacement_body=replacement_body,
            paths=self.paths,
            run_py_compile=bool(self.compile_var.get()),
            activity_log_cb=self.log
        )

        self.log(msg)

        if ok:
            self.recents = touch_recent(self.recents, self.file_path, patched=True)
            save_recents(self.paths["recent_path"], self.recents)
            self._refresh_recent_dropdown()
            self.on_refresh_sections()
            self.section_var.set(sel)
            self.on_section_selected()

    def on_undo(self):
        last = get_last_successful_patch_event(self.paths["ledger_path"])
        if not last:
            self.log("Nothing to undo.")
            return

        target_path = (last.get("target") or {}).get("file_path", "")
        section_identity = (last.get("target") or {}).get("section_identity", "")
        backup_path = (last.get("backup") or {}).get("backup_path", "")
        when = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(last.get("timestamp", time.time())))

        if not messagebox.askokcancel(
            "Undo Last Patch",
            f"Patched at: {when}\n\nFile:\n{target_path}\n\nSection:\n{section_identity}\n\nBackup:\n{backup_path}\n\nThis will overwrite the current file with the backup.\nProceed?"
        ):
            self.log("Undo canceled.")
            return

        ok, msg = undo_last_patch(self.paths, activity_log_cb=self.log)
        self.log(msg)

        if self.file_path and os.path.abspath(self.file_path) == os.path.abspath(target_path):
            self.on_refresh_sections()

    def _open_path(self, path: str):
        try:
            if os.name == "nt":
                os.startfile(path)
            elif sys.platform == "darwin":  # Fixed: added import sys at top if needed, but assuming it's global
                subprocess.run(["open", path], check=False)
            else:
                subprocess.run(["xdg-open", path], check=False)
        except Exception as e:
            self.log(f"Open failed: {e}")

    def on_open_backups(self):
        self._open_path(self.paths["backups_dir"])

    def on_open_ledger(self):
        self._open_path(self.paths["ledger_path"])


# ==================================================================
# SECTION: BOOT :: Entry point [PROTECTED]
# ==================================================================

def main():
    paths = ensure_dirs()
    app = FractureApp(paths)
    app.mainloop()


if __name__ == "__main__":
    main()