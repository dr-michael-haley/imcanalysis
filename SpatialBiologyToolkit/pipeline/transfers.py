"""Stable project file identities, safe bundles, uploads, backups, and restores."""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import stat
import uuid
import zipfile
from pathlib import Path, PurePosixPath
from typing import Any, Literal

from pydantic import Field

from .assets import resolve_assets
from .control import (
    ActionRecord,
    action_receipt_payload,
    make_preview_token,
    validate_preview_token,
)
from .manifests import utc_now, write_json
from .models import PipelineModel
from .project import ProjectContext

TRANSFER_ROOT = Path(".sbt/transfers")
BACKUP_ROOT = Path(".sbt/backups")
LARGE_TRANSFER_BYTES = 3 * 1024**3
MAX_TRANSFER_FILES = 100_000
SAFE_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._ -]{0,254}$")
HDF5_SIGNATURE = b"\x89HDF\r\n\x1a\n"
MAX_EXPANDED_UPLOAD_BYTES = 100 * 1024**3


class TransferItem(PipelineModel):
    item_id: str
    display_name: str
    relative_path: str
    kind: Literal["file", "directory"]
    role: str | None = None
    size_bytes: int
    file_count: int
    modified_ns: int


class TransferPreview(PipelineModel):
    schema_version: Literal[1] = 1
    direction: Literal["download", "upload"]
    total_bytes: int = Field(ge=0)
    file_count: int = Field(ge=0)
    requires_large_transfer_permission: bool
    items: list[TransferItem] = Field(default_factory=list)
    destination: str | None = None
    preview_token: str
    preview_expires_in_seconds: int = 900


def _within(root: Path, path: Path) -> bool:
    root = root.resolve(strict=False)
    path = path.resolve(strict=False)
    return path == root or root in path.parents


def _safe_relative(root: Path, path: Path) -> str:
    resolved = path.resolve(strict=False)
    if not _within(root, resolved):
        raise ValueError(f"Transfer path is outside the registered project: {path}")
    return resolved.relative_to(root.resolve(strict=False)).as_posix()


def _item_id(project_id: str, relative_path: str) -> str:
    digest = hashlib.sha256(f"{project_id}\0{relative_path}".encode()).hexdigest()
    return f"item-{digest[:24]}"


def _walk_stats(path: Path, *, max_files: int = MAX_TRANSFER_FILES) -> tuple[int, int]:
    if path.is_symlink():
        raise ValueError(f"Symbolic links are not transferable: {path}")
    if path.is_file():
        return path.stat().st_size, 1
    total = 0
    count = 0
    for directory, dirs, files in os.walk(path, followlinks=False):
        base = Path(directory)
        dirs.sort()
        for name in list(dirs):
            candidate = base / name
            if candidate.is_symlink():
                raise ValueError(f"Symbolic links are not transferable: {candidate}")
        for name in files:
            candidate = base / name
            if candidate.is_symlink() or not candidate.is_file():
                raise ValueError(f"Non-regular files are not transferable: {candidate}")
            count += 1
            if count > max_files:
                raise ValueError(f"Transfer exceeds the {max_files} file limit.")
            total += candidate.stat().st_size
    return total, count


def _make_item(
    context: ProjectContext,
    path: Path,
    *,
    role: str | None = None,
) -> TransferItem:
    relative = _safe_relative(context.root, path)
    size, count = _walk_stats(path)
    info = path.stat()
    return TransferItem(
        item_id=_item_id(context.project_metadata.project_id, relative),
        display_name=path.name,
        relative_path=relative,
        kind="file" if path.is_file() else "directory",
        role=role,
        size_bytes=size,
        file_count=count,
        modified_ns=info.st_mtime_ns,
    )


def list_transfer_items(
    context: ProjectContext,
    *,
    max_items: int = 10_000,
) -> list[TransferItem]:
    """List bounded project files and configured assets using stable IDs."""

    candidates: dict[Path, str | None] = {}
    for asset in resolve_assets(context.config, context.root, count_limit=max_items):
        if asset.exists and not asset.path.is_symlink() and _within(context.root, asset.path):
            candidates[asset.path.resolve(strict=False)] = asset.role
            if asset.path.is_dir():
                try:
                    children = sorted(asset.path.iterdir(), key=lambda item: item.name.casefold())
                except OSError:
                    children = []
                for child in children:
                    if len(candidates) >= max_items:
                        break
                    if (child.is_file() or child.is_dir()) and not child.is_symlink():
                        candidates.setdefault(child.resolve(strict=False), asset.role)
    try:
        root_entries = sorted(context.root.iterdir(), key=lambda item: item.name.casefold())
    except OSError:
        root_entries = []
    for entry in root_entries:
        if entry.name == ".sbt" or entry.is_symlink():
            continue
        if entry.is_file() or entry.is_dir():
            candidates.setdefault(entry.resolve(strict=False), None)

    items: list[TransferItem] = []
    for path, role in candidates.items():
        if len(items) >= max_items:
            break
        try:
            items.append(_make_item(context, path, role=role))
        except (OSError, ValueError):
            continue
    return sorted(items, key=lambda item: item.relative_path.casefold())


def resolve_transfer_items(
    context: ProjectContext,
    item_ids: list[str],
) -> list[tuple[TransferItem, Path]]:
    if not item_ids or len(item_ids) != len(set(item_ids)):
        raise ValueError("Select one or more unique transfer item IDs.")
    if len(item_ids) > 1000:
        raise ValueError("A transfer may select at most 1000 item IDs.")
    available = {item.item_id: item for item in list_transfer_items(context)}
    missing = [item_id for item_id in item_ids if item_id not in available]
    if missing:
        raise ValueError(f"Unknown or stale transfer item IDs: {', '.join(missing)}")
    selected = [available[item_id] for item_id in item_ids]
    for index, item in enumerate(selected):
        if item.kind != "directory":
            continue
        parent = PurePosixPath(item.relative_path)
        for other in selected[index + 1 :] + selected[:index]:
            if parent in PurePosixPath(other.relative_path).parents:
                raise ValueError(
                    "Transfer selections cannot include both a directory and one of its children."
                )
    return [(item, context.root / item.relative_path) for item in selected]


def download_snapshot(items: list[TransferItem]) -> dict[str, Any]:
    return {
        "kind": "download",
        "items": [item.model_dump(mode="json") for item in items],
        "total_bytes": sum(item.size_bytes for item in items),
        "file_count": sum(item.file_count for item in items),
    }


def preview_download(context: ProjectContext, item_ids: list[str]) -> dict[str, Any]:
    resolved = resolve_transfer_items(context, item_ids)
    items = [item for item, _path in resolved]
    total = sum(item.size_bytes for item in items)
    count = sum(item.file_count for item in items)
    token = make_preview_token(download_snapshot(items))
    preview = TransferPreview(
        direction="download",
        total_bytes=total,
        file_count=count,
        requires_large_transfer_permission=total > LARGE_TRANSFER_BYTES,
        items=items,
        preview_token=token,
    )
    return {
        **preview.model_dump(mode="json"),
        "bundle_recommended": len(items) > 1 or any(item.kind == "directory" for item in items),
        "action_receipt": action_receipt_payload(
            operation="preview_download",
            target=context.project_metadata.project_id,
            actions=[
                ActionRecord(
                    action="Resolved stable project item IDs and measured transfer size",
                    justification="Transfers require a bounded size/count preview before data movement.",
                    outcome="succeeded",
                    evidence=[f"bytes={total}", f"files={count}"],
                )
            ],
        ),
    }


def _sha256(path: Path, *, chunk_size: int = 8 * 1024**2) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def _path_digest(path: Path) -> str:
    """Hash a file or a directory tree for backup/restore verification."""

    if path.is_symlink():
        raise ValueError("Backup verification does not follow symbolic links.")
    if path.is_file():
        return _sha256(path)
    digest = hashlib.sha256()
    count = 0
    for directory, dirs, files in os.walk(path, followlinks=False):
        base = Path(directory)
        dirs.sort()
        for name in dirs:
            if (base / name).is_symlink():
                raise ValueError("Backup verification does not follow symbolic links.")
        for name in sorted(files):
            candidate = base / name
            if candidate.is_symlink() or not candidate.is_file():
                raise ValueError("Backups may contain regular files only.")
            count += 1
            if count > MAX_TRANSFER_FILES:
                raise ValueError("Backup verification exceeded the file-count limit.")
            relative = candidate.relative_to(path).as_posix().encode("utf-8")
            digest.update(len(relative).to_bytes(8, "big"))
            digest.update(relative)
            digest.update(bytes.fromhex(_sha256(candidate)))
    return digest.hexdigest()


def _zip_items(
    context: ProjectContext,
    resolved: list[tuple[TransferItem, Path]],
    destination: Path,
) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".part")
    with zipfile.ZipFile(
        temporary,
        "w",
        compression=zipfile.ZIP_DEFLATED,
        allowZip64=True,
    ) as archive:
        for item, path in resolved:
            if path.is_file():
                archive.write(path, arcname=item.relative_path)
                continue
            for directory, dirs, files in os.walk(path, followlinks=False):
                base = Path(directory)
                dirs.sort()
                for name in dirs:
                    if (base / name).is_symlink():
                        raise ValueError("Bundles cannot include symbolic links.")
                for name in files:
                    candidate = base / name
                    if candidate.is_symlink() or not candidate.is_file():
                        raise ValueError("Bundles can contain regular files only.")
                    archive.write(
                        candidate,
                        arcname=_safe_relative(context.root, candidate),
                    )
        archive.writestr(
            "SBT_TRANSFER_MANIFEST.json",
            json.dumps(
                {
                    "schema_version": 1,
                    "project_id": context.project_metadata.project_id,
                    "created_at": utc_now().isoformat(),
                    "items": [item.model_dump(mode="json") for item, _path in resolved],
                },
                indent=2,
            ),
        )
    os.replace(temporary, destination)


def prepare_download(
    context: ProjectContext,
    item_ids: list[str],
    *,
    preview_token: str,
    allow_large_transfer: bool = False,
    bundle_name: str | None = None,
    force_bundle: bool = False,
) -> dict[str, Any]:
    resolved = resolve_transfer_items(context, item_ids)
    items = [item for item, _path in resolved]
    validate_preview_token(preview_token, download_snapshot(items))
    total = sum(item.size_bytes for item in items)
    if total > LARGE_TRANSFER_BYTES and not allow_large_transfer:
        raise ValueError(
            "This transfer exceeds 3 GiB and requires explicit large-transfer permission."
        )
    transfer_id = f"download-{uuid.uuid4().hex}"
    if len(resolved) == 1 and resolved[0][0].kind == "file" and not force_bundle:
        remote_path = resolved[0][1]
        download_name = resolved[0][0].display_name
        bundled = False
    else:
        name = bundle_name or f"{context.root.name}_{transfer_id[-8:]}.zip"
        if not SAFE_NAME.fullmatch(name) or not name.lower().endswith(".zip"):
            raise ValueError("Bundle names must be safe filenames ending in .zip.")
        remote_path = context.root / TRANSFER_ROOT / "downloads" / transfer_id / name
        _zip_items(context, resolved, remote_path)
        download_name = name
        bundled = True
    record = {
        "schema_version": 1,
        "transfer_id": transfer_id,
        "direction": "download",
        "created_at": utc_now(),
        "items": [item.model_dump(mode="json") for item in items],
        "remote_path": str(remote_path),
        "download_name": download_name,
        "size_bytes": remote_path.stat().st_size,
        "sha256": _sha256(remote_path),
        "bundled": bundled,
    }
    write_json(context.root / TRANSFER_ROOT / "records" / f"{transfer_id}.json", record)
    return {
        **record,
        "action_receipt": action_receipt_payload(
            operation="prepare_download",
            target=transfer_id,
            actions=[
                ActionRecord(
                    action="Prepared a verified remote download source",
                    justification=(
                        "Multiple items and folders are bundled before transfer."
                        if bundled
                        else "The selected regular file can be transferred directly."
                    ),
                    outcome="succeeded",
                    state_changed=bundled,
                    evidence=[f"sha256={record['sha256']}", f"bytes={record['size_bytes']}"],
                )
            ],
        ),
    }


def _validate_name(name: str) -> str:
    normalized = name.strip()
    if (
        not SAFE_NAME.fullmatch(normalized)
        or normalized in {".", ".."}
        or "/" in normalized
        or "\\" in normalized
    ):
        raise ValueError("Upload names must be a single safe filename or folder name.")
    return normalized


def _upload_destination(
    context: ProjectContext,
    *,
    destination: str,
    name: str,
    kind: Literal["file", "directory"],
) -> Path:
    name = _validate_name(name)
    if destination == "project-root":
        if kind == "file" and Path(name).suffix.lower() != ".h5ad":
            raise ValueError("Direct project-root file uploads are restricted to .h5ad files.")
        target = context.root / name
    elif destination == "metadata":
        target = Path(context.config.general.metadata_folder)
        if not target.is_absolute():
            target = context.root / target
        target = target / name
    else:
        raise ValueError("Upload destination must be project-root or metadata.")
    target = target.resolve(strict=False)
    if not _within(context.root, target):
        raise ValueError("The upload destination resolves outside the registered project.")
    return target


def upload_snapshot(
    context: ProjectContext,
    *,
    name: str,
    destination: str,
    kind: Literal["file", "directory"],
    size_bytes: int,
    sha256: str,
    overwrite: bool,
) -> dict[str, Any]:
    target = _upload_destination(context, destination=destination, name=name, kind=kind)
    target_state: dict[str, Any] | None = None
    if target.exists():
        target_size, target_count = _walk_stats(target)
        target_state = {
            "kind": "file" if target.is_file() else "directory",
            "size_bytes": target_size,
            "file_count": target_count,
            "modified_ns": target.stat().st_mtime_ns,
        }
    return {
        "kind": "upload",
        "project_id": context.project_metadata.project_id,
        "name": name,
        "destination": destination,
        "target_relative": _safe_relative(context.root, target),
        "item_kind": kind,
        "size_bytes": size_bytes,
        "sha256": sha256,
        "overwrite": overwrite,
        "target_exists": target.exists(),
        "target_state": target_state,
    }


def preview_upload(
    context: ProjectContext,
    *,
    name: str,
    destination: str,
    kind: Literal["file", "directory"],
    size_bytes: int,
    sha256: str,
    overwrite: bool,
) -> dict[str, Any]:
    if size_bytes < 0 or not re.fullmatch(r"[a-f0-9]{64}", sha256):
        raise ValueError("Upload preview requires a non-negative size and SHA-256 digest.")
    snapshot = upload_snapshot(
        context,
        name=name,
        destination=destination,
        kind=kind,
        size_bytes=size_bytes,
        sha256=sha256,
        overwrite=overwrite,
    )
    if snapshot["target_exists"] and not overwrite:
        raise FileExistsError(
            "The upload destination exists; request overwrite to require a verified backup."
        )
    token = make_preview_token(snapshot)
    return {
        "schema_version": 1,
        **snapshot,
        "preview_token": token,
        "preview_expires_in_seconds": 900,
        "requires_large_transfer_permission": size_bytes > LARGE_TRANSFER_BYTES,
        "backup_required": bool(snapshot["target_exists"]),
        "action_receipt": action_receipt_payload(
            operation="preview_upload",
            target=str(snapshot["target_relative"]),
            actions=[
                ActionRecord(
                    action="Validated the SBT-managed upload destination and collision policy",
                    justification="Remote paths are resolved by SBT and replacements require backups.",
                    outcome="succeeded",
                    evidence=[f"bytes={size_bytes}", f"overwrite={overwrite}"],
                )
            ],
        ),
    }


def prepare_upload(
    context: ProjectContext,
    *,
    name: str,
    destination: str,
    kind: Literal["file", "directory"],
    size_bytes: int,
    sha256: str,
    overwrite: bool,
    preview_token: str,
    allow_large_transfer: bool = False,
) -> dict[str, Any]:
    snapshot = upload_snapshot(
        context,
        name=name,
        destination=destination,
        kind=kind,
        size_bytes=size_bytes,
        sha256=sha256,
        overwrite=overwrite,
    )
    validate_preview_token(preview_token, snapshot)
    if size_bytes > LARGE_TRANSFER_BYTES and not allow_large_transfer:
        raise ValueError(
            "This transfer exceeds 3 GiB and requires explicit large-transfer permission."
        )
    transfer_id = f"upload-{uuid.uuid4().hex}"
    staging_dir = context.root / TRANSFER_ROOT / "uploads" / transfer_id
    staging_dir.mkdir(parents=True, exist_ok=False)
    staging_path = staging_dir / ("payload.zip.part" if kind == "directory" else "payload.part")
    manifest = {
        "schema_version": 1,
        "transfer_id": transfer_id,
        "direction": "upload",
        "created_at": utc_now(),
        **snapshot,
        "staging_path": str(staging_path),
        "status": "awaiting_upload",
    }
    write_json(staging_dir / "manifest.json", manifest)
    return {
        **manifest,
        "action_receipt": action_receipt_payload(
            operation="prepare_upload",
            target=transfer_id,
            actions=[
                ActionRecord(
                    action="Created an isolated SBT upload staging target",
                    justification="Uploaded bytes must be verified before any project asset is changed.",
                    outcome="succeeded",
                    state_changed=True,
                )
            ],
        ),
    }


def _safe_extract(archive_path: Path, destination: Path) -> None:
    with zipfile.ZipFile(archive_path) as archive:
        infos = archive.infolist()
        if len(infos) > MAX_TRANSFER_FILES:
            raise ValueError("Upload archive exceeds the file-count limit.")
        total = 0
        for info in infos:
            pure = PurePosixPath(info.filename)
            if pure.is_absolute() or ".." in pure.parts:
                raise ValueError("Upload archive contains an unsafe path.")
            mode = info.external_attr >> 16
            if stat.S_ISLNK(mode):
                raise ValueError("Upload archives cannot contain symbolic links.")
            total += info.file_size
            if total > MAX_EXPANDED_UPLOAD_BYTES:
                raise ValueError("Upload archive expanded size exceeds the safety limit.")
        destination.mkdir(parents=True, exist_ok=False)
        archive.extractall(destination)


def _load_upload_manifest(context: ProjectContext, transfer_id: str) -> tuple[Path, dict[str, Any]]:
    if not re.fullmatch(r"upload-[a-f0-9]{32}", transfer_id):
        raise ValueError("Invalid upload transfer ID.")
    root = context.root / TRANSFER_ROOT / "uploads" / transfer_id
    try:
        payload = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Upload manifest is missing or invalid: {transfer_id}") from exc
    if not isinstance(payload, dict) or payload.get("transfer_id") != transfer_id:
        raise ValueError("Upload manifest identity mismatch.")
    return root, payload


def commit_upload(
    context: ProjectContext,
    transfer_id: str,
    *,
    provenance: dict[str, Any] | None = None,
) -> dict[str, Any]:
    staging_root, manifest = _load_upload_manifest(context, transfer_id)
    if manifest.get("status") == "committed":
        return {
            **manifest,
            "idempotent": True,
            "action_receipt": action_receipt_payload(
                operation="commit_upload",
                target=transfer_id,
                actions=[
                    ActionRecord(
                        action="Returned an already committed upload",
                        justification="Upload commits are idempotent.",
                        outcome="skipped",
                    )
                ],
            ),
        }
    staging_path = Path(str(manifest["staging_path"]))
    if not staging_path.is_file():
        raise FileNotFoundError("The staged upload payload is incomplete or missing.")
    if staging_path.stat().st_size != int(manifest["size_bytes"]):
        raise ValueError("The staged upload size does not match its preview.")
    if _sha256(staging_path) != manifest["sha256"]:
        raise ValueError("The staged upload checksum does not match its preview.")

    target = context.root / str(manifest["target_relative"])
    target = target.resolve(strict=False)
    if not _within(context.root, target):
        raise ValueError("The upload destination escaped the project root.")
    if target.exists() and not manifest.get("overwrite"):
        raise FileExistsError("The destination appeared after preview; create a new overwrite preview.")

    kind = manifest["item_kind"]
    if kind == "directory":
        prepared = staging_root / "prepared"
        _safe_extract(staging_path, prepared)
    else:
        prepared = staging_root / "prepared.file"
        shutil.copy2(staging_path, prepared)
        if target.suffix.lower() == ".h5ad":
            with prepared.open("rb") as handle:
                if handle.read(len(HDF5_SIGNATURE)) != HDF5_SIGNATURE:
                    raise ValueError("The uploaded .h5ad does not have an HDF5 file signature.")

    target.parent.mkdir(parents=True, exist_ok=True)
    backup_id: str | None = None
    backup_path: Path | None = None
    backup_digest: str | None = None
    if target.exists():
        backup_digest = _path_digest(target)
        backup_id = f"backup-{uuid.uuid4().hex}"
        backup_path = context.root / BACKUP_ROOT / backup_id / str(manifest["target_relative"])
        backup_path.parent.mkdir(parents=True, exist_ok=True)
        os.replace(target, backup_path)
        if not backup_path.exists() or _path_digest(backup_path) != backup_digest:
            if backup_path.exists() and not target.exists():
                os.replace(backup_path, target)
            raise RuntimeError("The original destination could not be verified after backup.")

    try:
        os.replace(prepared, target)
    except Exception:
        if backup_path is not None and backup_path.exists() and not target.exists():
            os.replace(backup_path, target)
        raise

    manifest.update(
        {
            "status": "committed",
            "committed_at": utc_now().isoformat(),
            "backup_id": backup_id,
            "backup_path": str(backup_path) if backup_path else None,
            "provenance": provenance,
        }
    )
    write_json(staging_root / "manifest.json", manifest)
    if backup_id:
        write_json(
            context.root / BACKUP_ROOT / backup_id / "backup.json",
            {
                "schema_version": 1,
                "backup_id": backup_id,
                "created_at": utc_now(),
                "target_relative": manifest["target_relative"],
                "transfer_id": transfer_id,
                "backup_path": str(backup_path),
                "sha256": backup_digest,
            },
        )
    return {
        **manifest,
        "idempotent": False,
        "action_receipt": action_receipt_payload(
            operation="commit_upload",
            target=transfer_id,
            actions=[
                ActionRecord(
                    action="Verified the staged upload size and checksum",
                    justification="Unverified bytes must never enter the project.",
                    outcome="succeeded",
                ),
                ActionRecord(
                    action=(
                        "Backed up the existing destination and atomically committed the replacement"
                        if backup_id
                        else "Atomically committed the new project item"
                    ),
                    justification="Project assets must remain recoverable across replacements.",
                    outcome="succeeded",
                    state_changed=True,
                    evidence=[f"backup_id={backup_id}" if backup_id else "new_destination"],
                ),
            ],
        ),
    }


def list_backups(context: ProjectContext) -> list[dict[str, Any]]:
    root = context.root / BACKUP_ROOT
    backups: list[dict[str, Any]] = []
    if not root.is_dir():
        return backups
    for record in sorted(root.glob("backup-*/backup.json")):
        try:
            payload = json.loads(record.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if isinstance(payload, dict):
            payload.pop("backup_path", None)
            backups.append(payload)
    return backups


def restore_backup(
    context: ProjectContext,
    backup_id: str,
    *,
    dry_run: bool,
    preview_token: str | None = None,
    provenance: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if not re.fullmatch(r"backup-[a-f0-9]{32}", backup_id):
        raise ValueError("Invalid backup ID.")
    record_path = context.root / BACKUP_ROOT / backup_id / "backup.json"
    try:
        record = json.loads(record_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Backup not found or invalid: {backup_id}") from exc
    backup_path = Path(str(record["backup_path"]))
    target = (context.root / str(record["target_relative"])).resolve(strict=False)
    if not _within(context.root, target) or not _within(context.root, backup_path):
        raise ValueError("Backup paths do not resolve inside the registered project.")
    if not backup_path.exists():
        raise FileNotFoundError("The backup payload is no longer present.")
    source_digest = _path_digest(backup_path)
    recorded_digest = record.get("sha256")
    if recorded_digest and source_digest != recorded_digest:
        raise ValueError("The retained backup no longer matches its recorded digest.")
    preview = {
        "schema_version": 1,
        "backup_id": backup_id,
        "target_relative": record["target_relative"],
        "target_exists": target.exists(),
        "backup_kind": "directory" if backup_path.is_dir() else "file",
        "backup_sha256": source_digest,
        "target_sha256": _path_digest(target) if target.exists() else None,
    }
    if dry_run:
        token = make_preview_token({"kind": "restore_backup", **preview})
        return {
            **preview,
            "preview_token": token,
            "preview_expires_in_seconds": 900,
            "action_receipt": action_receipt_payload(
                operation="preview_restore_backup",
                target=backup_id,
                actions=[
                    ActionRecord(
                        action="Verified the selected immutable backup and current destination",
                        justification="Restoration requires a state-bound preview before replacement.",
                        outcome="succeeded",
                    )
                ],
            ),
        }
    if not preview_token:
        raise ValueError("Backup restoration requires a token from restore --dry-run.")
    validate_preview_token(preview_token, {"kind": "restore_backup", **preview})

    prepared_root = context.root / TRANSFER_ROOT / "restores" / f"restore-{uuid.uuid4().hex}"
    prepared_root.mkdir(parents=True, exist_ok=False)
    prepared = prepared_root / "payload"
    if backup_path.is_dir():
        shutil.copytree(backup_path, prepared)
    else:
        shutil.copy2(backup_path, prepared)

    displaced_id: str | None = None
    displaced: Path | None = None
    displaced_digest: str | None = None
    if target.exists():
        displaced_digest = _path_digest(target)
        displaced_id = f"backup-{uuid.uuid4().hex}"
        displaced = context.root / BACKUP_ROOT / displaced_id / str(record["target_relative"])
        displaced.parent.mkdir(parents=True, exist_ok=True)
        os.replace(target, displaced)
        if not displaced.exists() or _path_digest(displaced) != displaced_digest:
            if displaced.exists() and not target.exists():
                os.replace(displaced, target)
            raise RuntimeError("The current destination could not be verified after backup.")
    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        os.replace(prepared, target)
        if _path_digest(target) != source_digest:
            raise RuntimeError("The restored destination failed digest verification.")
    except Exception:
        if displaced is not None and displaced.exists():
            if target.exists():
                failed = prepared_root / "failed-restoration"
                os.replace(target, failed)
            os.replace(displaced, target)
        raise
    if displaced_id and displaced is not None:
        write_json(
            context.root / BACKUP_ROOT / displaced_id / "backup.json",
            {
                "schema_version": 1,
                "backup_id": displaced_id,
                "created_at": utc_now(),
                "target_relative": record["target_relative"],
                "reason": f"Displaced while restoring {backup_id}",
                "backup_path": str(displaced),
                "sha256": displaced_digest,
            },
        )
    return {
        **preview,
        "restored": True,
        "displaced_backup_id": displaced_id,
        "source_backup_retained": backup_path.exists(),
        "provenance": provenance,
        "action_receipt": action_receipt_payload(
            operation="restore_backup",
            target=backup_id,
            actions=[
                ActionRecord(
                    action="Revalidated the backup restoration preview",
                    justification="The backup and destination must not change after preview.",
                    outcome="succeeded",
                ),
                ActionRecord(
                    action="Restored a verified copy while retaining the source backup",
                    justification="Restoration must remain recoverable and must not consume backups.",
                    outcome="succeeded",
                    state_changed=True,
                    evidence=[
                        f"displaced_backup_id={displaced_id}"
                        if displaced_id
                        else "new_destination"
                    ],
                ),
            ],
        ),
    }


__all__ = [
    "BACKUP_ROOT",
    "LARGE_TRANSFER_BYTES",
    "TRANSFER_ROOT",
    "TransferItem",
    "TransferPreview",
    "commit_upload",
    "list_backups",
    "list_transfer_items",
    "prepare_download",
    "prepare_upload",
    "preview_download",
    "preview_upload",
    "resolve_transfer_items",
    "restore_backup",
]
