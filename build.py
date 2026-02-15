#!/usr/bin/env python3
"""
PyInstaller build script for PySide6 + pygame + icons, with a runtime hook that forces
a writable per-user working directory (helps profile persistence in packaged apps).

What it does:
- Generates <name>.spec and builds from it
- Bundles pygame (datas + dynamic libs + hidden imports)
- Adds optional Qt hidden imports
- Embeds icon inside the spec (no --icon CLI)
- Adds a runtime hook that:
    - creates a per-user app data dir
    - sets env vars: PTZ_APPDATA_DIR and PTZ_PROFILES_DIR
    - os.chdir() into PTZ_APPDATA_DIR (so relative saves persist)

Usage:
  python build_pyside_pygame_profiles.py --name PTZoptics_JS --entry main1.py --console
"""

from __future__ import annotations
import argparse
import os
import sys
import subprocess
from pathlib import Path


def run(cmd: list[str]) -> None:
    print(">>", " ".join(cmd))
    subprocess.check_call(cmd)


def guess_icon() -> str | None:
    assets = Path("assets")
    if sys.platform == "darwin":
        p = assets / "app.icns"
        if p.exists():
            return str(p)
    if os.name == "nt":
        p = assets / "app.ico"
        if p.exists():
            return str(p)
    for candidate in ["app.icns", "app.ico"]:
        if Path(candidate).exists():
            return candidate
    return None


def collect_datas() -> list[tuple[str, str]]:
    datas: list[tuple[str, str]] = []
    for fname in [
        "PTZOptics-VISCA-over-IP-Rev-1_2-8-20.pdf",
        "visca_targets.json",
        "_meta.json",
        "profiles.json",
        "default_profiles.json",
    ]:
        p = Path(fname)
        if p.exists():
            datas.append((str(p), "."))

    for folder in ["assets", "audio"]:
        d = Path(folder)
        if d.exists() and d.is_dir():
            datas.append((str(d), folder))
    return datas


SPEC_TEMPLATE = r"""# -*- mode: python ; coding: utf-8 -*-

import sys
from PyInstaller.utils.hooks import (
    collect_submodules,
    collect_data_files,
    collect_dynamic_libs,
)

block_cipher = None

hiddenimports = []

# Optional Qt modules
hiddenimports += collect_submodules("PySide6.QtCharts")
hiddenimports += collect_submodules("PySide6.QtMultimedia")
hiddenimports += collect_submodules("PySide6.QtMultimediaWidgets")

# pygame
hiddenimports += collect_submodules("pygame")
pygame_datas = collect_data_files("pygame")
pygame_binaries = collect_dynamic_libs("pygame")

a = Analysis(
    [{entry!r}],
    pathex=[{pathex!r}],
    binaries=pygame_binaries,
    datas={datas!r} + pygame_datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={{}},
    runtime_hooks=[{runtime_hook!r}],
    excludes=[],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.zipfiles,
    a.datas,
    [],
    name={name!r},
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console={console},
    icon={icon!r},
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)

if sys.platform == "darwin":
    app = BUNDLE(
        exe,
        name=f"{name}.app",
        icon={icon!r},
        bundle_identifier=None,
    )
else:
    app = exe
"""


def write_spec(name: str, entry: str, console: bool, icon: str | None, runtime_hook: str) -> Path:
    entry_path = Path(entry)
    if not entry_path.exists():
        raise SystemExit(f"Entry file not found: {entry}")

    datas = collect_datas()

    spec_text = SPEC_TEMPLATE.format(
        name=name,
        entry=str(entry_path),
        pathex=str(Path.cwd()),
        datas=datas,
        console="True" if console else "False",
        icon=icon,
        runtime_hook=runtime_hook,
    )
    spec_path = Path(f"{name}.spec")
    spec_path.write_text(spec_text, encoding="utf-8")

    print(f"Spec created: {spec_path}")
    print(f"Runtime hook: {runtime_hook}")
    if icon:
        print(f"Icon in spec: {icon}")
    if datas:
        print("Extra bundled datas:")
        for src, dest in datas:
            print(f"  - {src} -> {dest}")
    return spec_path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="MyApp")
    ap.add_argument("--entry", default="main1.py")
    ap.add_argument("--console", action="store_true")
    ap.add_argument("--onefile", action="store_true")
    args = ap.parse_args()

    # Sanity: pygame should import if you expect it bundled
    try:
        import pygame  # noqa: F401
        print("pygame import OK")
    except Exception as e:
        print(f"WARNING: pygame not importable here: {e}")
        print("If you need pygame, install into this environment: python -m pip install pygame")

    icon = guess_icon()

    runtime_hook = str(Path("runtime_hook_appdata.py").resolve())
    if not Path(runtime_hook).exists():
        raise SystemExit("Missing runtime_hook_appdata.py. Keep it next to this build script.")

    spec_path = write_spec(args.name, args.entry, args.console, icon, runtime_hook)

    cmd = [sys.executable, "-m", "PyInstaller", "--noconfirm", "--clean", str(spec_path)]
    if args.onefile:
        cmd.insert(4, "--onefile")

    run(cmd)
    print("\nDone. Output is in dist/\n")
    print("Env vars set at runtime:")
    print("  PTZ_APPDATA_DIR  (cwd is changed to this dir)")
    print("  PTZ_PROFILES_DIR (PTZ_APPDATA_DIR/profiles)\n")


if __name__ == "__main__":
    main()
