# -*- mode: python ; coding: utf-8 -*-

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
    ['main1.py'],
    pathex=['/Users/sekkevin/Library/CloudStorage/OneDrive-PeterMac/Documents/DCC/PTZoptics_JS'],
    binaries=pygame_binaries,
    datas=[('PTZOptics-VISCA-over-IP-Rev-1_2-8-20.pdf', '.'), ('assets', 'assets')] + pygame_datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=['/Users/sekkevin/Library/CloudStorage/OneDrive-PeterMac/Documents/DCC/PTZoptics_JS/runtime_hook_appdata.py'],
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
    name='PTZoptics_JS',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,
    icon='assets/app.icns',
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)

if sys.platform == "darwin":
    app = BUNDLE(
        exe,
        name=f"PTZoptics_JS.app",
        icon='assets/app.icns',
        bundle_identifier=None,
    )
else:
    app = exe
