# -*- mode: python ; coding: utf-8 -*-

import sys
import os
from pathlib import Path

project_root = Path('.').absolute().parent
app_path = project_root / 'app'

# macOS-specific configuration
# No need for Windows-specific SSL DLL handling on macOS
# macOS handles SSL libraries differently through the system

a = Analysis(
    [str(app_path / 'run.py')],
    pathex=[str(app_path)],
    binaries=[],
    datas=[
        (str(app_path / 'cache'), 'cache'),
    ],
    hiddenimports=[
        'PyQt6.QtCore',
        'PyQt6.QtGui', 
        'PyQt6.QtWidgets',
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    cipher=None,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=None)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.zipfiles,
    a.datas,
    name='EhrenreichSegmenter',
    console=False,  # Set to False for macOS GUI app
)

# Create macOS app bundle
app = BUNDLE(
    exe,
    name="EhrenreichSegmenter.app",
    bundle_identifier="com.demule.ehrenreichsegmenter",
)