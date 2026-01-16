# -*- mode: python ; coding: utf-8 -*-

import sys
import os
from pathlib import Path

project_root = Path('.').absolute().parent
app_path = project_root / 'app'

# Get SSL DLLs from the actual Python installation (uv managed)
base_python_dlls = Path(r'C:\Users\Hugo\AppData\Roaming\uv\python\cpython-3.13.8-windows-x86_64-none\DLLs')

ssl_dlls = []

# Add the SSL files from the base Python installation
ssl_files = [
    '_ssl.pyd',
    '_hashlib.pyd',
    'libssl-3-x64.dll',
    'libcrypto-3-x64.dll'
]

print(f"Looking for SSL files in: {base_python_dlls}")

for ssl_file in ssl_files:
    ssl_path = base_python_dlls / ssl_file
    if ssl_path.exists():
        ssl_dlls.append((str(ssl_path), '.'))
        print(f"Including: {ssl_path}")
    else:
        print(f"Not found: {ssl_path}")

print(f"Total SSL files to include: {len(ssl_dlls)}")

a = Analysis(
    [str(app_path / 'run.py')],
    pathex=[str(app_path)],
    binaries=ssl_dlls,
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
    [],
    exclude_binaries=True,
    name='EhrenreichSegmenter',
    debug=True,
    console=True,
    cipher=None,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='EhrenreichSegmenter',
    distpath='dist-win'
)