# -*- mode: python ; coding: utf-8 -*-
from PyInstaller.utils.hooks import collect_submodules
from PyInstaller.utils.hooks import collect_all

datas = [
    ('PPE_model.pt', '.'),
    ('yolov8n-pose.pt', '.'),
    ('fire_smoke_model.pt', '.'),
    ('staticfiles', 'staticfiles'),
    ('camera_streaming', 'camera_streaming')
]
binaries = []
hiddenimports = []

# Custom submodules
hiddenimports += collect_submodules('celery')
hiddenimports += collect_submodules('camera')
hiddenimports += collect_submodules('camera.management.commands')
hiddenimports += collect_submodules('rest_framework_simplejwt')

# Django
tmp = collect_all('django')
datas += tmp[0]; binaries += tmp[1]; hiddenimports += tmp[2]

# Channels
tmp = collect_all('channels')
datas += tmp[0]; binaries += tmp[1]; hiddenimports += tmp[2]

# Uvicorn
tmp = collect_all('uvicorn')
datas += tmp[0]; binaries += tmp[1]; hiddenimports += tmp[2]

# Starlette
tmp = collect_all('starlette')
datas += tmp[0]; binaries += tmp[1]; hiddenimports += tmp[2]

# Celery again (no harm)
tmp = collect_all('celery')
datas += tmp[0]; binaries += tmp[1]; hiddenimports += tmp[2]

# ✅ Add drf_yasg templates & dependencies
tmp = collect_all('drf_yasg')
datas += tmp[0]; binaries += tmp[1]; hiddenimports += tmp[2]

a = Analysis(
    ['runserver.py'],
    pathex=[],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    [],
    name='runserver',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
