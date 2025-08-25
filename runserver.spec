# -*- mode: python ; coding: utf-8 -*-
from PyInstaller.utils.hooks import collect_submodules, collect_all

datas = [
    ('PPE_model.pt', '.'),
    ('yolov8n-pose.pt', '.'),
    ('fire_smoke_model.pt', '.'),
    ('staticfiles', 'staticfiles'),
    ('camera_streaming', 'camera_streaming'),
]

binaries = []
hiddenimports = []

# your apps and migrations
hiddenimports += collect_submodules('camera')
hiddenimports += collect_submodules('camera.migrations')
hiddenimports += collect_submodules('users')
hiddenimports += collect_submodules('users.migrations')
hiddenimports += collect_submodules('event')
hiddenimports += collect_submodules('event.migrations')

# libs you use
hiddenimports += collect_submodules('celery')
hiddenimports += collect_submodules('rest_framework_simplejwt')
hiddenimports += collect_submodules('shapely')
hiddenimports += collect_submodules('drf_yasg')

# include package data (templates/static/etc.)
for pkg in ['django', 'channels', 'uvicorn', 'starlette', 'celery', 'drf_yasg']:
    tmp = collect_all(pkg)
    datas += tmp[0]; binaries += tmp[1]; hiddenimports += tmp[2]

a = Analysis(
    ['runserver.py'],
    pathex=[],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],  # not needed; runserver.py handles runtime migrations
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
