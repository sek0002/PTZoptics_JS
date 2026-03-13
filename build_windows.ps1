# Build helper for Windows PowerShell.
# Run from project root: .\build_windows.ps1

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

function Get-HostPython {
  $pyLauncher = Get-Command py -ErrorAction SilentlyContinue
  if ($pyLauncher) { return @{ Cmd = "py"; Args = @("-3") } }
  $pythonCmd = Get-Command python -ErrorAction SilentlyContinue
  if ($pythonCmd) { return @{ Cmd = "python"; Args = @() } }
  throw "Could not find Python launcher ('py') or 'python' on PATH."
}

$venvDir = ".\.venv"
$venvPy = Join-Path $venvDir "Scripts\python.exe"

if (!(Test-Path $venvPy)) {
  $hostPy = Get-HostPython
  & $hostPy.Cmd @($hostPy.Args) -m venv $venvDir
}

if (!(Test-Path $venvPy)) {
  throw "Could not create/find venv python at $venvPy"
}

$py = $venvPy

& $py -m pip install -U pip setuptools wheel packaging
& $py -m pip install -U --force-reinstall pyinstaller pyinstaller-hooks-contrib pyside6

# Guard against broken/empty PySide6 package metadata, which causes
# PyInstaller hook-PySide6 to fail with TypeError in packaging.version.Version(None).
$ok = & $py -c "import importlib.metadata as m; v=m.version('PySide6'); print('PySide6', v); raise SystemExit(0 if isinstance(v,str) and v.strip() else 1)" 2>$null
if ($LASTEXITCODE -ne 0) {
  Write-Host "PySide6 metadata invalid; forcing reinstall once more..."
  & $py -m pip uninstall -y pyside6 pyside6-addons pyside6-essentials shiboken6
  & $py -m pip install -U --force-reinstall pyside6
}

& $py build.py --name PTZoptics_JS --entry main1.py
