# Build helper for Windows PowerShell.
# Run from project root: .\build_windows.ps1

if (!(Test-Path ".\.venv")) {
  py -m venv .venv
}

#.\.venv\Scripts\Activate.ps1

python -m pip install -U pip
python -m pip install -U pyinstaller pyside6

python build.py --name PTZoptics_JS --entry main1.py
