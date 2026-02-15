# Runtime hook for PyInstaller.
# Ensures the app has a writable per-user working directory and profile folder.
import os
import sys

APP_NAME = (os.environ.get("PTZ_APP_NAME") or "PTZoptics_JS").strip() or "PTZoptics_JS"

def _user_data_dir(app_name: str) -> str:
    env_dir = (os.environ.get("PTZ_APPDATA_DIR") or "").strip()
    if env_dir:
        p = os.path.abspath(env_dir)
        os.makedirs(p, exist_ok=True)
        return p

    home = os.path.expanduser("~")
    if sys.platform == "darwin":
        base = os.path.join(home, "Library", "Application Support")
    elif os.name == "nt":
        base = os.environ.get("APPDATA") or os.path.join(home, "AppData", "Roaming")
    else:
        base = os.environ.get("XDG_DATA_HOME") or os.path.join(home, ".local", "share")

    p = os.path.join(base, app_name)
    os.makedirs(p, exist_ok=True)
    return p

try:
    appdata = _user_data_dir(APP_NAME)
    os.environ.setdefault("PTZ_APPDATA_DIR", appdata)

    profiles = (os.environ.get("PTZ_PROFILES_DIR") or "").strip()
    if not profiles:
        profiles = os.path.join(appdata, "profiles")
        os.environ.setdefault("PTZ_PROFILES_DIR", profiles)
    os.makedirs(profiles, exist_ok=True)

    # Make relative writes land in the per-user dir.
    try:
        os.chdir(appdata)
    except Exception:
        pass
except Exception:
    pass
