"""Keep pytest's temporary files under the writable project workspace on Windows."""
import shutil
import tempfile
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RUN_TEMP = Path(tempfile.mkdtemp(prefix=".pytest-temp-run-", dir=PROJECT_ROOT))
tempfile.tempdir = str(RUN_TEMP)


def pytest_sessionfinish(session, exitstatus):
    """Remove only this test run's temporary directory, never stale user data."""
    if RUN_TEMP.parent == PROJECT_ROOT and RUN_TEMP.name.startswith(".pytest-temp-run-"):
        shutil.rmtree(RUN_TEMP, ignore_errors=True)
