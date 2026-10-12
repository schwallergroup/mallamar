# Capture campaign output without hiding console errors.

import os, sys, logging
from datetime import datetime
from pathlib import Path

logger = logging.getLogger(__name__)


class TeeOutput:
    """Duplicate stdout/stderr to both console and a log file."""
    def __init__(self, file_path, original_stream):
        self.file = open(file_path, 'a', encoding='utf-8')
        self.original = original_stream
        self._file_usable = True

    def _disable_file(self):
        """Stop using a broken log stream without disturbing the console."""
        self._file_usable = False
        try:
            self.file.close()
        except (OSError, ValueError):
            pass

    def write(self, data):
        written = self.original.write(data)
        if self._file_usable:
            try:
                self.file.write(data)
                self.file.flush()
            except (OSError, ValueError):
                # Logging must never hide the exception that the agent is trying
                # to report (notably Windows redirected-stream errno 22 errors).
                self._disable_file()
        return written

    def flush(self):
        try:
            self.original.flush()
        except (OSError, ValueError):
            pass
        if self._file_usable:
            try:
                self.file.flush()
            except (OSError, ValueError):
                self._disable_file()

    def close(self):
        if self._file_usable:
            try:
                self.file.close()
            except (OSError, ValueError):
                pass
        self._file_usable = False



def setup_logging(db_path: str):
    """Capture campaign output while retaining the original console streams."""
    db_path_obj = Path(db_path)
    campaign_name = db_path_obj.parent.name if db_path_obj.parent.name else "campaign"

    # Keep active logs on a local disk when OCEAN_LOG_DIR is provided. The
    # mailbox may live on a mapped/network drive whose file handles can become
    # temporarily invalid even while the experiment mailbox remains usable.
    if "OCEAN_LOG_DIR" in os.environ:
        log_dir = Path(os.environ["OCEAN_LOG_DIR"])
    elif "OCEAN_MAILBOX_ROOT" in os.environ:
        log_dir = Path(os.environ["OCEAN_MAILBOX_ROOT"]) / "logs"
    else:
        log_dir = Path.cwd() / "logs"

    log_dir = log_dir.expanduser().resolve()
    log_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    log_file = log_dir / f"logs_{campaign_name}_{timestamp}.txt"

    with open(log_file, 'x', encoding='utf-8') as f:
        f.write("Mallamar Agent Log\n")
        f.write(f"Campaign: {campaign_name}\n")
        f.write(f"Database: {db_path}\n")
        f.write(f"Started: {datetime.now().isoformat()}\n")
        f.write("\n")

    original_stdout = sys.stdout
    original_stderr = sys.stderr

    stdout = TeeOutput(log_file, original_stdout)
    try:
        stderr = TeeOutput(log_file, original_stderr)
    except BaseException:
        stdout.close()
        raise
    sys.stdout, sys.stderr = stdout, stderr

    print(f"[Logging] Console output will be saved to: {log_file}")

    return str(log_file), original_stdout, original_stderr



def cleanup_logging(original_stdout, original_stderr):
    """Restore original stdout/stderr and close log files."""
    tee_stdout = sys.stdout
    tee_stderr = sys.stderr
    # Restore first: even if closing a redirected Windows stream fails, later
    # exception reporting still has valid stdout/stderr handles.
    sys.stdout = original_stdout
    sys.stderr = original_stderr
    for stream in (tee_stdout, tee_stderr):
        if stream in (original_stdout, original_stderr):
            continue
        try:
            stream.close()
        except (OSError, ValueError):
            pass

