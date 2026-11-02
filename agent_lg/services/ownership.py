# Record who holds a lock; never infer that the hardware stopped from controller death.
import json, logging, math, os, socket, sys, time, uuid
from pathlib import Path

logger = logging.getLogger(__name__)


def process_start(pid):
    path = Path('/proc') / str(pid) / 'stat'
    try:
        return path.read_text().rsplit(')', 1)[1].split()[19] # Linux start ticks distinguish reused PIDs
    except (OSError, IndexError):
        return None


def owner_record():
    return {'version': 1, 'token': uuid.uuid4().hex, 'hostname': socket.gethostname(),
            'pid': os.getpid(), 'parent_pid': os.getppid(), 'process_started': process_start(os.getpid()),
            'executable': sys.executable, 'worker': Path(sys.argv[0]).name,
            'created_at': time.time()} # wall time is only for diagnostics


def lock_status(path):
    path = Path(path)
    report = {'path': str(path), 'state': 'missing'}
    try:
        owner = json.loads(path.read_text(encoding='utf-8'))
        if not isinstance(owner, dict) or owner.get('version') != 1:
            raise ValueError('Unknown lock format')
        report.update(owner=owner, state='unverified')
        if owner.get('hostname') != socket.gethostname():
            return report # remote processes cannot be checked locally
        pid = owner.get('pid')
        if isinstance(pid, bool) or not isinstance(pid, int) or pid <= 0:
            raise ValueError('Invalid lock PID')
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            report['state'] = 'owner_exited'
        except (PermissionError, OSError):
            report['state'] = 'unverified'
        else:
            started = process_start(pid)
            if started and owner.get('process_started'):
                report['state'] = 'owner_active' if started == owner['process_started'] else 'pid_reused'
            else:
                report['state'] = 'pid_exists_unverified'
    except FileNotFoundError:
        pass
    except (OSError, ValueError, TypeError) as error:
        report.update(state='unverified', error=str(error))
    return report


def acquire_lock(path, timeout_s=5, poll_s=0.05):
    try:
        if isinstance(timeout_s, bool) or isinstance(poll_s, bool):
            raise ValueError()
        timeout_s, poll_s = float(timeout_s), float(poll_s)
        if not math.isfinite(timeout_s) or timeout_s < 0 or not math.isfinite(poll_s) or poll_s <= 0:
            raise ValueError()
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError('[FileLock] Timeout must be finite and non-negative; polling interval must be finite and positive') from error
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    deadline = time.monotonic() + timeout_s
    while True:
        try:
            descriptor = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
        except FileExistsError:
            if time.monotonic() >= deadline:
                owner = lock_status(path)
                raise TimeoutError(f'[FileLock] File is busy: {path}; owner={owner}; verify controller, workers and hardware before removing a stale lock')
            time.sleep(poll_s)
            continue
        owner = owner_record()
        try:
            with os.fdopen(descriptor, 'w', encoding='utf-8') as stream:
                json.dump(owner, stream)
                stream.flush()
                os.fsync(stream.fileno())
        except BaseException:
            path.unlink(missing_ok=True)
            raise
        return owner


def release_lock(path, owner):
    path = Path(path)
    try:
        saved = json.loads(path.read_text(encoding='utf-8'))
        if isinstance(saved, dict) and saved.get('token') == owner['token']:
            path.unlink() # do not remove another process's replacement lock
        else:
            logger.warning(f'[FileLock] Ownership changed; retained {path}')
    except FileNotFoundError:
        pass
    except (OSError, ValueError, TypeError) as error:
        logger.warning(f'[FileLock] Could not verify release of {path}: {error}')
