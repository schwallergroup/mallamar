# Exchange experiment commands and robot status through a shared mailbox.
from __future__ import annotations
from datetime import datetime, timezone
from functools import wraps
import json, logging, os, time
from pathlib import Path
from typing import Any, Dict, Optional
from agent_lg.services.storage import FileLock, atomic_write, fsync_directory
from agent_lg.services.ownership import acquire_lock, release_lock
from agent_lg.services.protocols import normalize_experiment
from agent_lg.services.batches import CommandJournal

logger = logging.getLogger(__name__)


def _utcnow_iso():
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")

def _acquire_lock(lock_path: Path, timeout_s: float = 5.0, poll_s: float = 0.05):
    return acquire_lock(lock_path, timeout_s, poll_s)


def _release_lock(lock_path: Path, owner):
    release_lock(lock_path, owner)


def _write_snapshot_atomic(final_path: Path, lock_path: Path, payload: Dict[str, Any]):
    owner = _acquire_lock(lock_path)
    try:
        atomic_write(final_path, lambda temporary: temporary.write_text(
            json.dumps(payload, ensure_ascii=False, separators=(",", ":")), encoding="utf-8"))
    finally:
        _release_lock(lock_path, owner)


def _append_jsonl(path: Path, payload: Dict[str, Any]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(payload, ensure_ascii=False) + "\n")
        f.flush()
        os.fsync(f.fileno()) # posting evidence must survive a controller crash
    fsync_directory(path.parent)
    fsync_directory(path.parent.parent) # persist a newly created archive directory too

def _read_snapshot(path: Path):
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        return None
    except Exception:
        return None


def exclusive_mailbox(function):
    @wraps(function)
    def exchange(client, *args, **kwargs):
        if os.getenv('MALLAMAR_DRY_RUN'):
            raise RuntimeError('[mailbox] Live robot commands are disabled in dry-run mode; use a fake controller for pipeline tests')
        with FileLock(client.mailbox_root / "optimizer_exchange"):
            return function(client, *args, **kwargs)
    return exchange


class MailboxClient:
    """Owns one mailbox's paths; commands keep the robot's existing message format."""

    def __init__(self, mailbox_root, journal_path=None):
        if not str(mailbox_root).strip():
            raise ValueError("[MailboxClient] Mailbox root must be an explicit path")
        self.mailbox_root = Path(mailbox_root).resolve()
        mailbox_dir = self.mailbox_root / "mailbox"
        self.command_path = mailbox_dir / "optimizer_to_hw.txt"
        self.command_lock = mailbox_dir / "optimizer_to_hw.lock"
        self.command_archive = mailbox_dir / "archive" / "optimizer_to_hw.jsonl"
        self.status_path = mailbox_dir / "hw_to_optimizer.txt"
        self.sequence_path = self.mailbox_root / "optimizer_seq.txt"
        self.journal = CommandJournal(journal_path, self.mailbox_root) if journal_path else None

    @classmethod
    def from_environment(cls, journal_path=None):
        if not os.getenv("OCEAN_MAILBOX_ROOT", "").strip():
            raise EnvironmentError(
                "OCEAN_MAILBOX_ROOT environment variable is not set. "
                "Please set it in your .env file or system environment. "
            )
        return cls(os.environ["OCEAN_MAILBOX_ROOT"], journal_path or os.getenv("MALLAMAR_COMMAND_JOURNAL"))

    def _load_seq(self):
        try:
            sequence = int(self.sequence_path.read_text(encoding="utf-8").strip())
            if sequence < 0:
                raise ValueError("Negative mailbox sequence")
            return sequence
        except FileNotFoundError:
            return 0

    def _save_seq(self, sequence_number: int):
        atomic_write(self.sequence_path, lambda temporary: temporary.write_text(str(sequence_number), encoding="utf-8"))

    def _next_seq(self):
        with FileLock(self.sequence_path):
            sequence_number = self._load_seq() + 1
            self._save_seq(sequence_number)
            return sequence_number

    def _post_or_recover(self, key, payload):
        command = self.journal.get(key) if self.journal else None
        if command:
            original = command['payload']
            if {k: v for k, v in original.items() if k not in ('seq', 'ts')} != payload:
                raise RuntimeError('[mailbox] Command changed after submission; inspect the original batch')
            if command.get('state') == 'terminal':
                return original['seq'], command['result']
            if command.get('state') == 'prepared':
                snapshot = _read_snapshot(self.command_path)
                evidence = snapshot == original
                if not evidence and self.command_archive.exists():
                    evidence = any(json.loads(line) == original for line in self.command_archive.read_text(encoding='utf-8').splitlines() if line.strip())
                status = _read_snapshot(self.status_path)
                evidence = evidence or (isinstance(status, dict) and status.get('seq') == original['seq'])
                if not evidence:
                    raise RuntimeError('[mailbox] Crash at prepare/post boundary: verify hardware before clearing this command as never posted')
                command['state'] = 'posted'
                self.journal.save(key, command)
            if command.get('state') == 'posted':
                return original['seq'], None # wait for the original seq; do not post it again
            if command.get('state') != 'not_posted':
                raise RuntimeError('[mailbox] Unknown command state; inspect the journal')
            payload = original # operator confirmed that this prepared command was never posted
        else:
            if self.journal:
                recorded = [value['payload']['seq'] for value in self.journal.load()['commands'].values()]
                if recorded and max(recorded) > self._load_seq():
                    raise RuntimeError('[mailbox] Sequence file is behind saved commands; reconcile it before posting')
            sequence_number = self._next_seq()
            timestamp = _utcnow_iso()
            payload = {'seq': sequence_number, 'sender': payload['sender'], 'ts': timestamp,
                       **{k: v for k, v in payload.items() if k != 'sender'}}
            command = {'state': 'prepared', 'payload': payload}
            if self.journal:
                self.journal.save(key, command)
        _write_snapshot_atomic(self.command_path, self.command_lock, payload)
        _append_jsonl(self.command_archive, payload)
        if self.journal:
            command['state'] = 'posted'
            self.journal.save(key, command)
        return payload['seq'], None

    def _record_terminal(self, key, sequence_number, status):
        state = str(status.get('state', '')).lower()
        result = {'ok': state == 'finished' and (status.get('result') or {}).get('status') == 'ok',
                  'seq': sequence_number, 'state': state,
                  'exp_dir': status.get('exp_dir'), 'result': status.get('result') or {}}
        if self.journal:
            command = self.journal.get(key)
            command.update(state='terminal', result=result)
            self.journal.save(key, command)
        return result

    def _status_for_sequence(self, sequence_number):
        snapshot = _read_snapshot(self.status_path)
        if snapshot is not None:
            if (not isinstance(snapshot, dict) or isinstance(snapshot.get('seq'), bool)
                    or not isinstance(snapshot.get('seq'), int) or snapshot['seq'] < 0
                    or not isinstance(snapshot.get('state'), str)
                    or (snapshot.get('progress') is not None and not isinstance(snapshot['progress'], dict))
                    or (snapshot.get('result') is not None and not isinstance(snapshot['result'], dict))):
                return None # malformed snapshots cannot finish a command or renew its heartbeat
        if isinstance(snapshot, dict) and isinstance(snapshot.get('seq'), int) and snapshot['seq'] > sequence_number and self.journal:
            raise RuntimeError('[mailbox] Hardware has advanced beyond an unresolved command; reconcile its terminal evidence before continuing')
        return snapshot

    @exclusive_mailbox
    def run_experiment(self, exp: dict, index: int):
        """Send concentrations and wait for a matching robot result or stalled heartbeat."""
        exp = normalize_experiment(exp) # reject invalid concentrations and builder hints before posting
        if isinstance(index, bool) or int(index) < 1 or float(index) != int(index):
            raise ValueError("[MailboxClient] Experiment index must be a positive integer")
        # Build command payload (NO protocol path here)
        command_payload = {
            "sender": "optimizer",
            "command": "run_concentrations", # key change
            "index": int(index),
            "requires_init": True, # HW will run 00_INIT before the protocol
            "concentrations": {
                "MA": exp.get("conc_MA_M"),
                "NaBrO3": exp.get("conc_NaBrO3_M"),
                "H2SO4": exp.get("conc_H2SO4_M"),
                "Ru": exp.get("conc_Ru_M"),
            },
            # optional HW builder hints (all optional)
            "builder_params": {
                "target_volume_uL": exp.get("target_volume_uL"),
                "set_temperature_C": exp.get("set_temperature_C"),
                "stir_rpm": exp.get("stir_rpm"),
                "run_duration_s": exp.get("run_duration_s"),
                "run_step_max_s": exp.get("run_step_max_s")
            }
        }

        key = f'experiment:{int(index)}'
        sequence_number, cached = self._post_or_recover(key, command_payload)
        if cached is not None:
            return cached
        print(f"[BO] Waiting for seq={sequence_number} idx={index}")

        # Wait for completion; only missing heartbeats trigger a timeout.
        # The controller must update status regularly, even during week-long runs.
        # On timeout, the robot may still be running—verify before retrying.

        poll_s = 1.0
        stall_timeout_s = float(exp.get("stall_timeout_s") or 172800.0) # 2 days for long BZ experiments
        last_beat = time.monotonic()
        last_sig = None
        last_status = None

        while True:
            status_snapshot = self._status_for_sequence(sequence_number)
            if status_snapshot and isinstance(status_snapshot, dict) and status_snapshot.get("seq") == sequence_number:
                last_status = status_snapshot
                state = str(status_snapshot.get("state", "")).lower()
                if state in ("finished", "failed"):
                    return self._record_terminal(key, sequence_number, status_snapshot)
                elif state in ("accepted", "running"):
                    progress = status_snapshot.get("progress", {}) or {}
                    if progress:
                        print(f"[BO] seq={sequence_number} {state}: step {progress.get('step')}/{progress.get('total_steps')} {progress.get('name')}")

            # Heartbeat: a change in seq/state/progress OR in the snapshot's mtime
            # means the robot is alive — reset the stall clock.
            try:
                mtime = self.status_path.stat().st_mtime
            except Exception:
                mtime = None
            heartbeat_signature = (
                status_snapshot.get("seq") if isinstance(status_snapshot, dict) else None,
                status_snapshot.get("state") if isinstance(status_snapshot, dict) else None,
                repr(status_snapshot.get("progress")) if isinstance(status_snapshot, dict) else None,
                mtime,
            )
            if (isinstance(status_snapshot, dict) and status_snapshot.get("seq") == sequence_number
                    and str(status_snapshot.get("state", "")).lower() in ("accepted", "running") and heartbeat_signature != last_sig):
                last_sig, last_beat = heartbeat_signature, time.monotonic()
            if time.monotonic() - last_beat > stall_timeout_s:
                raise TimeoutError(
                    f"No robot heartbeat for {stall_timeout_s:.0f}s (seq={sequence_number}); "
                    f"the robot may STILL be running — do NOT assume failure or mock mode. "
                    f"last_state={last_status}"
                )
            time.sleep(poll_s)

    @exclusive_mailbox
    def send_end_batch(self, reason: str = "optimization_complete"):
        """Request robot cleanup and wait for its completion heartbeat."""
        # Build end_batch command payload
        command_payload = {
            "sender": "optimizer",
            "command": "end_batch",
            "reason": reason
        }

        key = 'end_batch'
        sequence_number, cached = self._post_or_recover(key, command_payload)
        if cached is not None:
            return cached
        print(f"[BO] Waiting for end_batch seq={sequence_number}")

        # Wait for HW to finish the END protocol
        # END protocol takes ~12-15 minutes, but allow up to 1 day for safety
        poll_s = 2.0
        stall_timeout_s = 86400.0 # 1 day for END protocol
        last_beat = time.monotonic()
        last_sig = None
        last_status = None

        while True:
            status_snapshot = self._status_for_sequence(sequence_number)
            if status_snapshot and isinstance(status_snapshot, dict) and status_snapshot.get("seq") == sequence_number:
                last_status = status_snapshot
                state = str(status_snapshot.get("state", "")).lower()
                if state in ("finished", "failed"):
                    result = status_snapshot.get("result", {}) or {}
                    print(f"[BO] end_batch {state}: {result.get('message', '')}")
                    return self._record_terminal(key, sequence_number, status_snapshot)
                elif state == "accepted":
                    print(f"[BO] end_batch accepted, waiting for END protocol to complete...")
                elif state == "running":
                    progress = status_snapshot.get("progress", {}) or {}
                    if progress:
                        print(f"[BO] END protocol running: step {progress.get('step')}/{progress.get('total_steps')} {progress.get('name')}")

            # Heartbeat: check for changes in state or file modification time
            try:
                mtime = self.status_path.stat().st_mtime
            except Exception:
                mtime = None
            heartbeat_signature = (
                status_snapshot.get("seq") if isinstance(status_snapshot, dict) else None,
                status_snapshot.get("state") if isinstance(status_snapshot, dict) else None,
                repr(status_snapshot.get("progress")) if isinstance(status_snapshot, dict) else None,
                mtime,
            )
            if (isinstance(status_snapshot, dict) and status_snapshot.get("seq") == sequence_number
                    and str(status_snapshot.get("state", "")).lower() in ("accepted", "running") and heartbeat_signature != last_sig):
                last_sig, last_beat = heartbeat_signature, time.monotonic()
            if time.monotonic() - last_beat > stall_timeout_s:
                raise TimeoutError(
                    f"No robot heartbeat for {stall_timeout_s:.0f}s during end_batch (seq={sequence_number}); "
                    f"the robot may STILL be running the END protocol. "
                    f"last_state={last_status}"
                )
            time.sleep(poll_s)


def run_experiment(exp: dict, index: int):
    """Run one experiment using the mailbox configured for this call."""
    return MailboxClient.from_environment().run_experiment(exp, index)


def send_end_batch(reason: str = "optimization_complete", journal_path=None):
    """Request robot cleanup using the mailbox configured for this call."""
    return MailboxClient.from_environment(journal_path=journal_path).send_end_batch(reason)
