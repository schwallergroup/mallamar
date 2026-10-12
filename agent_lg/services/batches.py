# Keep submission evidence outside graph checkpoints so a crash cannot silently resubmit.
import hashlib, json, logging
from pathlib import Path
from agent_lg.services.storage import FileLock, write_json_atomic
from agent_lg.services.protocols import load_protocol

logger = logging.getLogger(__name__)


class CommandJournal:
    """Durable command identity and terminal robot evidence for one batch."""

    def __init__(self, path, mailbox_root):
        self.path = Path(path).resolve()
        self.mailbox_root = str(Path(mailbox_root).resolve())

    def load(self):
        if not self.path.exists():
            return {'version': 1, 'mailbox_root': self.mailbox_root, 'commands': {}}
        record = json.loads(self.path.read_text(encoding='utf-8'))
        if not isinstance(record, dict) or record.get('version') != 1 or record.get('mailbox_root') != self.mailbox_root or not isinstance(record.get('commands'), dict):
            raise ValueError('[CommandJournal] Invalid record or different hardware mailbox; original preserved')
        return record

    def bind_protocol(self, digest):
        with FileLock(self.path):
            record = self.load()
            if record.get('protocol_sha256', digest) != digest:
                raise ValueError('[CommandJournal] Protocol changed since the original submission')
            record['protocol_sha256'] = digest
            write_json_atomic(self.path, record)

    def get(self, key):
        return self.load()['commands'].get(key)

    def save(self, key, command):
        with FileLock(self.path):
            record = self.load()
            previous = record['commands'].get(key)
            if previous and previous['payload'] != command['payload']:
                raise ValueError('[CommandJournal] Command identity changed; inspect its original sequence')
            record['commands'][key] = command
            write_json_atomic(self.path, record)

    def confirm_terminal(self, key, result, evidence):
        """Operator reconciliation when matching terminal hardware evidence is unavailable in the mailbox."""
        if not isinstance(evidence, str) or not evidence.strip():
            raise ValueError('[CommandJournal] Record the verified hardware evidence')
        command = self.get(key)
        if not command or result.get('seq') != command['payload']['seq'] or result.get('state') not in ('finished', 'failed'):
            raise ValueError('[CommandJournal] Terminal evidence must match the original command sequence')
        expected_ok = result['state'] == 'finished' and (result.get('result') or {}).get('status') == 'ok'
        if result.get('ok') is not expected_ok:
            raise ValueError('[CommandJournal] Terminal state and result disagree')
        command.update(state='terminal', result=result, recovery_evidence=evidence)
        self.save(key, command)

    def confirm_not_posted(self, key, evidence):
        """Explicit operator clearance for the ambiguous prepare/post boundary."""
        if not isinstance(evidence, str) or not evidence.strip():
            raise ValueError('[CommandJournal] Record why hardware could not have executed this command')
        command = self.get(key)
        if not command or command['state'] != 'prepared':
            raise ValueError('[CommandJournal] Only a prepared command can be cleared as never posted')
        command.update(state='not_posted', recovery_evidence=evidence)
        self.save(key, command)


class BatchJournal:
    """Persist one batch's submission and collection without changing robot messages."""

    def __init__(self, db_path, batch_id):
        self.batch_id = batch_id
        self.db_path = str(Path(db_path).resolve())
        identifier = hashlib.sha256(batch_id.encode()).hexdigest()
        self.path = Path(self.db_path).parent / ".mallamar" / "batches" / (identifier + ".json")

    @property
    def command_path(self):
        return self.path.with_suffix('.commands.json')

    def load(self):
        if not self.path.exists():
            return {}
        record = json.loads(self.path.read_text(encoding="utf-8"))
        if not isinstance(record, dict) or record.get("batch_id") != self.batch_id or record.get("db_path") != self.db_path:
            raise ValueError("[BatchJournal] Batch record does not match this campaign")
        return record

    def save(self, record):
        write_json_atomic(self.path, {**record, "batch_id": self.batch_id, "db_path": self.db_path})

    def begin_submission(self, json_path, existing_ids):
        digest = hashlib.sha256(Path(json_path).read_bytes()).hexdigest()
        with FileLock(self.path):
            record = self.load()
            if record:
                if record.get("protocol_sha256") != digest:
                    raise RuntimeError("[BatchJournal] Submitted batch has changed; inspect the original protocol")
                if record.get("state") in ("submitted", "collected", "committed"):
                    return record["submission_result"]
                if record.get('state') in ('submitting', 'uncertain') and self.command_path.is_file():
                    ledger = json.loads(self.command_path.read_text(encoding='utf-8'))
                    journal = CommandJournal(self.command_path, ledger['mailbox_root'])
                    if journal.load().get('protocol_sha256') != digest:
                        raise RuntimeError('[BatchJournal] Command journal does not match the original protocol')
                    return None # worker will recover each original command, including unfinished ones
                raise RuntimeError(f"[BatchJournal] Submission is {record.get('state')}; robot may still be running. Reconcile hardware and journal before retrying")
            _, suggestions = load_protocol(json_path)
            submitted = [suggestion.get('conc', suggestion) for suggestion in suggestions]
            if hashlib.sha256(Path(json_path).read_bytes()).hexdigest() != digest:
                raise RuntimeError('[BatchJournal] Protocol changed while recording its compositions')
            self.save({"state": "submitting", "submitted_concentrations": submitted,
                       "kind": 'qc' if self.batch_id.endswith((':qc', ':mid_qc')) else 'experiment', "protocol_path": str(Path(json_path).resolve()),
                       "protocol_sha256": digest, "existing_ids": sorted(existing_ids)})
        return None

    def finish_submission(self, result):
        with FileLock(self.path):
            record = self.load()
            record.update(state="submitted" if result.get("status") == "ok" else "uncertain",
                          submission_result=result)
            self.save(record)

    def confirm_submission(self, evidence):
        """Operator recovery after checking hardware; never sends a robot command."""
        if not isinstance(evidence, str) or not evidence.strip():
            raise ValueError("[BatchJournal] Record the hardware evidence before confirming submission")
        with FileLock(self.path):
            record = self.load()
            if record.get("state") not in ("submitting", "uncertain"):
                raise ValueError("[BatchJournal] Only uncertain submissions need operator confirmation")
            record.update(state="submitted", recovery_evidence=evidence,
                          submission_result={"status": "ok", "json_path": record["protocol_path"],
                                             "message": "Submission confirmed by operator; no command resent"})
            self.save(record)

    def collected(self, result):
        with FileLock(self.path):
            record = self.load()
            record.update(state="collected", collection_result=result)
            self.save(record)

    def committed(self, result):
        with FileLock(self.path):
            record = self.load()
            record.update(state="committed", collection_result=result)
            self.save(record)
