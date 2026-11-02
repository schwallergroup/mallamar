# Replay recorded measurements through the normal campaign collection and scoring.
import hashlib, json
from pathlib import Path
import numpy as np
import pandas as pd
from agent_lg.rewards import CONC_COLS, RAW_COLS, REWARD_COLS, FAILURE_MODE_COL, FAILURE_CHEMISTRY, concentration_keys
from agent_lg.services.artifacts import file_digest
from agent_lg.services.batches import BatchJournal
from agent_lg.services.pools import CampaignPool
from agent_lg.services.protocols import load_protocol, normalize_experiment, positive_number, reaction_prompt
from agent_lg.services.storage import FileLock, file_revision, read_csv, write_csv_atomic, write_json_atomic
from agent_lg.services.validation import CampaignSafetyError


class ReplayController:
    """Keep unrevealed outcomes private and execute only the recorded batch."""

    def __init__(self, source_path, db_path):
        self.source_path = Path(source_path).expanduser().resolve()
        self.db_path = Path(db_path).expanduser().resolve()
        identifier = hashlib.sha256(str(self.db_path).encode()).hexdigest()
        self.path = self.db_path.parent / '.mallamar' / 'replay' / (identifier + '.json')
        self.results_path = self.path.with_suffix('.results.csv')
        if self.source_path in (self.db_path, self.results_path):
            raise ValueError('[replay] Source and working pool/results must be separate files')
        if self.source_path.suffix.lower() != '.csv':
            raise ValueError('[replay] Provide a measured .csv file, separate from the literature corpus')
        self.digest = file_digest(self.source_path)
        self.data = read_csv(self.source_path)
        columns = CONC_COLS + RAW_COLS
        if not set(columns).issubset(self.data.columns):
            raise ValueError(f'[replay] Source needs concentration and raw measurement columns: {columns}')
        source = self.data
        self.data = source[columns].apply(pd.to_numeric, errors='raise')
        if FAILURE_MODE_COL in source:
            self.data[FAILURE_MODE_COL] = source[FAILURE_MODE_COL]
        if len(self.data) < 6 or not np.isfinite(self.data[CONC_COLS]).all().all() or (self.data[CONC_COLS] < 0).any().any():
            raise ValueError('[replay] Provide at least six candidates with finite non-negative concentrations')
        n_osc = self.data['n_osc']
        if not np.isfinite(n_osc).all() or (n_osc < 0).any() or (n_osc != n_osc.astype(int)).any():
            raise ValueError('[replay] Each recorded experiment needs a non-negative integer n_osc')
        metrics = ['interp_period_mean_s', 'interp_period_std_s', 'interp_maxima_mean_V',
                   'interp_amplitude_mean_V', 'total_time_s']
        successful = self.data.loc[n_osc > 0, metrics]
        if (not np.isfinite(successful).all().all() or (successful < 0).any().any()
                or (successful['interp_period_mean_s'] <= 0).any()):
            raise ValueError('[replay] Oscillating experiments need finite, valid raw reward metrics')
        keys = concentration_keys(self.data)
        if len(set(keys)) != len(keys):
            raise ValueError('[replay] Source must contain one measurement per distinct pool composition')
        self.positions = dict(zip(keys, range(len(keys))))
        self.identifiers = []
        for index in range(len(self.data)):
            self.identifiers.append(f'replay_{self.digest[:16]}_{index}')
        if file_digest(self.source_path) != self.digest:
            raise ValueError('[replay] Source changed while being read')

    def _validate_measurements(self, frame, pool=False):
        columns = {'exp_id', *self.data.columns}
        if pool:
            columns.update(['prompt', *REWARD_COLS, FAILURE_MODE_COL])
        if (set(frame.columns) != columns or frame['exp_id'].isna().any()
                or frame['exp_id'].duplicated().any()):
            raise CampaignSafetyError('[replay] Invalid replay measurements; original file preserved')
        identifiers = set(frame['exp_id'])
        lookup = dict(zip(self.identifiers, range(len(self.data))))
        if not identifiers.issubset(lookup):
            raise CampaignSafetyError('[replay] Results contain an unknown experiment identity')
        submitted = set()
        if identifiers:
            history = CampaignPool(self.db_path)
            for record in history.batch_records():
                if (record.get('replay_source') != self.digest
                        or record.get('state') not in ('submitting', 'submitted', 'collected', 'committed', 'uncertain')):
                    raise CampaignSafetyError('[replay] Measurement journal differs from the current replay')
                points = []
                for point in history.submitted_concentrations(record):
                    points.append(normalize_experiment(point))
                if len(points) != 6:
                    raise CampaignSafetyError('[replay] Incomplete recorded replay batch')
                for key in concentration_keys(pd.DataFrame(points)):
                    if key not in self.positions:
                        raise CampaignSafetyError('[replay] Recorded batch contains an unknown candidate')
                    submitted.add(self.identifiers[self.positions[key]])
        if not identifiers.issubset(submitted):
            raise CampaignSafetyError('[replay] Measurements have no recorded replay submission')
        positions = []
        for identifier in frame['exp_id']:
            positions.append(lookup[identifier])
        expected = self.data.iloc[positions]
        values = frame[CONC_COLS + RAW_COLS].apply(pd.to_numeric, errors='raise')
        if not np.allclose(values.to_numpy(), expected[CONC_COLS + RAW_COLS].to_numpy(), rtol=1e-12, atol=1e-12, equal_nan=True):
            raise CampaignSafetyError('[replay] Recorded results differ from their source measurements')
        if FAILURE_MODE_COL in frame:
            actual_modes = frame[FAILURE_MODE_COL].fillna('').astype(str).to_numpy()
            expected_modes = expected.get(FAILURE_MODE_COL, pd.Series('', index=expected.index)).fillna('').astype(str).to_numpy()
            matching = actual_modes == expected_modes
            if pool:
                matching |= (expected_modes == '') & (expected['n_osc'].to_numpy() == 0) & (actual_modes == FAILURE_CHEMISTRY)
            if not matching.all():
                raise CampaignSafetyError('[replay] Recorded failure labels differ from their source')

    def prepare(self):
        identity = {'version': 1, 'source_path': str(self.source_path), 'source_sha256': self.digest,
                    'db_path': str(self.db_path)}
        saved = None
        if self.path.exists():
            saved = json.loads(self.path.read_text())
            if not isinstance(saved, dict) or {key: saved.get(key) for key in identity} != identity:
                raise ValueError('[replay] Source or workspace differs from the recorded replay')
        elif self.db_path.exists() or self.results_path.exists():
            raise ValueError('[replay] Start with a new working pool; existing files are never overwritten')
        if saved is None:
            write_json_atomic(self.path, {**identity, 'initialized': False})
        inputs = self.data[CONC_COLS].copy()
        inputs['prompt'] = inputs.apply(reaction_prompt, axis=1)
        if not self.db_path.exists():
            if saved and saved.get('initialized'):
                raise ValueError('[replay] Recorded working pool is missing; restore it before resuming')
            masked = inputs.copy()
            masked.insert(0, 'exp_id', pd.Series(None, index=masked.index, dtype=object))
            for column in RAW_COLS + REWARD_COLS + [FAILURE_MODE_COL]:
                masked[column] = np.nan
            write_csv_atomic(masked, self.db_path, None)
        current = read_csv(self.db_path)
        if (not set(inputs.columns).issubset(current.columns)
                or concentration_keys(current) != concentration_keys(inputs)
                or current['prompt'].tolist() != inputs['prompt'].tolist()):
            raise ValueError('[replay] Working pool inputs differ from the recorded candidates')
        outcomes = ['exp_id', *RAW_COLS, *REWARD_COLS, FAILURE_MODE_COL]
        if not set(outcomes).issubset(current.columns):
            raise CampaignSafetyError('[replay] Working pool outcome columns are missing; original file preserved')
        revealed = current['exp_id'].notna()
        self._validate_measurements(current.loc[revealed], pool=True)
        if current.loc[~revealed, outcomes].notna().any().any():
            raise CampaignSafetyError('[replay] Unsubmitted candidates contain outcomes; restore the masked working pool')
        if self.results_path.exists():
            self._validate_measurements(read_csv(self.results_path))
        if saved is None or not saved.get('initialized'):
            write_json_atomic(self.path, {**identity, 'initialized': True})

    def submit(self, json_path=None, proto_path=None, _batch_id=None, _db_path=None):
        if not _batch_id or not _db_path or Path(_db_path).resolve() != self.db_path:
            raise CampaignSafetyError('[replay] Submission requires the current campaign batch identity')
        if file_digest(self.source_path) != self.digest:
            raise CampaignSafetyError('[replay] Measured source changed; restore the original file')
        json_path = json_path or proto_path
        _, suggestions = load_protocol(json_path)
        if len(suggestions) != 6:
            raise CampaignSafetyError('[replay] Campaign batches contain six experiments')
        normalized = []
        for point in suggestions:
            normalized.append(normalize_experiment(point))
        keys = concentration_keys(pd.DataFrame(normalized))
        if len(set(keys)) != 6 or any(key not in self.positions for key in keys):
            raise CampaignSafetyError('[replay] Batch must contain six distinct source candidates')
        positions = [self.positions[key] for key in keys]
        selected = self.data.iloc[positions].copy()
        selected.insert(0, 'exp_id', [self.identifiers[index] for index in positions])
        journal = BatchJournal(self.db_path, _batch_id)
        with FileLock(self.results_path):
            revision = file_revision(self.results_path)
            existing = read_csv(self.results_path) if revision is not None else selected.iloc[:0].copy()
            self._validate_measurements(existing)
            known = set(existing['exp_id'])
            record = journal.load()
            if not record:
                pool = read_csv(self.db_path)
                attempted = pool.iloc[positions]['exp_id'].notna().any()
                if attempted or known.intersection(selected['exp_id']):
                    raise CampaignSafetyError('[replay] A previously attempted candidate cannot be submitted again')
            cached = journal.begin_submission(json_path, known, replay_source=self.digest)
            pending = selected[~selected['exp_id'].isin(known)]
            if len(pending):
                combined = pd.concat([existing, pending], ignore_index=True)
                write_csv_atomic(combined, self.results_path, revision)
            if cached is not None:
                return cached
            result = {'status': 'ok', 'returncode': 0, 'json_path': str(Path(json_path).resolve()),
                      'simulated': True, 'message': 'Recorded batch measurements are ready for collection'}
            journal.finish_submission(result)
        return result

    def run_qc(self, volume_uL, db_path, **options):
        positive_number(volume_uL, 'QC volume_uL')
        if Path(db_path).resolve() != self.db_path:
            raise CampaignSafetyError('[replay] QC belongs to another campaign')
        result = {'status': 'ok', 'n_found': 0, 'n_missing': 0, 'simulated': True,
                  'message': 'QC simulated; no hardware or chemical validation was performed'}
        batch_id = options.get('_batch_id')
        if batch_id:
            if not batch_id.endswith((':qc', ':mid_qc')):
                raise CampaignSafetyError('[replay] QC requires its own batch identity')
            journal = BatchJournal(self.db_path, batch_id)
            with FileLock(journal.path):
                record = journal.load()
                if record:
                    if record.get('replay_source') != self.digest or record.get('state') != 'committed':
                        raise CampaignSafetyError('[replay] QC record differs from the current replay')
                    return record['collection_result']
                journal.save({'state': 'committed', 'kind': 'qc', 'replay_source': self.digest,
                              'collection_result': result})
        return result
