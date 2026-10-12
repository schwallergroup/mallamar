# Verify candidate membership against durable experiment history before BO.
import hashlib, json, logging, os
from contextlib import nullcontext
from pathlib import Path
import numpy as np
import pandas as pd
from agent_lg.rewards import CONC_COLS, CONC_ROUND, REWARD_COLS, concentration_keys
from agent_lg.services.artifacts import verify_current_campaign
from agent_lg.services.protocols import CONCENTRATION_ALIASES, load_protocol
from agent_lg.services.storage import FileLock, file_revision, read_csv, write_json_atomic

logger = logging.getLogger(__name__)


def concentration_key(concentrations): 
    "Generate a unique key for a set of concentrations."

    if not isinstance(concentrations, dict):
        raise ValueError('[CampaignPool] Recorded concentrations must be an object')
    concentrations = concentrations.get('conc', concentrations)

    if not isinstance(concentrations, dict):
        raise ValueError('[CampaignPool] Invalid recorded concentration object')
    
    values = [] 
    for column, aliases in zip(CONC_COLS, CONCENTRATION_ALIASES.values()):
        value = next((concentrations[name] for name in aliases if name in concentrations), None)
        if value is None or not np.isfinite(float(value)):
            raise ValueError('[CampaignPool] Invalid recorded concentrations; inspect the batch journal')
        values.append(round(float(value), CONC_ROUND[column])) 
    return json.dumps(values, separators=(',', ':'))


class CampaignPool:
    "Keep previously attempted points out of the next candidate set."

    def __init__(self, data_path):
        self.data_path = Path(data_path).resolve()
        identifier = hashlib.sha256(str(self.data_path).encode()).hexdigest()
        self.path = self.data_path.parent / '.mallamar' / 'pools' / (identifier + '.json')

    def batch_records(self):
        for path in sorted((self.data_path.parent / '.mallamar' / 'batches').glob('*.json')):
            if path.name.endswith('.commands.json'):
                continue
            record = json.loads(path.read_text(encoding='utf-8'))
            if not isinstance(record, dict) or not record.get('db_path') or not record.get('batch_id'):
                raise ValueError('[CampaignPool] Unreadable batch identity; original files preserved')
            if Path(record['db_path']).resolve() != self.data_path:
                continue
            if record.get('kind') == 'qc' or record['batch_id'].endswith((':qc', ':mid_qc')):
                continue # calibration runs are separate from the BO candidate pool
            yield record

    def submitted_concentrations(self, record):
        "Return the submitted concentrations for a given batch record"

        if record.get('submitted_concentrations'):
            return record['submitted_concentrations']
        
        collection = record.get('collection_result', {}) # legacy collection evidence
        if not isinstance(collection, dict):
            raise ValueError('[CampaignPool] Invalid recorded collection; reconcile the batch')
        found, missing = collection.get('found', []), collection.get('missing', [])

        if found or missing:
            if len(found) != collection.get('n_found') or len(missing) != collection.get('n_missing'):
                raise ValueError('[CampaignPool] Incomplete legacy collection evidence; reconcile the batch')
            return [item['conc'] for item in found] + missing
        
        protocol = Path(record.get('protocol_path', ''))
        if not protocol.is_file() or file_revision(protocol) != record.get('protocol_sha256'):
            raise ValueError('[CampaignPool] Legacy batch has no immutable composition evidence; reconcile it before BO')
        return load_protocol(protocol)[1]

    def check(self, expected_exp_ids=(), read_only=False):
        guard = nullcontext() if read_only else FileLock(self.path)
        with guard:
            # Check the current CSV before trusting any recovered state.
            original_revision = file_revision(self.data_path)
            frame = read_csv(self.data_path)
            required_columns = CONC_COLS + REWARD_COLS + ['exp_id']
            missing_columns = [column for column in required_columns if column not in frame]

            if missing_columns:
                raise ValueError(f'[CampaignPool] Missing pool columns: {missing_columns}')
            concentrations = frame[CONC_COLS].apply(pd.to_numeric, errors='raise')

            if not np.isfinite(concentrations.to_numpy()).all():
                raise ValueError('[CampaignPool] Non-finite concentrations')
            
            rewards = frame[REWARD_COLS].apply(pd.to_numeric, errors='raise') # convert reward columns to numeric
            missing_rewards = rewards.isna()

            partially_scored = missing_rewards.any(axis=1) & ~missing_rewards.all(axis=1)
            if partially_scored.any() or np.isinf(rewards.to_numpy()).any():
                raise ValueError('[CampaignPool] Partially scored or infinite objectives; finish scoring before BO')

            # Identify rows that have complete reward information
            has_rewards = ~missing_rewards.any(axis=1)
            experiment_ids = frame['exp_id'].fillna('').astype(str).str.strip()
            was_attempted = ~experiment_ids.isin(('', 'nan', 'NaN', 'None'))
            unscored_attempts = frame.index[was_attempted & ~has_rewards].tolist()

            # Ensure that any attempted experiments have corresponding rewards before proceeding with BO
            if unscored_attempts:
                raise ValueError(f'[CampaignPool] Attempted rows {unscored_attempts[:8]} have missing rewards and would re-enter heldout; recover synchronization/scoring before BO (including when reward mode is off)')
            if experiment_ids[was_attempted].duplicated().any():
                raise ValueError('[CampaignPool] Duplicate experiment IDs; reconcile pool identity')
            compositions = [json.dumps(key, separators=(',', ':')) for key in concentration_keys(concentrations)]
            if len(set(compositions)) != len(compositions):
                raise ValueError('[CampaignPool] Duplicate pool compositions; a measured point could remain a candidate')

            # Every completed batch must still have its measurements in this pool.
            row_by_composition = dict(zip(compositions, frame.index))
            measured_ids = set(experiment_ids[was_attempted & has_rewards])
            if not set(map(str, expected_exp_ids)).issubset(measured_ids):
                raise ValueError('[CampaignPool] Checkpoint measurements are missing or unscored; restore the current pool before BO')
            for batch in self.batch_records():
                if batch.get('state') != 'committed':
                    raise ValueError(f"[CampaignPool] Batch {batch['batch_id']} is {batch.get('state')}; recover it before another BO")
                collection = batch.get('collection_result', {})
                if not isinstance(collection, dict):
                    raise ValueError('[CampaignPool] Invalid recorded collection; reconcile the batch')
                for suggestion in self.submitted_concentrations(batch):
                    row = row_by_composition.get(concentration_key(suggestion))
                    measurement_complete = row is not None and was_attempted.loc[row] and has_rewards.loc[row]
                    if not measurement_complete:
                        raise ValueError(f"[CampaignPool] Committed batch {batch['batch_id']} has a missing/unscored attempted point; pool may be stale")
                for measurement in collection.get('found', []):
                    row = row_by_composition.get(concentration_key(measurement['conc']))
                    same_experiment = row is not None and experiment_ids.loc[row] == str(measurement['exp_id'])
                    if not same_experiment:
                        raise ValueError('[CampaignPool] Recorded measurement no longer matches its pool composition; restore the current pool')

            # A stale CSV must never turn previously measured points into candidates.
            identity_columns = CONC_COLS + (['prompt'] if 'prompt' in frame else [])
            pool_identity = hashlib.sha256(frame[identity_columns].to_csv(index=False).encode()).hexdigest()
            observed = {}
            for composition, experiment_id, attempted, scored in zip(compositions, experiment_ids, was_attempted, has_rewards):
                if scored:
                    observed[composition] = experiment_id if attempted else None
            saved_history = None
            if self.path.exists():
                saved_history = json.loads(self.path.read_text(encoding='utf-8'))
                valid_history = (isinstance(saved_history, dict)
                    and saved_history.get('version') == 1
                    and saved_history.get('db_path') == str(self.data_path)
                    and saved_history.get('input_sha256') == pool_identity
                    and isinstance(saved_history.get('observed'), dict))
                if not valid_history:
                    raise ValueError('[CampaignPool] Pool identity/history changed; restore the original pool and records')
                for composition, experiment_id in saved_history['observed'].items():
                    same_observation = (composition in observed
                        and (experiment_id is None or observed[composition] == experiment_id))
                    if not same_observation:
                        raise ValueError('[CampaignPool] Previously observed point became missing or changed identity; pool may be stale')

            # Save only after verification; read-only inspection leaves history untouched.
            if file_revision(self.data_path) != original_revision:
                raise ValueError('[CampaignPool] Pool changed during verification; retry after its writer finishes')
            history = {'version': 1, 'db_path': str(self.data_path), 'input_sha256': pool_identity, 'observed': observed}
            if history != saved_history and not read_only:
                write_json_atomic(self.path, history) # recomputed rewards may change; observed membership cannot regress
        return frame


def check_campaign_pool(data_path=None, expected_exp_ids=(), pending_batch=False):
    
    campaign = os.getenv('OCEAN_CAMPAIGN_DB')
    if not campaign:
        return None # offline datasets keep their original split behavior
    if data_path is not None and Path(data_path).resolve() != Path(campaign).resolve():
        raise ValueError('[CampaignPool] BO dataset differs from the active campaign pool')
    verify_current_campaign(campaign)
    pool = CampaignPool(campaign)
    if pending_batch and any(record.get('state') != 'committed' for record in pool.batch_records()):
        return None # unfinished batches need synchronization/scoring before this check can pass
    return pool.check(expected_exp_ids)
