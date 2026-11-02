# Match measured rows to the current batch without changing result order.
import logging
import numpy as np
import pandas as pd
from agent_lg.config import CONC_COLS, CONC_TOLERANCE
from agent_lg.rewards import CONC_ROUND
from agent_lg.services.protocols import normalize_experiment, validate_concentrations

logger = logging.getLogger(__name__)


class ResultMatcher:
    """QC and campaign collection share concentration aliases and matching rules."""

    def __init__(self, concentrations, existing_ids=()):
        validate_concentrations(concentrations)
        self.concentrations = [item.get("conc", item) for item in concentrations]
        normalized = [normalize_experiment(item) for item in concentrations]
        self.targets = np.asarray([[float(item[column]) for column in CONC_COLS] for item in normalized])
        self.keys = [tuple(round(value, CONC_ROUND[column]) for column, value in zip(CONC_COLS, target))
                     for target in self.targets]
        self.known_keys = set(self.keys)
        self.pending_indices = list(range(len(self.concentrations)))
        self.seen_ids = {str(value).strip() for value in existing_ids}

    @property
    def pending(self):
        return [self.concentrations[index] for index in self.pending_indices]

    def collect(self, frame, mark_unmatched=False):
        missing = [column for column in ['exp_id', *CONC_COLS] if column not in frame]
        if missing:
            raise ValueError(f"[ResultMatcher] Results file is missing columns: {missing}")
        identifiers = frame['exp_id'].fillna('').astype(str).str.strip()
        eligible = ~identifiers.isin(['', 'nan', 'NaN', 'None', *self.seen_ids])
        positions = np.flatnonzero(eligible.to_numpy())
        matched = []
        for start in range(0, len(positions), 4096):
            if not self.pending_indices:
                break
            chunk = positions[start:start + 4096]
            features = frame.iloc[chunk][CONC_COLS].apply(pd.to_numeric, errors='coerce').to_numpy(dtype=float)
            if not np.isfinite(features).all() or (features < 0).any():
                raise ValueError('[ResultMatcher] New measured rows have invalid concentrations; preserve and inspect the results file')
            targets = self.targets[self.pending_indices]
            matches = np.ones((len(chunk), len(targets)), dtype=bool)
            for column in range(len(CONC_COLS)):
                matches &= np.abs(features[:, column, None] - targets[None, :, column]) < CONC_TOLERANCE
            pending_at_start = list(self.pending_indices)
            candidate_rows = range(len(chunk)) if mark_unmatched else np.flatnonzero(matches.any(axis=1))
            for offset in candidate_rows:
                row_index = chunk[offset]
                identifier = identifiers.iloc[row_index]
                if identifier in self.seen_ids:
                    continue
                if mark_unmatched:
                    self.seen_ids.add(identifier) # QC keeps the first row for each new ID
                available = [index for index, matched_here in zip(pending_at_start, matches[offset])
                             if matched_here and index in self.pending_indices]
                if not available:
                    continue
                measured_key = tuple(round(value, CONC_ROUND[column]) for column, value in zip(CONC_COLS, features[offset]))
                if measured_key in self.known_keys:
                    available = [index for index in available if self.keys[index] == measured_key]
                    if not available:
                        continue # a repeated result must not fill a nearby, still-pending composition
                else:
                    possible = np.all(np.abs(self.targets - features[offset]) < CONC_TOLERANCE, axis=1)
                    if len({key for key, matches_here in zip(self.keys, possible) if matches_here}) > 1:
                        raise ValueError('[ResultMatcher] A result matches multiple distinct batch compositions; reconcile concentrations before continuing')
                selected = available[0] # retain first-row order; exact composition resolves overlapping tolerances
                self.pending_indices.remove(selected)
                self.seen_ids.add(identifier)
                matched.append((frame.iloc[row_index], self.concentrations[selected]))
        return matched
