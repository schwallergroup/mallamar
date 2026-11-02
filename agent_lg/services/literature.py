# Resolve the caller's local papers without searching other checkouts.
import os
from pathlib import Path


class LocalCorpus:
    """One explicit directory shared by preflight, extraction and fingerprints."""
    patterns = ('*.pdf', '*.txt', '*.md')

    def __init__(self, directory=None):
        directory = directory or os.getenv('OCEAN_CORPUS_DIR')
        self.path = Path(directory).expanduser().resolve() if directory else None

    def files(self):
        if self.path is None or not self.path.is_dir():
            return []
        return sorted(path for pattern in self.patterns for path in self.path.glob(pattern) if path.is_file())
