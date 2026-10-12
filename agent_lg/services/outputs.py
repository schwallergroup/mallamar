# Keep reports and optional diagnostics away from campaign inputs and runtime state.
import os
from pathlib import Path
from agent_lg import config
from agent_lg.services.literature import LocalCorpus


class OutputPaths:
    """Resolve a diagnostic destination once and protect shared campaign files."""

    def __init__(self):
        self.directories = [config.PACKAGE_DIR, config.SOURCE_ROOT / 'vendor',
                            config.SOURCE_ROOT / 'alice', config.SOURCE_ROOT / '.git',
                            config.SOURCE_ROOT / 'configs', config.CONFIGS_DIR]
        corpus = LocalCorpus()
        if corpus.path is not None:
            self.directories.append(corpus.path)
        mailbox = os.getenv('OCEAN_MAILBOX_ROOT')
        if mailbox:
            self.directories.append(Path(mailbox) / 'mailbox')
        self.files = {config.SOURCE_ROOT / name for name in ('README.md', 'LICENSE', 'NOTICE', 'CLEANING_PROGRESS.md')}
        for key in ('OCEAN_CAMPAIGN_DB', 'OCEAN_RESULTS_DB', 'OCEAN_AUDIT_PATH',
                    'OCEAN_HYPOTHESIS_PATH', 'MALLAMAR_ARTIFACTS_RECORD'):
            value = os.getenv(key)
            if value:
                self.files.add(Path(value))
        if mailbox:
            self.files.add(Path(mailbox) / 'optimizer_seq.txt')
        pool = os.getenv('OCEAN_CAMPAIGN_DB')
        if pool:
            self.files.add(Path(pool).with_name('experiments_db.txt'))
        self.files.update(Path(value) for value in os.getenv('MALLAMAR_REFERENCE_DATASETS', '').split(os.pathsep) if value)
        self.directories = [path.expanduser().resolve() for path in self.directories]
        self.files = {path.expanduser().resolve() for path in self.files}

    def resolve(self, path, extensions):
        destination = Path(path).expanduser().resolve()
        if destination.suffix.lower() not in extensions:
            raise ValueError(f'[output] Expected {extensions}, got {destination.suffix!r}')
        if (destination in self.files or '.mallamar' in destination.parts
                or any(destination.is_relative_to(directory) for directory in self.directories)):
            raise ValueError(f'[output] {destination} is a protected campaign input or runtime path')
        if destination.is_dir():
            raise ValueError(f'[output] {destination} is a directory')
        return destination
