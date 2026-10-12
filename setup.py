# Exclude local helpers and clear stale modules when rebuilding the package.
from pathlib import Path
import fnmatch, shutil
from setuptools import setup
from setuptools.command.build_py import build_py


LOCAL_HELPERS = {
    "agent_lg.scripts.analyze_agent_metrics",
    "agent_lg.scripts.analyze_campaign_pool",
    "agent_lg.scripts.build_bz_pool",
    "agent_lg.scripts.check_environment",
    "agent_lg.scripts.compare_runs",
    "agent_lg.scripts.gen_design_space",
    "agent_lg.scripts.search_campaign_pool_multi",
    "agent_lg.scripts.suggest_final_batch",
}
LOCAL_PATTERNS = ("mock_*", "test_*", "*_test", "conftest")


class BuildPackage(build_py):
    def find_package_modules(self, package, package_dir):
        return [(owner, module, path) for owner, module, path in super().find_package_modules(package, package_dir)
                if f"{owner}.{module}" not in LOCAL_HELPERS
                and not any(fnmatch.fnmatchcase(module, pattern) for pattern in LOCAL_PATTERNS)]

    def run(self):
        for name in ('agent_lg', 'alice'):
            generated = Path(self.build_lib) / name
            if generated.exists():
                shutil.rmtree(generated) # deleted modules must not survive in a rebuilt wheel
        super().run()


setup(cmdclass={'build_py': BuildPackage})
