# Generate campaign suggestions with ALICE and save batches for the robot mailbox.
import warnings, re, os, logging, tempfile, numpy as np, torch
from pathlib import Path
from botorch.exceptions import InputDataWarning
from alice.bo.optimizer import BayesianOptimization
from alice.utils.diagnostics import mean_gp_lengthscale
from agent_lg.backends.alice import alice_config
from agent_lg.config import BO_KEY_MAP

from agent_lg.services.profiling import profile_step
from agent_lg.services.pools import check_campaign_pool
from agent_lg.services.artifacts import prepare_models
from agent_lg.services.storage import write_json_atomic
from agent_lg.services.validation import SuggestionInputs
from agent_lg.services.outputs import OutputPaths

logger = logging.getLogger(__name__)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def _suggestion_dir(config: dict, seed: int):
    """Keep each BO call's saved batch so earlier batch records stay usable."""
    out_root = config.get("out_root", "./results_MOBBOL")
    def _clean(s): return re.sub(r'[^\w\-_\. ]', '_', str(s))
    bo_cfg = config["bo"]["init_args"]
    feat_init = config["data"]["init_args"]["featurizer"]["init_args"]
    acqf_raw = config.get("acquisition", {}).get("class_path", "").split(".")[-1]
    sur_short = config.get("surrogate_model", {}).get("class_path", "").split(".")[-1]
    rep = _clean(feat_init.get("representation", "unk"))
    model = _clean((feat_init.get("model_name") or "unk").split("/")[-1])
    bmode = bo_cfg.get("batch_mode", "greedy")
    bsize = bo_cfg.get("batch_size", 5)
    sug_dir = os.path.join(
        out_root,
        f"{sur_short}-{acqf_raw}-{rep}-{model}",
        f"seed={seed}",
        f"batch-{bmode}_q{bsize}",
        "suggestions",
    )
    destination = OutputPaths().resolve(Path(sug_dir) / 'suggestions.json', ('.json',)).parent
    destination.mkdir(parents=True, exist_ok=True)
    return tempfile.mkdtemp(prefix='batch-', dir=destination)


def _mailbox_suggestion(rec, iter_num, hint):
    """Label concentrations for the batch; hardware builds the execution protocol."""

    conc_M = {BO_KEY_MAP[k]: v for k, v in rec.items() if k in BO_KEY_MAP}
    return f"mailbox_iter{iter_num}_{hint}", conc_M


def _generate_suggestions(config, target_volume_uL):
    inputs = SuggestionInputs(config, target_volume_uL) #
    config = alice_config(config)
    config['seed'], config['n_iters'] = inputs.seed, inputs.n_iters
    bo = BayesianOptimization(config, device=device) # from Alice optimizer
    check_campaign_pool(config['data']['init_args']['data_path'])
    sug_dir = _suggestion_dir(config, inputs.seed)
    prepare_models(config)
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "max_split_size_mb:128")
    with profile_step("model_loading_and_data_setup"):
        bo.setup()
    data = bo.data
    print(f"[suggestions] Training rows: {len(data.train_indexes)}; candidates: {len(data.heldout_indices)}")

    concentration_columns = {
        "conc_NaBrO3_M": "NaBrO3_M", "conc_MA_M": "MA_M",
        "conc_H2SO4_M": "H2SO4_M", "conc_Ru_M": "Ru(bpy)3_M",
    }
    seed_ids = np.asarray(data.train_indexes, dtype=int)
    seeds = data.data.iloc[seed_ids][list(concentration_columns)].rename(columns=concentration_columns)
    cumulative = [{"iter": 0, "suggestions": seeds.to_dict(orient="records"), "reserve": []}]
    mean_lengthscale = None
    
    for iteration in range(config["n_iters"]):
        with profile_step("gp_fit_and_acquisition"):
            primary_ids, reserve_ids = bo.suggest()
        try:
            lengthscale = mean_gp_lengthscale(bo.optimizer)
            if lengthscale is not None:
                mean_lengthscale = lengthscale
                print(f"LENGTHSCALE: {mean_lengthscale:.8g}")
        except Exception as error:
            print(f"[lengthscale] extraction skipped: {error}")

        batch = {"iter": iteration + 1, "suggestions": [], "reserve": []}
        for kind, row_ids, prefix, hint in [("suggestions", primary_ids, "PROTO", "sug"),
                                          ("reserve", reserve_ids, "RESERVE", "reserve")]:
            rows = data.data.iloc[row_ids][list(concentration_columns)].rename(columns=concentration_columns)
            for index, row in enumerate(rows.to_dict(orient="records"), 1):
                path, concentrations = _mailbox_suggestion(row, iteration + 1, f"{hint}_{index}")
                values = " ".join(f"{name}={value:.8g}" for name, value in concentrations.items())
                print(f"{prefix}: {path} | {values}") # the worker parser reads these records
                batch[kind].append({"path": path, "conc": concentrations})
        cumulative.append(batch)

    out_path = os.path.join(sug_dir, f"suggestions_iter{config['n_iters']:d}.json")
    write_json_atomic(out_path, cumulative)
    print(f"JSON: {out_path}")
    return SuggestionResult(cumulative, out_path, mean_lengthscale)


class SuggestionResult:
    """Suggestions and saved output from one BO call."""

    def __init__(self, iterations, json_path, mean_lengthscale=None):
        self.iterations = iterations
        self.json_path = json_path
        self.mean_lengthscale = mean_lengthscale
        self.primaries = iterations[-1]["suggestions"]
        self.reserves = iterations[-1]["reserve"]


class SuggestionService:
    """Calls ALICE and writes suggestions for a campaign batch."""

    def __init__(self, config):
        self.config = config # translated on each call bcs ALICE adds runtime values

    def suggest(self, target_volume_uL: float):
        precision = torch.get_float32_matmul_precision()
        suppress_errors = torch._dynamo.config.suppress_errors
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', category=InputDataWarning)
            torch.set_float32_matmul_precision('high') # keep the campaign's fitting precision
            torch._dynamo.config.suppress_errors = True
            try:
                return _generate_suggestions(self.config, target_volume_uL)
            finally:
                torch.set_float32_matmul_precision(precision)
                torch._dynamo.config.suppress_errors = suppress_errors
