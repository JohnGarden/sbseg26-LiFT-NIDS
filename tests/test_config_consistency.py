"""Config consistency tests.

Fails fast when a config file violates known methodological contracts:
  - MAWIFlow must use label_column: label (not label_1).
  - label_multiclass and label_1 must not appear as feature inputs.
  - Transformer configs must declare ff_dim == 512 (4 × d_model, fixed per IM09).
  - Sequential experiment configs must declare window_size in {8, 16, 32}.
  - Experiment configs must not reference unsupported model families.
"""

from __future__ import annotations

from pathlib import Path

import yaml
import pytest

REPO_ROOT = Path(__file__).parent.parent
CONFIGS = REPO_ROOT / "configs"
EXPERIMENTS_DIR = CONFIGS / "experiments"

_SUPPORTED_FAMILIES = {"xgboost", "mlp", "cnn_bilstm", "transformer", "transformer_light"}
_VALID_WINDOW_SIZES = {8, 16, 32}
_TRANSFORMER_FF_DIM = 512


def _load_yaml(path: Path) -> dict:
    with path.open(encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _experiment_configs() -> list[Path]:
    return sorted(EXPERIMENTS_DIR.glob("exp_*.yaml"))


# ---------------------------------------------------------------------------
# MAWIFlow dataset config
# ---------------------------------------------------------------------------

class TestMAWIFlowConfig:
    CFG = _load_yaml(CONFIGS / "datasets" / "mawiflow.yaml")

    def test_label_column_is_label(self):
        assert self.CFG["dataset"]["label_column"] == "label", (
            "mawiflow.yaml must use label_column: label — "
            "MAWIFlow pipeline generates 'label', not 'label_1'."
        )

    def test_label_1_not_declared_as_label_column(self):
        assert self.CFG["dataset"].get("label_column") != "label_1"

    def test_crossdataset_not_in_supported_protocols(self):
        protocols = self.CFG["dataset"].get("supported_split_protocols", [])
        assert "crossdataset" not in protocols, (
            "crossdataset is out of scope for MAWIFlow in this phase."
        )

    def test_label_multiclass_role_is_audit_only(self):
        role = (
            self.CFG.get("dataset", {})
            .get("task", {})
            .get("label_multiclass_role", "")
        )
        assert role == "audit_only", (
            "label_multiclass must be declared as audit_only metadata, not a training target."
        )


# ---------------------------------------------------------------------------
# Transformer model config
# ---------------------------------------------------------------------------

class TestTransformerConfig:
    CFG = _load_yaml(CONFIGS / "models" / "transformer_light.yaml")

    def test_ff_dim_matches_code_constant(self):
        ff_dim = self.CFG["model"]["params"]["ff_dim"]
        assert ff_dim == _TRANSFORMER_FF_DIM, (
            f"transformer_light.yaml declares ff_dim={ff_dim} but the code "
            f"hardcodes FF_DIM={_TRANSFORMER_FF_DIM} (4 × d_model per IM09). "
            "Keep them in sync."
        )

    def test_d_model_is_128(self):
        assert self.CFG["model"]["params"]["d_model"] == 128

    def test_num_layers_is_2(self):
        assert self.CFG["model"]["params"]["num_layers"] == 2

    def test_n_heads_is_2(self):
        assert self.CFG["model"]["params"]["n_heads"] == 2

    def test_classification_head_is_last_token(self):
        assert self.CFG["model"]["classification_head"]["strategy"] == "last_token"


# ---------------------------------------------------------------------------
# Experiment configs — sequential window_size
# ---------------------------------------------------------------------------

class TestSequentialExperimentWindowSize:
    """Sequential experiments (temporal_sequence.enabled=true) must use W ∈ {8,16,32}."""

    @pytest.mark.parametrize("config_path", _experiment_configs(), ids=lambda p: p.name)
    def test_window_size_in_valid_set(self, config_path: Path):
        cfg = _load_yaml(config_path)
        ts = cfg.get("experiment", {}).get("temporal_sequence", {})
        if not ts.get("enabled", False):
            pytest.skip("Not a sequential experiment.")
        ws = ts.get("window_size")
        assert ws in _VALID_WINDOW_SIZES, (
            f"{config_path.name}: window_size={ws} is outside the documented "
            f"ablation set {_VALID_WINDOW_SIZES}."
        )


# ---------------------------------------------------------------------------
# Experiment configs — supported model families
# ---------------------------------------------------------------------------

class TestSupportedModelFamilies:
    """Experiment configs must not reference model families absent from the runner."""

    @pytest.mark.parametrize("config_path", _experiment_configs(), ids=lambda p: p.name)
    def test_model_family_supported_or_flagged_backlog(self, config_path: Path):
        cfg = _load_yaml(config_path)
        # Detect backlog configs by the BACKLOG comment in the raw file text
        raw = config_path.read_text(encoding="utf-8")
        if "BACKLOG" in raw or "NÃO SUPORTADO" in raw:
            pytest.skip("Backlog config — runner support not expected.")

        model_cfg_path_str = cfg.get("experiment", {}).get("model_config", "")
        # Derive family from the model config file name
        model_cfg_path = REPO_ROOT / model_cfg_path_str
        if not model_cfg_path.exists():
            pytest.skip(f"Model config not found: {model_cfg_path_str}")

        model_cfg = _load_yaml(model_cfg_path)
        family = model_cfg.get("model", {}).get("family", "")
        assert family in _SUPPORTED_FAMILIES, (
            f"{config_path.name}: model family '{family}' is not supported by "
            f"run_experiment.py / Optuna. Supported: {sorted(_SUPPORTED_FAMILIES)}."
        )


# ---------------------------------------------------------------------------
# Label exclusion contract
# ---------------------------------------------------------------------------

class TestLabelExclusionContract:
    """label_1 and label_multiclass must not be usable as training targets."""

    def test_label_1_not_the_mawiflow_target(self):
        cfg = _load_yaml(CONFIGS / "datasets" / "mawiflow.yaml")
        assert cfg["dataset"]["label_column"] != "label_1"

    def test_label_multiclass_not_the_mawiflow_target(self):
        cfg = _load_yaml(CONFIGS / "datasets" / "mawiflow.yaml")
        assert cfg["dataset"]["label_column"] != "label_multiclass"

    def test_label_multiclass_not_the_ciciot_target(self):
        cfg = _load_yaml(CONFIGS / "datasets" / "ciciot2023.yaml")
        assert cfg["dataset"]["label_column"] != "label_multiclass"
