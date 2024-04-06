from config import ConfigurationSet, config_from_env, config_from_dotenv, config_from_toml
from pathlib import Path

def config():
    config = Path(__file__).parent
    root = config.parent.parent
    return ConfigurationSet(
        config_from_env(prefix="ML_PIPELINE", separator="__", lowercase_keys=True),
        config_from_dotenv(root / ".env", read_from_file=True, lowercase_keys=True, interpolate=True, interpolate_type=1),
        config_from_toml(config / "training.toml", read_from_file=True),
        config_from_toml(config / "data.toml", read_from_file=True),
        config_from_toml(config / "model.toml", read_from_file=True),
    )

