import logging
import os
import random
import sys
from pathlib import Path

from hydra.utils import to_absolute_path
import numpy as np
import torch


class ColoredHeaderFormatter(logging.Formatter):
    RESET = "\033[0m"
    DIM = "\033[2m"
    BOLD = "\033[1m"
    NAME = "\033[36m"
    LEVEL_STYLES = {
        logging.DEBUG: "\033[36m",
        logging.INFO: "\033[32m",
        logging.WARNING: "\033[33m",
        logging.ERROR: "\033[31m",
        logging.CRITICAL: "\033[1;37;41m",
    }

    def format(self, record: logging.LogRecord) -> str:
        colored_record = logging.makeLogRecord(record.__dict__.copy())
        level_style = self.LEVEL_STYLES.get(record.levelno, self.BOLD)
        colored_record.levelname = f"{level_style}[{record.levelname:^8}]{self.RESET}"
        colored_record.name = f"{self.NAME}{record.name}{self.RESET}"

        rendered = super().format(colored_record)
        if self.usesTime():
            timestamp = self.formatTime(record, self.datefmt)
            rendered = rendered.replace(timestamp, f"{self.DIM}{timestamp}{self.RESET}", 1)

        return rendered


def should_use_colors(color_mode: object, stream: object) -> bool:
    if os.getenv("NO_COLOR"):
        return False

    if isinstance(color_mode, bool):
        return color_mode

    mode = str(color_mode).lower()
    if mode in {"false", "off", "no", "0"}:
        return False
    if mode in {"true", "on", "yes", "1", "always"}:
        return True

    return hasattr(stream, "isatty") and stream.isatty()


def setup_logging(cfg) -> logging.Logger:
    log_cfg = getattr(cfg, "logging", None)
    level_name = str(getattr(log_cfg, "level", "INFO")).upper()
    level = getattr(logging, level_name, logging.INFO)
    log_format = str(
        getattr(
            log_cfg,
            "format",
            "%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        )
    )
    date_format = str(getattr(log_cfg, "datefmt", "%Y-%m-%d %H:%M:%S"))
    file_path = getattr(log_cfg, "file", None)
    capture_warnings = bool(getattr(log_cfg, "capture_warnings", True))
    color_mode = getattr(log_cfg, "colors", "auto")

    console_handler = logging.StreamHandler(stream=sys.stderr)
    formatter_cls = ColoredHeaderFormatter if should_use_colors(color_mode, console_handler.stream) else logging.Formatter
    console_handler.setFormatter(formatter_cls(log_format, datefmt=date_format))

    handlers: list[logging.Handler] = [console_handler]

    if file_path:
        resolved_file_path = resolve_runtime_path(file_path)
        ensure_parent_dir(resolved_file_path)
        file_handler = logging.FileHandler(resolved_file_path)
        file_handler.setFormatter(logging.Formatter(log_format, datefmt=date_format))
        handlers.append(file_handler)

    root_logger = logging.getLogger()
    root_logger.handlers.clear()
    root_logger.setLevel(level)
    for handler in handlers:
        root_logger.addHandler(handler)

    logging.captureWarnings(capture_warnings)

    logger = logging.getLogger("PeakDeepMaster")
    logger.setLevel(level)
    return logger


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def set_execution_device(cfg_device: str) -> str:
    logger = logging.getLogger(__name__)

    if cfg_device is None:
        return "cpu"

    if cfg_device == "cuda" and not torch.cuda.is_available():
        logger.warning("CUDA requested but unavailable. Falling back to CPU.")
        return "cpu"

    if cfg_device == "mps" and not torch.backends.mps.is_available():
        logger.warning("MPS requested but unavailable. Falling back to CPU.")
        return "cpu"

    return cfg_device


def resolve_runtime_path(path_str: str) -> str:
    if path_str is None:
        return None
    return to_absolute_path(path_str)


def ensure_parent_dir(path_str: str) -> None:
    parent = Path(path_str).parent
    parent.mkdir(parents=True, exist_ok=True)


def get_latest_checkpoint_path(checkpoint_dir: str) -> str:
    checkpoint_dir = resolve_runtime_path(checkpoint_dir)

    if os.path.isfile(checkpoint_dir):
        return checkpoint_dir

    if not os.path.isdir(checkpoint_dir):
        raise FileNotFoundError(f"Checkpoint path not found: {checkpoint_dir}")

    checkpoints = [f for f in os.listdir(checkpoint_dir) if f.endswith(".ckpt")]
    if len(checkpoints) != 1:
        raise ValueError(f"Expected exactly one .ckpt file in {checkpoint_dir}, found {len(checkpoints)}")

    return os.path.join(checkpoint_dir, checkpoints[0])


def load_checkpoint_into_model(model: torch.nn.Module, ckpt_path: str) -> torch.nn.Module:
    checkpoint = torch.load(ckpt_path, map_location="cpu")
    state_dict = checkpoint.get("state_dict", checkpoint) if isinstance(checkpoint, dict) else checkpoint
    model.load_state_dict(state_dict, strict=True)
    return model
