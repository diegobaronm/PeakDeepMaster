import hydra
import logging
import time
from omegaconf import DictConfig, OmegaConf
from omegaconf.errors import ConfigAttributeError
from rich import print
from rich.syntax import Syntax

from src.data.DataModule import PeakDeepMasterDataModule
from src.models.RatioEstimator import CouplingRatioEstimator
from src.utils.Train import train
from src.utils.Predict import predict
from src.utils.Performance import testing
from src.utils.Inference import run_inference
from src.utils.utils import setup_logging


@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(cfg: DictConfig) -> None:
    start_time = time.time()
    logger = setup_logging(cfg)

    try:
        syntax = Syntax(OmegaConf.to_yaml(cfg), "yaml", theme="monokai", line_numbers=False)
        logger.info("Configuration:")
        print(syntax)

        logger.info("Initializing data module...")
        datamodule = PeakDeepMasterDataModule(cfg)

        if cfg.general.mode == "train":
            logger.info("Starting training...")
            train(datamodule, CouplingRatioEstimator, cfg)
        elif cfg.general.mode == "predict":
            predict(datamodule, CouplingRatioEstimator, cfg)
        elif cfg.general.mode == "performance":
            testing(datamodule, CouplingRatioEstimator, cfg)
        elif cfg.general.mode == "infer":
            run_inference(datamodule, CouplingRatioEstimator, cfg)
        else:
            raise ValueError(f"Unsupported mode: {cfg.general.mode}")
    except ConfigAttributeError:
        logger.exception("Configuration error: please check required keys in config.")
        raise
    
    end_time = time.time()
    elapsed_time = end_time - start_time
    logger.info(f"Execution completed in {elapsed_time:.2f} seconds.")

if __name__ == "__main__":
    main()
