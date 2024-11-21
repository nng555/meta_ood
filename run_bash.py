import os
import pathlib
import hydra
import subprocess
import logging
from omegaconf import DictConfig

log = logging.getLogger("run_bash")

# config path set automatically but can also be manually changed
#CONFIG_PATH = pathlib.Path(__file__).parent.resolve()
#print(CONFIG_PATH)

@hydra.main(version_base=None, config_path='.', config_name="config")
def launch(cfg: DictConfig):
    log.info(cfg.command)
    subprocess.run(cfg.command, shell=True)

if __name__ == "__main__":
    launch()

