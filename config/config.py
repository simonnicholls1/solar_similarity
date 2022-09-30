import yaml
import os

params = yaml.safe_load(
    open(os.path.join(os.path.dirname(__file__), "config.yaml")))