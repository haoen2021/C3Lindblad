"""Read one paper case from the adjacent JSON file; paths do not depend on cwd."""

import json
from pathlib import Path


def parameters(case):
    with Path(__file__).with_name("parameters.json").open() as stream:
        return json.load(stream)[case]
