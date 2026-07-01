from __future__ import annotations

import re
import pathlib
import yaml


YAML_PATH = pathlib.Path(__file__).parent / 'mitre_techniques.yaml'


def load_techniques() -> dict:
    with open(YAML_PATH, 'r') as f:
        return yaml.safe_load(f)


def classify_technique(command: str) -> str | None:
    techniques = load_techniques()
    command_lower = command.lower()
    for technique_id, details in techniques.get('techniques', {}).items():
        for keyword in details.get('keywords', []):
            if re.search(rf"\b{re.escape(keyword)}\b", command_lower):
                return technique_id
    return None
