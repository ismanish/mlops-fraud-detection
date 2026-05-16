import os
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


def load_params(path: str | None = None) -> dict[str, Any]:
    if path is None:
        path = str(PROJECT_ROOT / "params.yaml")
    with open(path) as f:
        return dict(yaml.safe_load(f))


def get_project_root() -> Path:
    return PROJECT_ROOT


def get_aws_config(params: dict | None = None) -> dict:
    if params is None:
        params = load_params()
    return {
        "region": os.getenv("AWS_REGION", params["aws"]["region"]),
        "s3_bucket": os.getenv("S3_BUCKET", params["aws"]["s3_bucket"]),
        "ecr_repository": params["aws"]["ecr_repository"],
        "lambda_function": params["aws"]["lambda_function"],
    }
