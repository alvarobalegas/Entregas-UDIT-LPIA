from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Literal

# 👉 En Pydantic v2 existe field_validator; en v1 NO (provoca ImportError)
from pydantic import BaseModel, ValidationError, field_validator


class Settings(BaseModel):
    app_name: str
    workers: int
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"]

    @field_validator("workers")
    @classmethod
    def positive_workers(cls, v: int) -> int:
        if v < 1:
            raise ValueError("workers must be >= 1")
        return v


def load_json(path: Path | None) -> dict:
    """Carga JSON desde ruta o devuelve un ejemplo válido por defecto."""
    if path is None:
        return {"app_name": "DemoApp", "workers": 4, "log_level": "INFO"}
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise TypeError("JSON root must be an object")
    return data


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Demo de validación de config con Pydantic v2."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Ruta a un JSON con la configuración (opcional).",
    )
    args = parser.parse_args()

    try:
        settings = Settings(**load_json(args.config))
    except ValidationError as e:
        print("❌ Config inválida. Detalles:")
        print(e)
        return 1

    print("✅ Config válida. Resumen normalizado:")
    print(json.dumps(settings.model_dump(), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())