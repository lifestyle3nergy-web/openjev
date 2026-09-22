"""Runtime settings, all from the environment."""
import os
from dataclasses import dataclass, field


def _env(name, default):
    return os.environ.get(name, default)


@dataclass(frozen=True)
class Settings:
    upstream: str = field(default_factory=lambda: _env("OPENJEV_UPSTREAM", "http://127.0.0.1:8000"))
    upstream_model: str = field(default_factory=lambda: _env("OPENJEV_UPSTREAM_MODEL", "dgemma"))
    tokenizer: str = field(default_factory=lambda: _env("OPENJEV_TOKENIZER", "nvidia/diffusiongemma-26B-A4B-it-NVFP4"))
    backend: str = field(default_factory=lambda: _env("OPENJEV_BACKEND", "vllm"))
    mlx_model: str = field(default_factory=lambda: _env("OPENJEV_MLX_MODEL", "mlx-community/diffusiongemma-26B-A4B-it-4bit"))
    mlx_max_prompt: int = field(default_factory=lambda: int(_env("OPENJEV_MLX_MAX_PROMPT", "32768")))
    canvas: int = field(default_factory=lambda: int(_env("OPENJEV_CANVAS", "64")))
    canvas_step: int = field(default_factory=lambda: int(_env("OPENJEV_CANVAS_STEP", "16")))
    max_inflight: int = field(default_factory=lambda: int(_env("OPENJEV_MAX_INFLIGHT", "64")))
    max_queue: int = field(default_factory=lambda: int(_env("OPENJEV_MAX_QUEUE", "512")))
    # Optional auth. OPENJEV_API_KEY: clients send it as a Bearer token.
    # OPENJEV_ORIGIN_SECRET: a front proxy sends it as X-Origin-Secret.
    api_key: str = field(default_factory=lambda: _env("OPENJEV_API_KEY", ""))
    origin_secret: str = field(default_factory=lambda: _env("OPENJEV_ORIGIN_SECRET", ""))
    auto_threshold: float = field(default_factory=lambda: float(_env("OPENJEV_AUTO_THRESHOLD", "0.1")))
    auto_max: int = field(default_factory=lambda: int(_env("OPENJEV_AUTO_MAX", "4")))
    max_images: int = field(default_factory=lambda: int(_env("OPENJEV_MAX_IMAGES", "8")))
    max_image_bytes: int = field(default_factory=lambda: int(_env("OPENJEV_MAX_IMAGE_BYTES", str(5 * 1024 * 1024))))
    # Kept small: generation denoises many blocks and must not crowd out System One reads.
    gen_max_inflight: int = field(default_factory=lambda: int(_env("OPENJEV_GEN_MAX_INFLIGHT", "8")))
    gen_max_queue: int = field(default_factory=lambda: int(_env("OPENJEV_GEN_MAX_QUEUE", "32")))
    gen_max_tokens: int = field(default_factory=lambda: int(_env("OPENJEV_GEN_MAX_TOKENS", "8192")))
    # The encoder backends (OPENJEV_BACKEND=laya or verdict). OPENJEV_DEVICE empty picks CUDA when present.
    laya_model: str = field(default_factory=lambda: _env("OPENJEV_LAYA_MODEL", "convaiinnovations/laya-typed-decisions"))
    verdict_model: str = field(default_factory=lambda: _env("OPENJEV_VERDICT_MODEL", "heman10x/rlcd-modernbert-151m"))
    device: str = field(default_factory=lambda: _env("OPENJEV_DEVICE", ""))
    encoder_batch: int = field(default_factory=lambda: int(_env("OPENJEV_ENCODER_BATCH", "16")))
    # Other System One models served by other OpenJev containers: "name=url,name=url".
    # A request for one of them is passed through unchanged, so one origin serves all.
    model_routes: dict = field(default_factory=lambda: parse_routes(_env("OPENJEV_MODEL_ROUTES", "")))


def parse_routes(text):
    routes = {}
    for part in filter(None, (p.strip() for p in text.split(","))):
        name, sep, url = part.partition("=")
        if not (sep and name.strip() and url.strip()):
            raise ValueError(f"OPENJEV_MODEL_ROUTES: {part!r} is not name=url")
        routes[name.strip()] = url.strip().rstrip("/")
    return routes


MODEL_VERSION = "openjev-0.1"
# openjev-0.1 is the wire name of this model, not the package version (see __init__.py).
MODEL_ALIASES = {"openjev-latest", MODEL_VERSION}
GEN_MODEL = "diffusiongemma-26b"
# accepted so TypeSafe's SDKs work unchanged (their default is jev-latest)
SDK_ALIASES = {"jev-latest", "jev-preview"}
MODELS = [
    {"name": "openjev-latest", "description": "Alias for the newest OpenJev release. Currently openjev-0.1.",
     "release_date": "2026-09-18"},
    {"name": "openjev-0.1", "description": "OpenJev 0.1: DiffusionGemma 26B-A4B (NVFP4) on vLLM's structured reads.",
     "release_date": "2026-09-18"},
    {"name": GEN_MODEL, "description": "DiffusionGemma 26B-A4B (NVFP4) text generation at POST /v1/chat/completions.",
     "release_date": "2026-09-18"},
]

# The encoder backends each serve one model, under its own name.
ENCODER_MODELS = {
    "laya": {"name": "laya-typed-decisions",
             "description": "Laya typed-decisions by Convai Innovations: a ModernBERT-large encoder (421M), "
                            "fine-tuned on the typed-decisions workflows. Text only, 1,024 tokens.",
             "release_date": "2026-09-22"},
    "verdict": {"name": "verdict-151m",
                "description": "Verdict by Heman10x: a ModernBERT-base + GLiClass encoder (151M), calibrated with "
                               "RLCD. Text only, 512 tokens, up to 24 choices.",
                "release_date": "2026-09-22"},
}


def served_models(backend):
    """(the model version a response names, the names a request may use, the /v1/models list)."""
    if backend in ENCODER_MODELS:
        m = ENCODER_MODELS[backend]
        return m["name"], {m["name"]} | SDK_ALIASES, [m]
    return MODEL_VERSION, MODEL_ALIASES | SDK_ALIASES, MODELS
