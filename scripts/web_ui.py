"""Serve a small local interface for comparing Baguette decoding methods."""

from __future__ import annotations

import argparse
import json
import math
import sys
import threading
import time
import traceback
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlsplit

import torch
from tokenizers import Tokenizer

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from specdec.baseline import generate, synchronize  # noqa: E402
from specdec.cache import allocate, forward_all  # noqa: E402
from specdec.cached_generation import generate_cached, generate_speculative_cached  # noqa: E402
from specdec.loader import assert_tokenizer_compatible, load_baguette  # noqa: E402
from specdec.speculative import generate_speculative  # noqa: E402
from specdec.lookup import generate_lookup  # noqa: E402


INDEX = Path(__file__).resolve().parents[1] / "web" / "index.html"
MAX_BODY_BYTES = 16_384
MAX_NEW_TOKENS = 256


@torch.inference_mode()
def prepare_mps_blocks(target) -> float:
    """Warm common block shapes once, using synthetic BOS tokens, not a demo.

    MPS can compile a shape on its first use. Move that one-time work into
    server preparation, for both ordinary and block verification forwards.
    """
    device = next(target.parameters()).device
    start = time.perf_counter()
    token = target.cfg.bos_id if target.cfg.bos_id is not None else 0
    for prefix in sorted({min(16, target.cfg.max_seq_len // 2),
                          min(64, target.cfg.max_seq_len // 2)}):
        width_limit = min(16, target.cfg.max_seq_len - prefix)
        caches = allocate(target, prefix + width_limit)
        ids = torch.full((1, prefix + width_limit), token, dtype=torch.long, device=device)
        forward_all(target, ids[:, :prefix], caches, 0, last_only=True)
        for width in range(1, width_limit + 1):
            forward_all(target, ids[:, :width], caches, prefix)
    synchronize(device)
    return time.perf_counter() - start


def _integer(payload: dict, key: str, default: int, minimum: int, maximum: int) -> int:
    value = payload.get(key, default)
    if isinstance(value, bool) or not isinstance(value, int) or not minimum <= value <= maximum:
        raise ValueError(f"{key} doit être un entier entre {minimum} et {maximum}")
    return value


def _number(payload: dict, key: str, default: float, minimum: float, maximum: float,
            *, strict_minimum: bool = False) -> float:
    value = payload.get(key, default)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{key} doit être un nombre")
    value = float(value)
    if not math.isfinite(value) or (value <= minimum if strict_minimum else value < minimum) or value > maximum:
        raise ValueError(f"{key} doit être entre {minimum} et {maximum}")
    return value


class ComparisonService:
    def __init__(self, target, draft, tokenizer: Tokenizer, device: torch.device,
                 *, draft_loader=None):
        self.target = target
        self.draft = draft
        self.draft_loader = draft_loader
        self.tokenizer = tokenizer
        self.device = device
        self.lock = threading.Lock()
        self.context_limit = target.cfg.max_seq_len
        self.cache_supported = all(layer.kind == "attn" for layer in target.layers)

    def info(self) -> dict:
        return {
            "device": self.device.type,
            "target_parameters": sum(p.numel() for p in self.target.parameters()),
            "draft_parameters": (sum(p.numel() for p in self.draft.parameters())
                                 if self.draft is not None else None),
            "draft_available": self.draft is not None or self.draft_loader is not None,
            "vocab_size": self.target.cfg.vocab_size,
            "context_limit": self.context_limit,
            "cache_supported": self.cache_supported,
        }

    def compare(self, payload: dict) -> dict:
        request_start = time.perf_counter()
        if not isinstance(payload, dict):
            raise ValueError("La requête doit contenir un objet JSON")
        prompt = payload.get("prompt", "")
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError("Le prompt ne peut pas être vide")
        if len(prompt) > 4000:
            raise ValueError("Le prompt est trop long pour cette interface")
        mode = payload.get("mode", "sampling")
        if mode not in ("sampling", "greedy"):
            raise ValueError("Mode de génération inconnu")
        greedy = mode == "greedy"
        strategy = payload.get("strategy", "auto")
        if strategy not in ("auto", "draft"):
            raise ValueError("Stratégie inconnue")
        max_new_tokens = _integer(payload, "max_new_tokens", 64, 1, MAX_NEW_TOKENS)
        draft_length = _integer(payload, "draft_length", 8 if strategy == "auto" else 2, 1, 16)
        seed = _integer(payload, "seed", 42, 0, 2**32 - 1)
        cache = payload.get("cache", self.cache_supported)
        if not isinstance(cache, bool):
            raise ValueError("cache doit être vrai ou faux")
        if cache and not self.cache_supported:
            raise ValueError("Le cache n'est disponible que pour les modèles sans DeltaNet")
        if strategy == "auto" and not cache:
            raise ValueError("Le mode automatique nécessite le cache d'attention")
        temperature = _number(payload, "temperature", 0.5, 0, 10, strict_minimum=True)
        top_k = _integer(payload, "top_k", min(20, self.target.cfg.vocab_size),
                         0, self.target.cfg.vocab_size)
        top_p = _number(payload, "top_p", 0.95, 0, 1, strict_minimum=True)

        ids = self.tokenizer.encode(prompt).ids
        if not ids:
            raise ValueError("Le prompt ne produit aucun token")
        if len(ids) + max_new_tokens > self.context_limit:
            raise ValueError(f"Prompt trop long : la limite commune est de {self.context_limit} tokens")

        settings = dict(max_new_tokens=max_new_tokens, greedy=greedy,
                        temperature=temperature, top_k=top_k, top_p=top_p,
                        seed=seed, eos_id=self.target.cfg.eos_id)
        baseline_fn = generate_cached if cache else generate
        speculative_fn = generate_speculative_cached if cache else generate_speculative
        warmup = {**settings, "max_new_tokens": min(4, max_new_tokens)}
        with self.lock:
            if strategy == "draft":
                if self.draft is None:
                    if self.draft_loader is None:
                        raise ValueError("Aucun checkpoint brouillon disponible ; utiliser le mode automatique")
                    draft = self.draft_loader()
                    if draft.cfg.vocab_size != self.target.cfg.vocab_size:
                        raise ValueError("Vocabulaire du brouillon incompatible")
                    self.draft = draft
                limit = min(self.context_limit, self.draft.cfg.max_seq_len)
                if len(ids) + max_new_tokens > limit:
                    raise ValueError(f"Le brouillon limite cette requête à {limit} tokens")
                if cache and not all(layer.kind == "attn" for layer in self.draft.layers):
                    raise ValueError("Le brouillon ne prend pas en charge le cache")

            def run_spec(options):
                if strategy == "auto":
                    return generate_lookup(self.target, ids, draft_length=draft_length, **options)
                return speculative_fn(self.target, self.draft, ids,
                                      draft_length=draft_length, **options)

            baseline_fn(self.target, ids, **warmup)
            run_spec(warmup)
            baseline = baseline_fn(self.target, ids, **settings)
            speculative = run_spec(settings)
        if greedy and baseline.token_ids != speculative.token_ids:
            raise RuntimeError("La sortie greedy spéculative diffère de la cible")

        return {
            "device": self.device.type,
            "greedy": greedy,
            "strategy": strategy,
            "execution": ("draft" if strategy == "draft" else
                          "lookup" if speculative.verification_blocks else "target"),
            "same_tokens": baseline.token_ids == speculative.token_ids,
            "request_seconds": time.perf_counter() - request_start,
            "baseline": {
                "text": self.tokenizer.decode(ids + baseline.token_ids,
                                              skip_special_tokens=False),
                "generated_tokens": len(baseline.token_ids),
                "tokens_per_second": baseline.tokens_per_second,
                "time_to_first_token": baseline.time_to_first_token,
                "total_seconds": baseline.total_seconds,
            },
            "speculative": {
                "text": self.tokenizer.decode(ids + speculative.token_ids,
                                              skip_special_tokens=False),
                "generated_tokens": len(speculative.token_ids),
                "tokens_per_second": speculative.tokens_per_second,
                "time_to_first_token": speculative.time_to_first_token,
                "total_seconds": speculative.total_seconds,
                "acceptance_rate": speculative.acceptance_rate,
                "proposed_tokens": speculative.proposed,
                "accepted_tokens": speculative.accepted,
                "accepted_per_block": speculative.accepted_per_block,
                "target_forward_passes": speculative.target_forward_passes,
                "draft_forward_passes": speculative.draft_forward_passes,
                "verification_blocks": getattr(speculative, "verification_blocks", speculative.blocks),
                "fallback_tokens": getattr(speculative, "fallback_tokens", 0),
                "speedup": speculative.tokens_per_second / baseline.tokens_per_second,
            },
        }


def make_handler(service: ComparisonService):
    index = INDEX.read_bytes()

    class Handler(BaseHTTPRequestHandler):
        def respond(self, status: int, body: bytes, content_type: str) -> None:
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.end_headers()
            self.wfile.write(body)

        def json_response(self, status: int, data: dict) -> None:
            body = json.dumps(data, ensure_ascii=False, allow_nan=False).encode("utf-8")
            self.respond(status, body, "application/json; charset=utf-8")

        def do_GET(self) -> None:
            path = urlsplit(self.path).path
            if path == "/":
                self.respond(200, index, "text/html; charset=utf-8")
            elif path == "/api/info":
                self.json_response(200, service.info())
            elif path == "/favicon.ico":
                self.respond(204, b"", "image/x-icon")
            else:
                self.json_response(404, {"error": "Page introuvable"})

        def do_POST(self) -> None:
            if urlsplit(self.path).path != "/api/compare":
                self.json_response(404, {"error": "Page introuvable"})
                return
            if self.headers.get("Content-Type", "").split(";", 1)[0] != "application/json":
                self.json_response(415, {"error": "Envoyer du JSON"})
                return
            try:
                size = int(self.headers.get("Content-Length", "0"))
                if not 1 <= size <= MAX_BODY_BYTES:
                    raise ValueError("Requête trop grande ou vide")
                payload = json.loads(self.rfile.read(size))
                self.json_response(200, service.compare(payload))
            except ValueError as exc:
                self.json_response(400, {"error": str(exc)})
            except Exception:
                traceback.print_exc()
                self.json_response(500, {"error": "Erreur pendant la génération ; voir le terminal"})

    return Handler


def main() -> None:
    repo = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baguette-source", type=Path, default=repo.parent / "LLM")
    parser.add_argument("--target-checkpoint", type=Path)
    parser.add_argument("--draft-checkpoint", type=Path)
    parser.add_argument("--tokenizer", type=Path)
    parser.add_argument("--draft-tokenizer", type=Path)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda", "mps"), default="auto")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()
    if not 1 <= args.port <= 65535:
        parser.error("port must be between 1 and 65535")

    source = args.baguette_source.resolve()
    target_checkpoint = args.target_checkpoint or source / "baguette-123m-sft.pt"
    draft_checkpoint = (args.draft_checkpoint or
                        source / "runs/specdec-draft-nano/distill-run1/draft-step3000.pt")
    tokenizer_path = args.tokenizer or source / "tokenizer.json"
    draft_tokenizer_path = (args.draft_tokenizer or
                            source / "runs/specdec-draft-nano/tokenizer.json")
    for path in (target_checkpoint, tokenizer_path):
        if not path.is_file():
            parser.error(f"fichier introuvable : {path}")
    name = ("cuda" if torch.cuda.is_available() else
            "mps" if torch.backends.mps.is_available() else "cpu") if args.device == "auto" else args.device
    device = torch.device(name)
    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    target = load_baguette(source, target_checkpoint, device)
    if target.cfg.vocab_size != tokenizer.get_vocab_size():
        parser.error("la taille du vocabulaire du tokenizer ne correspond pas aux modèles")
    if device.type == "mps" and all(layer.kind == "attn" for layer in target.layers):
        print("Préparation des blocs MPS…", flush=True)
        elapsed = prepare_mps_blocks(target)
        print(f"Préparation terminée en {elapsed:.2f} s.", flush=True)

    def load_draft():
        assert_tokenizer_compatible(tokenizer_path, draft_tokenizer_path)
        return load_baguette(source, draft_checkpoint, device)

    available = draft_checkpoint.is_file() and draft_tokenizer_path.is_file()
    service = ComparisonService(target, None, tokenizer, device,
                                draft_loader=load_draft if available else None)
    server = ThreadingHTTPServer(("127.0.0.1", args.port), make_handler(service))
    print(f"Interface prête : http://127.0.0.1:{args.port} ({name})", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nArrêt de l'interface.")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
