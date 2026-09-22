"""Small encoder System One models: Laya and Verdict, one per container.

Selected with OPENJEV_BACKEND=laya or OPENJEV_BACKEND=verdict. Each reads the
questions of a request in batched forward passes of a bidirectional encoder and
a classification head, one sequence per question, so an answer is a distribution over the caller's
options exactly as with DiffusionGemma. The API layer, the error contract and
the answer shapes are shared with the vLLM backend; only the read differs.

- laya-1.0: Laya by Nandakishor M / Convai Innovations
  (github.com/NandhaKishorM/laya), checkpoint convaiinnovations/laya-typed-decisions
  (ModernBERT-large, 421M, Apache-2.0), through the `laya` package. Its system_one already takes Jev's
  questions; its answers are reshaped here to Jev's exact shapes.
- verdict-1.4: Verdict by Heman10x, checkpoint heman10x/rlcd-modernbert-151m (ModernBERT-base + GLiClass, 151M,
  Apache-2.0). The prompt format and the per-option-count temperatures follow
  core/formatting.py and core/engine_encoder.py of
  github.com/Heman10x-NGU/Verdict-open-jev (v1.4 inference). That package is
  not installed: it ships top-level `openjev` and `core` packages.

Neither model has images, denoise steps, noise draws or a thought, so those
request options are refused with a 400. torch, gliclass and laya are imported
lazily so the vLLM image never needs them.
"""
import asyncio
import json
import logging
import math
import os
from concurrent.futures import ThreadPoolExecutor

from .engine import Overloaded, SchemaError, text_of, to_answer

log = logging.getLogger("openjev")


class EncoderEngine:
    """The Engine contract (decide, close) for an in-process encoder. The model
    lives on one thread, from loading on; a request is one call on it."""

    model_name = ""
    max_choices = 128

    def __init__(self, settings):
        self.s = settings
        self.waiting = 0
        self.pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix=f"openjev-{self.model_name}")
        self.pool.submit(self.load).result()

    async def close(self):
        self.pool.shutdown(wait=True, cancel_futures=True)

    def load(self):
        raise NotImplementedError

    def read(self, state, qs):
        """[probabilities per question, in the caller's option order], input tokens.
        Runs on the model's thread. Questions are read in batches of at most
        OPENJEV_ENCODER_BATCH, because every question is a full-length sequence
        and a shared GPU has little room: Laya moves itself to the CPU for good
        after one CUDA out-of-memory error."""
        n = self.s.encoder_batch
        probs, tokens = [], 0
        for i in range(0, len(qs), n):
            p, t = self.read_batch(state, qs[i:i + n])
            probs += p
            tokens += t
        return probs, tokens

    def read_batch(self, state, qs):
        raise NotImplementedError

    def build_schema(self, questions):
        """The questions to read, and the answers that need no read: a choice with
        one option or a score with one level, as in Engine.build_schema."""
        qs, forced = [], {}
        for qid, q in questions.items():
            loc = ("body", "questions", qid, "criteria")
            kind, crit = q["type"], q.get("criteria")
            if kind == "noul":
                crit = crit or {}
                choices = [("yes", text_of(crit.get("true"))), ("no", text_of(crit.get("false")))]
            elif kind == "choice":
                if not crit:
                    raise SchemaError(f"Choice question must have at least one choice: {qid}", loc)
                if len(crit) == 1:
                    only = next(iter(crit))
                    forced[qid] = {"type": "choice", "choice": only, "probabilities": {only: 1.0}, "confidence": 1.0}
                    continue
                if len(crit) > self.max_choices:
                    raise SchemaError(f"Too many choices. Must have at most {self.max_choices} choices.", loc)
                choices = [(name, text_of(desc)) for name, desc in crit.items()]
            elif kind == "score":
                if len(crit) > 10:
                    raise SchemaError("Too many score levels. Must have at most 10 levels.", loc)
                if len(crit) == 1:
                    forced[qid] = {"type": "score", "score": 0.0, "legend": {"0": crit[0]},
                                   "probabilities": {"0": 1.0}, "confidence": 1.0}
                    continue
                choices = [(str(i), text_of(c)) for i, c in enumerate(crit)]
            else:
                raise SchemaError(f"unknown question type {kind!r}", ("body", "questions", qid, "type"))
            qs.append({"key": qid, "type": kind, "instructions": text_of(q.get("instructions")),
                       "criteria": crit, "choices": choices,
                       "legend": list(crit) if kind == "score" else None})
        return qs, forced

    async def decide(self, questions, state, seed, images=None, options=None):
        """Engine.decide's contract. ``seed`` is unused: a read is deterministic."""
        opts = options or {}
        unsupported = [("images", bool(images)), ("steps", (opts.get("steps") or 1) > 1),
                       ("samples", (opts.get("samples") or 1) > 1), ("think", bool(opts.get("think"))),
                       ("sequential", bool(opts.get("sequential")))]
        for field, used in unsupported:
            if used:
                raise SchemaError(f"{self.model_name} does not support {field}", ("body", field))
        if self.waiting >= self.s.max_queue:
            raise Overloaded(f"{self.model_name} is at capacity. Retry shortly.")
        qs, forced = self.build_schema(questions)
        self.waiting += 1
        try:
            probs, tokens = (await asyncio.get_running_loop().run_in_executor(self.pool, self.read, state, qs)
                             if qs else ([], 0))
        finally:
            self.waiting -= 1
        answers = dict(forced)
        for q, p in zip(qs, probs):
            answers[q["key"]] = to_answer(q, p)
        return {k: answers[k] for k in questions}, tokens, 0


class LayaEngine(EncoderEngine):
    model_name = "laya-1.0"

    def load(self):
        import laya

        self.agent = laya.load(self.s.laya_model, device=self.s.device or None)
        # laya falls back to the CPU instead of failing, at about 10x the latency
        self.device = self.agent.device
        if self.device.type == "cpu" and self.s.device != "cpu" and _cuda_available():
            raise RuntimeError("laya could not use the GPU; see the warning above")

    def read_batch(self, state, qs):
        # instructions go as text: laya would send a missing one to the model as "null"
        questions = {str(i): {"type": q["type"], "instructions": q["instructions"], "criteria": q["criteria"]}
                     for i, q in enumerate(qs)}
        try:
            out = self.agent.system_one(state, questions)
        except ValueError:  # the options overflow the head's token budget
            raise SchemaError(f"Too many choices for {self.model_name}: a question's options must fit in "
                              f"{self.agent.cfg.get('head_max_len')} tokens.") from None
        if self.agent.device != self.device:
            # A CUDA out-of-memory error moved the model to the CPU for good. Exit, so the
            # container restarts on the GPU instead of serving slowly with no error.
            log.error("laya moved from %s to %s after a GPU error; exiting", self.device, self.agent.device)
            os._exit(3)
        probs = []
        for i, q in enumerate(qs):
            a = out["answers"][str(i)]
            if q["type"] == "noul":
                p = [a["noul"], 1.0 - a["noul"]]
            else:
                p = list(a["probabilities"].values())
                z = sum(p)  # laya rounds each to 4 places
                p = [v / z for v in p]
            probs.append(p)
        return probs, out["usage"]["input_tokens"]


def _cuda_available():
    import torch

    return torch.cuda.is_available()


# Verdict's prompt contract (core/formatting.py)
LABEL_MARKER, SEP_MARKER = "<<LABEL>>", "<<SEP>>"
ABSTAIN = "insufficient evidence"
VERDICT_MAX_LEN = 512  # the weights were trained on short states; v1.4 cut 1024 to 512


def verdict_prompt(q, context):
    """Verdict's model input for one question: the label prefix, then the text. Every
    question gets an "insufficient evidence" option as its last label."""
    ins = q["instructions"]
    if q["type"] == "noul":
        labels = [f"true: {ins}", f"false: not {ins}"]
        text = f"Context:\n{context}\n\nEvaluate proposition: {ins}"
    else:
        if q["type"] == "choice":
            labels = [f"It is {desc or name}" for name, desc in q["choices"]]
        else:
            labels = [f"{desc} (Value: {float(i)})" for i, (_, desc) in enumerate(q["choices"])]
        text = f"Question: {ins}\n\nContext:\n{context}" if ins else context
    labels.append(ABSTAIN)
    return "".join(LABEL_MARKER + label for label in labels) + SEP_MARKER + text, len(labels)


class VerdictEngine(EncoderEngine):
    model_name = "verdict-1.4"
    max_choices = 24  # the head has 25 logits, the last kept for "insufficient evidence"

    def load(self):
        import torch
        from gliclass import GLiClassModel
        from huggingface_hub import hf_hub_download
        from transformers import AutoTokenizer

        self.torch = torch
        self.device = self.s.device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = GLiClassModel.from_pretrained(self.s.verdict_model).to(self.device).eval()
        self.tok = AutoTokenizer.from_pretrained(self.s.verdict_model)
        path = self.s.verdict_model
        cal_file = (os.path.join(path, "calibrator.json") if os.path.isdir(path)
                    else hf_hub_download(path, "calibrator.json"))
        with open(cal_file) as f:
            cal = json.load(f)
        self.temperature = float(cal["temperature"])
        self.per_k = {int(k): float(v) for k, v in cal.get("per_k", {}).items()}

    def read_batch(self, state, qs):
        torch = self.torch
        context = state if isinstance(state, str) else json.dumps(state, ensure_ascii=False)
        prompts, ks = zip(*(verdict_prompt(q, context) for q in qs))
        batch = self.tok(list(prompts), padding=True, truncation=True, max_length=VERDICT_MAX_LEN,
                         return_tensors="pt").to(self.device)
        with torch.inference_mode():
            logits = self.model(**batch).logits.float().cpu()
        probs = []
        for row, k in zip(logits, ks):
            z = row[:k] / self.per_k.get(k, self.temperature)
            p = torch.softmax(z, -1)[:-1]
            # Jev's distribution is over the caller's options: the abstention mass is dropped
            p = (p / p.sum()).tolist()
            probs.append(p if math.isfinite(sum(p)) else [1.0 / (k - 1)] * (k - 1))
        return probs, int(batch["attention_mask"].sum())


ENGINES = {"laya": LayaEngine, "verdict": VerdictEngine}
