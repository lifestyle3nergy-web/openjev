"""Offline tests for the encoder backends (Laya, Verdict): stubbed models, the real API."""
import math

import httpx
import pytest
from fastapi.testclient import TestClient

from openjev import encoders
from openjev.api import create_app
from openjev.config import Settings, parse_routes
from openjev.encoders import EncoderEngine, VerdictEngine, verdict_prompt

REQUEST = {
    "state": "I was charged twice this month.",
    "model": "laya-1.0",
    "questions": {
        "team": {"type": "choice", "instructions": "Which team should handle it?",
                 "criteria": {"outage": "service down", "billing": "charges, refunds", "feature": None}},
        "tone": {"type": "score", "instructions": "How upset is the customer?", "criteria": ["calm", "annoyed", "furious"]},
        "urgent": {"type": "noul", "instructions": "Does the customer need a reply within the hour?"},
    },
}


class FakeEngine(EncoderEngine):
    model_name = "laya-1.0"

    def load(self):
        self.reads = []

    def read_batch(self, state, qs):
        self.reads.append((state, qs))
        # the second option 70%, the rest share 30%
        out = []
        for q in qs:
            n = len(q["choices"])
            p = [0.3 / (n - 1)] * n
            p[1] = 0.7
            out.append(p)
        return out, 99


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setitem(encoders.ENGINES, "laya", FakeEngine)
    with TestClient(create_app(Settings(backend="laya"))) as c:
        yield c


def test_answers_in_jevs_shapes(client):
    r = client.post("/v1/systemone", json=REQUEST)
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["model"] == "laya-1.0"
    assert body["usage"] == {"input_tokens": 99, "output_tokens": 0}
    a = body["answers"]
    assert list(a) == ["team", "tone", "urgent"]
    assert a["team"]["choice"] == "billing"
    assert a["team"]["probabilities"]["billing"] == pytest.approx(0.7)
    assert a["tone"]["score"] == pytest.approx(0.15 * 0 + 0.7 * 1 + 0.15 * 2)
    assert a["tone"]["legend"] == {"0": "calm", "1": "annoyed", "2": "furious"}
    assert a["urgent"] == {"type": "noul", "noul": pytest.approx(0.3)}  # P(yes) is the first option


def test_server_timing_counts_the_read(client):
    timing = client.post("/v1/systemone", json=REQUEST).headers["server-timing"]
    assert timing.startswith("model;dur=")


def test_typesafe_sdk_default_model_is_accepted(client):
    from typesafe_sdk import TypeSafeClient

    sdk = TypeSafeClient(api_key="x", base_url="http://testserver", http_client=client)
    r = sdk.system_one(REQUEST["state"], REQUEST["questions"])
    assert r.choices["team"].choice == "billing"


def test_models_and_unknown_model(client):
    assert [m["name"] for m in client.get("/v1/models").json()["models"]] == ["laya-1.0"]
    r = client.post("/v1/systemone", json=dict(REQUEST, model="openjev-latest"))
    assert r.status_code == 400
    assert r.json()["detail"]["message"] == "Unknown model: openjev-latest"


def test_no_text_generation(client):
    r = client.post("/v1/chat/completions", json={"model": "diffusiongemma-26b", "messages": [{"role": "user", "content": "hi"}]})
    assert r.status_code == 404


@pytest.mark.parametrize("extra", [{"steps": 2}, {"samples": 4}, {"think": 64}, {"sequential": True},
                                   {"images": ["data:image/png;base64,iVBORw0KGgo="]}])
def test_read_options_are_refused(client, extra):
    r = client.post("/v1/systemone", json=dict(REQUEST, **extra))
    assert r.status_code == 400
    assert "does not support" in r.json()["detail"]


def test_options_left_at_their_defaults_are_fine(client):
    r = client.post("/v1/systemone", json=dict(REQUEST, steps=1, samples=1, think=0, sequential=False))
    assert r.status_code == 200


def test_one_option_and_one_level_need_no_read(client):
    q = {"only": {"type": "choice", "criteria": {"yes": None}}, "lvl": {"type": "score", "criteria": ["one"]}}
    r = client.post("/v1/systemone", json=dict(REQUEST, questions=q))
    assert r.json()["answers"]["only"]["probabilities"] == {"yes": 1.0}
    assert r.json()["usage"]["input_tokens"] == 0
    assert client.app.state.engine.reads == []


def test_many_questions_are_read_in_batches(client):
    q = {f"n{i}": {"type": "noul", "instructions": "x"} for i in range(40)}
    r = client.post("/v1/systemone", json=dict(REQUEST, questions=q))
    assert list(r.json()["answers"]) == list(q)
    assert [len(qs) for _, qs in client.app.state.engine.reads] == [16, 16, 8]
    assert r.json()["usage"]["input_tokens"] == 3 * 99


def test_limits(client):
    many = {"c": {"type": "choice", "criteria": {f"o{i}": None for i in range(256)}}}
    assert client.post("/v1/systemone", json=dict(REQUEST, questions=many)).json() == {
        "detail": "Too many choices. Must have at most 255 choices."}
    empty = {"c": {"type": "choice", "criteria": {}}}
    assert client.post("/v1/systemone", json=dict(REQUEST, questions=empty)).status_code == 400


def test_routes_forward_other_models(monkeypatch):
    monkeypatch.setenv("OPENJEV_MODEL_ROUTES", "verdict-1.4=http://verdict:8080/, laya-1.0=http://laya:8080")
    monkeypatch.setitem(encoders.ENGINES, "laya", FakeEngine)
    seen = []

    def handler(req):
        seen.append(req)
        return httpx.Response(200, json={"model": "verdict-1.4", "answers": {}, "usage": {}})

    with TestClient(create_app(Settings(backend="laya", origin_secret="s"))) as c:
        c.app.state.routes = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        names = [m["name"] for m in c.get("/v1/models", headers={"x-origin-secret": "s"}).json()["models"]]
        assert names == ["laya-1.0", "verdict-1.4"]  # its own model is not listed twice
        r = c.post("/v1/systemone", json=dict(REQUEST, model="verdict-1.4"), headers={"x-origin-secret": "s"})
        assert r.json()["model"] == "verdict-1.4"
        assert str(seen[0].url) == "http://verdict:8080/v1/systemone"
        assert seen[0].headers["x-origin-secret"] == "s"
        # its own model is answered here, not forwarded
        assert c.post("/v1/systemone", json=REQUEST, headers={"x-origin-secret": "s"}).json()["model"] == "laya-1.0"
        assert len(seen) == 1
        # an unreachable route is a 503, like an unreachable vLLM
        c.app.state.routes = httpx.AsyncClient(transport=httpx.MockTransport(lambda req: (_ for _ in ()).throw(httpx.ConnectError("x"))))
        assert c.post("/v1/systemone", json=dict(REQUEST, model="verdict-1.4"), headers={"x-origin-secret": "s"}).status_code == 503


def test_parse_routes():
    assert parse_routes("") == {}
    assert parse_routes("a=http://x/, b = http://y") == {"a": "http://x", "b": "http://y"}
    with pytest.raises(ValueError):
        parse_routes("a")


def test_verdict_prompt_matches_upstream_contract():
    # core/formatting.py of Verdict-open-jev: labels first, then the text
    q = {"type": "choice", "instructions": "Which team?", "choices": [("outage", "service down"), ("sales", "")]}
    assert verdict_prompt(q, "ctx") == (
        "<<LABEL>>It is service down<<LABEL>>It is sales<<LABEL>>insufficient evidence<<SEP>>Question: Which team?\n\nContext:\nctx", 3)
    q = {"type": "score", "instructions": "How bad?", "choices": [("0", "low"), ("1", "high")]}
    assert verdict_prompt(q, "ctx")[0].startswith("<<LABEL>>low (Value: 0.0)<<LABEL>>high (Value: 1.0)<<LABEL>>")
    q = {"type": "noul", "instructions": "It is urgent", "choices": []}
    assert verdict_prompt(q, "ctx") == (
        ("<<LABEL>>true: It is urgent<<LABEL>>false: not It is urgent<<LABEL>>insufficient evidence"
         "<<SEP>>Context:\nctx\n\nEvaluate proposition: It is urgent"), 3)


def test_verdict_read_calibrates_and_drops_abstention():
    torch = pytest.importorskip("torch")

    class Tok:
        def __call__(self, prompts, **kw):
            n = len(prompts)
            return _Batch({"input_ids": torch.ones(n, 4, dtype=torch.long), "attention_mask": torch.ones(n, 4, dtype=torch.long)})

    class _Batch(dict):
        def to(self, device):
            return self

    class Model:
        def __call__(self, **kw):
            logits = torch.full((2, 25), -100.0)
            logits[0, :3] = torch.tensor([2.0, 0.0, 5.0])  # noul: true, false, abstain
            logits[1, :4] = torch.tensor([1.0, 3.0, 0.0, 0.0])
            return type("O", (), {"logits": logits})

    eng = VerdictEngine.__new__(VerdictEngine)
    eng.torch, eng.device, eng.tok, eng.model = torch, "cpu", Tok(), Model()
    eng.temperature, eng.per_k = 2.0, {3: 1.0}
    qs = [{"type": "noul", "instructions": "x", "choices": [("yes", ""), ("no", "")]},
          {"type": "choice", "instructions": "y", "choices": [("a", ""), ("b", ""), ("c", "")]}]
    eng.s = Settings()
    probs, tokens = eng.read("state", qs)
    assert tokens == 8
    # k=3 uses its own temperature (1.0); the abstention's mass is renormalized away
    assert probs[0] == pytest.approx([1 / (1 + math.exp(-2.0)), 1 / (1 + math.exp(2.0))])
    # k=4 has no entry, so the global temperature (2.0) applies
    e = [math.exp(v / 2.0) for v in (1.0, 3.0, 0.0)]
    assert probs[1] == pytest.approx([v / sum(e) for v in e])
