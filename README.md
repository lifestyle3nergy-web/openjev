# OpenJev

**Fast, calibrated, typed decisions from an open model.** OpenJev is an open-source
"System One" decision server. Send it a state and a set of typed questions (yes/no, choice,
score). It returns a probability and a confidence for every answer in tens of milliseconds.
It reads the answers straight off the model's probabilities. It generates no text and parses
nothing, so an answer cannot go off-schema. Questions can also ask about images.

OpenJev speaks the same wire API as TypeSafe's
[Jev](https://typesafe.ai/blog/introducing-system-one-models-and-jev), so their SDKs work
against it unchanged. It runs
[DiffusionGemma 26B-A4B](https://huggingface.co/nvidia/diffusiongemma-26B-A4B-it-NVFP4)
(Apache-2.0) two ways: through vLLM on an NVIDIA GPU, or in-process through MLX on Apple
silicon. Both backends also serve text generation on an OpenAI-compatible
`/v1/chat/completions`, at no extra GPU memory.

> **Hosted for free on [Codiv](https://codiv.ai)**, an inference platform for open System One
> models: sign up and get 100M input tokens, no card required. `https://api.codiv.ai/v1/systemone`

OpenJev is an independent project. It is not affiliated with or endorsed by TypeSafe AI.

## Try it

```bash
pip install typesafe-sdk
export TYPESAFE_BASE_URL=https://api.codiv.ai   # or http://127.0.0.1:8080 for your own server
export TYPESAFE_API_KEY=sk-codiv-...
```

```python
from typesafe_sdk import TypeSafeClient

client = TypeSafeClient()
r = client.system_one(
    "Everything is down and we have a demo with our biggest client at noon.",
    {
        "urgent": {"type": "noul", "instructions": "Does the customer need a reply within the hour?"},
        "team":   {"type": "choice", "instructions": "Which team should handle it?",
                   "criteria": {"outage": "service down", "billing": "charges, refunds", "feature": "requests, how-to"}},
        "tone":   {"type": "score", "instructions": "How upset is the customer?",
                   "criteria": ["calm", "annoyed", "furious"]},
    },
)
r.nouls["urgent"].noul        # 1.00
r.choices["team"].choice      # "outage", confidence 1.00
r.scores["tone"].score        # 2.00 (expected level, 0-indexed)
```

Or with curl:

```bash
curl https://api.codiv.ai/v1/systemone \
  -H "Authorization: Bearer $TYPESAFE_API_KEY" -H "Content-Type: application/json" \
  -d '{"model": "openjev-latest", "state": "I was charged twice this month.",
       "questions": {"is_billing": {"type": "noul", "instructions": "Is this a billing issue?"}}}'
```

## API

| | |
|---|---|
| `POST /v1/systemone` | `{state, model, questions}` → `{model, answers, usage}` |
| `POST /v1/chat/completions` | OpenAI-style text generation with model `diffusiongemma-26b` ([below](#text-generation)) |
| `GET /v1/models` | `openjev-0.1` and its alias `openjev-latest`. The server also accepts `jev-latest` and `jev-preview`, so TypeSafe SDK defaults work. The list includes `diffusiongemma-26b`, and the [small encoder models](#small-encoder-models) when they run beside it. |

Question types:

- **`noul`** (yes/no): takes optional `criteria: {true, false}` and returns `{noul: P(yes)}`.
- **`choice`**: takes `criteria: {name: description}` and returns `{choice, probabilities, confidence}`.
- **`score`**: takes `criteria: [level0, level1, …]` (2–10 levels) and returns `{score: Σ i·pᵢ, legend, probabilities, confidence}`.

`confidence` is `1 − H(p)/ln K`. It is 1 when the model is certain and 0 when the distribution
is uniform. `usage.input_tokens` counts prompt tokens, including image tokens.
`usage.output_tokens` is 0 unless you set `think` (below).

Every response carries a `Server-Timing` header that splits where the time went:

```
server-timing: model;dur=41.2, server;dur=2.8, total;dur=44.0
```

`model` is time spent waiting on the model, summed over the request's reads, so it can exceed
`total` when reads run in parallel. `server` is the rest of the wall clock: compiling the
schema, tokenizing, validating and serializing. Neither includes your network.

Errors follow the same shapes as Jev, checked against the live API:

- `422` with a FastAPI validation list for a field of the wrong shape.
- `400` with the reason as plain text for a question the server cannot ask (no options, too many
  options, or too many score levels).
- `400` `api_usage_error` for an unknown model or question type.
- `{"detail": {"error_type", "message"}}` for auth errors (`401`/`403`).
- `429` for rate limits, and `529` when the server is overloaded.

Known differences from Jev:

- Model names are OpenJev's own. `jev-latest` and `jev-preview` are aliases. A pinned Jev
  version such as `jev-1.13.0` answers `400` `Unknown model`.
- A choice with one option, or a score with one level, gets a direct answer (probability 1)
  with no read, so it bills no tokens. Jev returns the same values and bills for the read.
- A body nested a thousand levels deep answers `422`. Jev answers `500`.
- The server answers many questions in chunks of about 12 per read, still in parallel.

### Extensions

These optional request fields are OpenJev additions to Jev's contract. Leave them out and a
request behaves exactly like Jev. TypeSafe's SDKs never send them. They come from the example
server in vllm-project/vllm#57250.

| Field | Values | What it does | Cost |
|---|---|---|---|
| `images` | up to 8 | Images the questions ask about, placed ahead of the state. Each is a `data:image/...;base64,` URL or `{"content_type", "base64"}`. JPEG, PNG, WebP or GIF, 5 MB each. | about 280 input tokens per image |
| `steps` | 1–8, default 1 | Denoise steps per read. More steps let the answers settle against each other. | same tokens, more GPU time |
| `samples` | 1–32 | Read N times with different noise and average. Replaces the automatic re-reads. `samples: 1` pins one read, which is the fastest answer. | N × input tokens |
| `think` | 0–4096 tokens | The model writes a thought first, then reads the answers after it. The number is a hard cap. A thought that hits the cap gets cut off, so give multi-step problems 512 or more. | input tokens twice, plus the thought as output tokens |
| `sequential` | `true` | For long question lists answered in chunks: read the chunks in order. Each chunk sees the answers already chosen. | one read per chunk, run one after another |

```bash
curl https://api.codiv.ai/v1/systemone \
  -H "Authorization: Bearer $TYPESAFE_API_KEY" -H "Content-Type: application/json" \
  -d '{"model": "openjev-latest", "state": "Look at the photo.",
       "images": ["data:image/jpeg;base64,/9j/4AAQ..."],
       "questions": {"hotdog": {"type": "noul", "instructions": "The photo shows a hot dog"}}}'
```

`think` and `sequential` need a text state, so you cannot combine them with `images`. Such a
request gets a 400.

### Text generation

`POST /v1/chat/completions` generates text with the same DiffusionGemma, OpenAI style. Tools
that talk to a chat model can point their base URL at OpenJev. Use model `diffusiongemma-26b`.
Both backends serve it, streaming and not, and a client cannot tell them apart except where
this section says so.

vLLM refuses some fields for diffusion models, so OpenJev adjusts requests instead of failing
them:

- It ignores `temperature`, `seed`, `min_p`, `logit_bias`, penalties and `reasoning`.
- `response_format` (`json_object` or `json_schema`) becomes an instruction to reply with JSON
  only. The server returns the first JSON object in the reply as the content.
- `max_tokens` defaults to 1024, with a cap of 8192. Thinking stays off unless you set
  `chat_template_kwargs: {"enable_thinking": true}`. The thought then comes back separately in
  `message.reasoning`, never in `content`.
- `stream: true` streams server-sent events and always ends with a usage chunk.
- The endpoint supports tools (`tools`, `tool_choice`).

On MLX, three of these differ. `tools`, `tool_choice`, `logprobs` and `top_logprobs` are
accepted and ignored, not answered. A `stop` string that is more than one token is dropped,
because only a single token can end a denoised block. And there is no `message.reasoning`: the
model's thought channel is stripped from the reply, so `enable_thinking` changes the answer but
does not return the thought.

Generation denoises the output in 64-token blocks, so it costs far more GPU time than a System
One read. At most 8 generations run at once, so they never crowd out reads. On MLX the model
runs on one thread, so generations queue rather than overlap, and a client that disconnects
stops the reply at the next block instead of the next token.

```python
from openai import OpenAI

client = OpenAI(base_url="https://api.codiv.ai/v1", api_key="sk-codiv-...")
r = client.chat.completions.create(
    model="diffusiongemma-26b",
    messages=[{"role": "user", "content": "Summarize: 3 nonstop flights, cheapest $212 on Delta."}],
)
print(r.choices[0].message.content)
```

## How it works

DiffusionGemma is a discrete diffusion model. It denoises a whole canvas of tokens per forward
pass instead of generating left to right. OpenJev uses that to read answers instead of writing
them.

It builds a canvas where the only masked tokens are the answer slots, one token per question:

```
canvas in                 one read-only pass         answer out
  q1: [?]        ──►      P(yes) 0.001        ──►    noul  0.001
  q2: [?]                 P(A) 0.000                 choice "billing"
                          P(B) 0.999                 confidence 0.997
                          P(C) 0.000
  q3: [?]                 P(0) 0.000                 score 1.00
                          P(1) 0.996
                          P(2) 0.004
```

Every label is a single token: `yes`/`no` for a `noul`, `A`/`B`/`C` for a choice, `0`/`1`/`2`
for a score. The model never writes into those slots. One read-only pass returns the
probability distribution over each slot, and that distribution **is** the answer. The numbers
above are a real read of "The invoice looks wrong again. Second time this quarter.": not
urgent, billing, mildly annoyed.

Two consequences follow. An answer cannot go off-schema, because the read scores only the label
tokens. And the confidence comes from the model's own distribution, not from a number the model
reports about itself.

If any slot is uncertain (entropy > 0.1), OpenJev reads three more times with fresh noise and
averages the four results. The policy is all or nothing: one read, or four. One uncertain question
causes a re-read of all the questions in the request. The extra reads are the server's policy, so
they add no tokens to `usage`. Set `samples: 1` to pin a single read. Question ids never reach the
model: it sees `q1`, `q2`, `q3`.

The vLLM side of this is
[vllm-project/vllm#57250](https://github.com/vllm-project/vllm/pull/57250), merged on 2026-09-22,
which adds seeded canvases, read-only steps, step caps and pinned canvas positions for
DiffusionGemma. `openjev/engine.py` is adapted from
that PR's `structured_server.py` example. It adds async I/O, bounded concurrency and
backpressure.

## Run your own

Two backends serve the same `/v1/systemone`. Pick by hardware:

| | vLLM (default) | MLX |
|---|---|---|
| Hardware | NVIDIA GPU, 24 GB or more | Apple silicon, about 16 GB free |
| Setup | Docker image | `pip install -e '.[mlx]'` |
| Reads | up to 64 in flight | one at a time |
| `images` | yes | yes |
| `steps` > 1 | yes | yes |
| `think` | yes | yes |
| Text generation | yes | yes, streaming included |

### NVIDIA GPU

You need an NVIDIA GPU with at least 24 GB of memory for the NVFP4 checkpoint (tested on an
RTX PRO 6000 Blackwell, sm_120).

A prebuilt image is on Docker Hub, so there is nothing to compile.
[`razorback16/openjev`](https://hub.docker.com/r/razorback16/openjev) runs vLLM's DiffusionGemma
structured reads in one container with the Jev-compatible API server, on CUDA 13. It pins the
upstream vLLM commit shown below and makes the two changes listed under [Caveats](#caveats).

```bash
git clone https://github.com/razorback16/openjev && cd openjev
docker compose up -d          # OpenJev on 127.0.0.1:8080 once the model has loaded
curl localhost:8080/v1/models
```

Or without compose:

```bash
docker run -d --gpus all --ipc=host -p 127.0.0.1:8080:8080 \
  -v ~/.cache/huggingface:/root/.cache/huggingface razorback16/openjev:0.5.0
```

The model weights (about 18 GB) download on first start into `~/.cache/huggingface`. Use
`docker compose build` to build the image yourself instead. vLLM listens only inside the
container. Set `OPENJEV_UPSTREAM` to skip it and use a vLLM server you already run.

Measured on an RTX PRO 6000 using 38% of the GPU, with 3 questions per request and cache-busted
states:

| Concurrency | req/s | p50 | p95 |
|---:|---:|---:|---:|
| 1 | 10.7 | 94 ms | 94 ms |
| 16 | 43.3 | 367 ms | 369 ms |
| 32 | 51.7 | 545 ms | 618 ms |
| 64 | 57.4 | 760 ms | 1109 ms |

One request at a time, on the same GPU, with `samples: 1`:

| Request | p50 | p95 |
|---|---:|---:|
| 1 question | 27 ms | 28 ms |
| 3 questions | 31 ms | 32 ms |

Without Docker:

```bash
git clone https://github.com/vllm-project/vllm && cd vllm
VLLM_COMMIT=1b3b88ec2b7457aa030db4d0e7d8aaf04f6d0fb8   # the commit the image pins
git checkout $VLLM_COMMIT
# a choice of more than 128 options needs a larger cap, as in the image
sed -i 's/^MAX_LOGPROB_TOKEN_IDS = 128$/MAX_LOGPROB_TOKEN_IDS = 512/' vllm/sampling_params.py
VLLM_USE_PRECOMPILED=1 VLLM_PRECOMPILED_WHEEL_COMMIT=$VLLM_COMMIT pip install -e .
vllm serve nvidia/diffusiongemma-26B-A4B-it-NVFP4 --served-model-name dgemma \
  --diffusion-config '{"canvas_length": 64}' --max-logprobs 32 --enable-prefix-caching \
  --async-scheduling --attention-backend TRITON_ATTN \
  --limit-mm-per-prompt '{"image": 8, "video": 0}' \
  --enable-auto-tool-choice --tool-call-parser gemma4 --reasoning-parser gemma4 \
  --override-generation-config '{"max_new_tokens": null}'
pip install -e path/to/openjev && python -m openjev
```

This build reads and generates exactly as the image does, but it prefills images causally: the
image's other change, bidirectional attention inside each image, is a patch to a vLLM source
file (see [Caveats](#caveats)). Apply `docker/patches/vision_prefix_lm.py` to the checkout above
to match the image.

### Apple silicon

A Mac needs no vLLM and no Docker. `OPENJEV_BACKEND=mlx` runs DiffusionGemma inside the OpenJev
process through [MLX](https://github.com/ml-explore/mlx) and
[mlx-vlm](https://github.com/Blaizzy/mlx-vlm). The 4-bit weights need about 16 GB of memory.

```bash
pip install -e '.[mlx]'
OPENJEV_BACKEND=mlx python -m openjev     # 127.0.0.1:8080
```

`/v1/systemone` answers reads with the same prompts, canvases and seeds as the vLLM backend,
including `images`, `samples`, `sequential`, `steps` and the automatic re-reads. Each denoise
step past the first reuses the one prefill of the prompt and writes back only the answer slots,
so the template cannot drift: more steps cost GPU time, not prompt tokens. On vLLM the same
holds because a multi-step read pins every canvas position but the answer slots. `think` writes the
thought with mlx-vlm's own denoise loop, then reads after it, and bills exactly as vLLM does.

An image read builds its prompt with the mlx-vlm processor, which expands each image into its
soft tokens. `usage.input_tokens` and `OPENJEV_MLX_MAX_PROMPT` both count that expanded prompt.
The server keeps the prefill of a recent prompt, so the re-reads and the `samples` of one
request share a single vision pass.

Reads run one at a time, so this backend suits local use rather than serving. A 3-question
request takes about 0.2–0.4 s on an M3 Ultra and about 0.39 s on an M4 Max, both with the 4-bit
weights. 16 concurrent requests finish at about 4 req/s.

```bash
OPENJEV_MLX_TEST_MODEL=path/to/weights pytest tests/test_mlx_model.py   # tests against the real model
```

### Small encoder models

OpenJev also serves two small System One models that other people built and trained. The
credit for them is theirs: OpenJev only puts them behind the same API. Each is a bidirectional
encoder with a classification head, not a diffusion model. It reads each question in one
forward pass, so an answer is still a distribution over your options.

| Model id | Model | Size | State limit | Choices |
|---|---|---|---|---|
| `laya-1.0` | **[Laya](https://github.com/NandhaKishorM/laya)** by Nandakishor M / [Convai Innovations](https://huggingface.co/convaiinnovations). The [laya-typed-decisions](https://huggingface.co/convaiinnovations/laya-typed-decisions) checkpoint: ModernBERT-large, fine-tuned on the typed-decisions workflows. | 421M | 1,024 tokens, options included | up to 255. The options share 256 tokens, so with many options each is cut to a few tokens. Keep to about 20, or split the question. |
| `verdict-1.4` | **[Verdict](https://github.com/Heman10x-NGU/Verdict-open-jev)** by [Heman10x](https://huggingface.co/heman10x). The [rlcd-modernbert-151m](https://huggingface.co/heman10x/rlcd-modernbert-151m) checkpoint with Verdict's v1.4 inference engine: ModernBERT-base with a GLiClass head, calibrated per option count. | 151M | 512 tokens, options included | up to 24 |

Each model runs in its own container on the same API server: `OPENJEV_BACKEND=laya` or
`OPENJEV_BACKEND=verdict`. `docker compose up -d` starts both beside the vLLM container on the
same GPU. The `openjev` container passes a request for either model through to its container
(`OPENJEV_MODEL_ROUTES`), so `:8080` serves all three models. On a GPU both keep their weights in bf16.
Together they need about 3.7 GB of GPU memory, so set `OPENJEV_GPU_UTIL` to leave that much
free.

Measured on an RTX PRO 6000 with 16 questions per request, GPU memory as `nvidia-smi` reports it
(CUDA context included). A short state is about 50 tokens; a full one fills the model's limit.

| Model | GPU memory | 16 questions, short state | 16 questions, full state |
|---|---:|---:|---:|
| `laya-1.0` | 2.5 GB | 10 ms | 109 ms (16 × 1,024 tokens) |
| `verdict-1.4` | 1.2 GB | 7 ms | 21 ms (16 × 512 tokens) |

FlashAttention 2 did not help: the same memory, the same speed within 6%, and slower for short
states. The questions of one request share one state, so their sequences have almost the same
length and there is little padding to skip. PyTorch's own attention kernel already avoids the
full attention matrix. `/v1/models` lists the routed
models even when their containers are not running; a request for one then gets a 503.

To run one alone, on a GPU or on the CPU:

```bash
docker build -f docker/Dockerfile --target encoder --build-arg BACKEND=laya -t openjev-laya .
docker run -d --gpus all -p 127.0.0.1:8081:8080 \
  -v ~/.cache/huggingface:/root/.cache/huggingface openjev-laya
# or without Docker:
pip install -e '.[laya]' && OPENJEV_BACKEND=laya python -m openjev
```

All three images come from `docker/Dockerfile` and share its CUDA, Python and PyTorch layers
(about 8.7 GB on disk), so a server that runs all three pulls and stores those once. Together
they take about 21 GB on disk.

A server that runs one of these models alone also accepts `jev-latest` and `jev-preview` for it.

Differences from the DiffusionGemma model:

- Text only. `images`, `steps` above 1, `samples` above 1, `think` and `sequential` get a 400.
- A state longer than the limit is cut at the end without an error. `usage.input_tokens` counts
  what the model read. Each question is its own sequence, so each question bills the state again.
- Verdict adds an "insufficient evidence" option to every question. OpenJev drops its
  probability and scales the others to sum to 1, as Jev's answer shapes require. Verdict
  also ignores a noul's `criteria`.
- Laya rounds each probability to 4 decimal places.
- On a GPU the weights are in bf16, not fp32. On 36 test answers per model this changed no
  answer's top option, and moved probabilities by at most 0.021 (Laya) and 0.007 (Verdict).
- The Verdict prompt format and temperatures are copied from Verdict's inference engine
  (v1.4). For the same input, the probabilities match that engine's own output.

For benchmarks, training details, fine-tuning and known limits, read the authors' own
repositories: [Laya](https://github.com/NandhaKishorM/laya) and
[Verdict](https://github.com/Heman10x-NGU/Verdict-open-jev). Laya's
[`laya`](https://pypi.org/project/laya/) package runs inside the `laya` container. Verdict builds
on [ModernBERT](https://huggingface.co/answerdotai/ModernBERT-base) (Answer.AI, LightOn) and
[GLiClass](https://github.com/Knowledgator/GLiClass) (Knowledgator).

### Settings

The server reads its settings from the environment.

| Variable | Default | Meaning |
|---|---|---|
| `OPENJEV_BACKEND` | `vllm` | `mlx` to run the model in-process on Apple silicon. `laya` or `verdict` for a [small encoder model](#small-encoder-models) |
| `OPENJEV_MODEL_ROUTES` | unset | `name=url,...`: other OpenJev servers. A request for one of those model names goes to that server unchanged. |
| `OPENJEV_LAYA_MODEL` | `convaiinnovations/laya-typed-decisions` | Laya weights: a local directory or a Hugging Face id |
| `OPENJEV_VERDICT_MODEL` | `heman10x/rlcd-modernbert-151m` | Verdict weights: a local directory or a Hugging Face id |
| `OPENJEV_DEVICE` | unset | `laya`/`verdict`: `cuda` or `cpu`. Unset uses CUDA when there is a GPU |
| `OPENJEV_ENCODER_BATCH` | `16` | `laya`/`verdict`: most questions in one forward pass. A larger request uses more passes. |
| `OPENJEV_UPSTREAM` | unset | external vLLM server URL. When set, the container does not start its own |
| `OPENJEV_MODEL` | `nvidia/diffusiongemma-26B-A4B-it-NVFP4` | weights the built-in vLLM serves |
| `OPENJEV_MLX_MODEL` | `mlx-community/diffusiongemma-26B-A4B-it-4bit` | MLX weights: a local directory or a Hugging Face id. Also supplies the tokenizer. `8bit` and `bf16` builds exist too. |
| `OPENJEV_MLX_MAX_PROMPT` | `32768` | longest request, in tokens, before a 400 |
| `OPENJEV_GPU_UTIL` | `0.9` | vLLM `--gpu-memory-utilization` |
| `OPENJEV_MAX_NUM_SEQS` | `64` | vLLM `--max-num-seqs` |
| `OPENJEV_MAX_MODEL_LEN` | `65536` | vLLM `--max-model-len` |
| `OPENJEV_VLLM_ARGS` | unset | extra `vllm serve` flags |
| `OPENJEV_CANVAS` | `64` | canvas length. Also sets the built-in vLLM's `--diffusion-config` |
| `OPENJEV_MAX_INFLIGHT` | `64` | reads in flight to vLLM |
| `OPENJEV_MAX_QUEUE` | `512` | waiting decisions before the server returns 529 |
| `OPENJEV_API_KEY` | unset | require `Authorization: Bearer <key>` |
| `OPENJEV_ORIGIN_SECRET` | unset | require an `X-Origin-Secret` header (for use behind a proxy) |
| `OPENJEV_MAX_IMAGES` | `8` | images per request. Also sets the built-in vLLM's `--limit-mm-per-prompt` |
| `OPENJEV_MAX_IMAGE_BYTES` | `5242880` | size limit per image, after base64 decoding |
| `OPENJEV_GEN_MAX_INFLIGHT` | `8` | text generations running at once |
| `OPENJEV_GEN_MAX_QUEUE` | `32` | waiting generations before the server returns 529 |
| `OPENJEV_GEN_MAX_TOKENS` | `8192` | cap on `max_tokens` for text generation |
| `OPENJEV_WARMUP` | `1` | `0` skips the warmup requests sent before the API opens. They save the first users several seconds of compiling. |

## Caveats

- OpenJev pins upstream vLLM rather than a fork, and makes two changes to that commit when it
  builds the image. Both are checked at build time, so a moved anchor fails the build instead of
  shipping silently:
  - Upstream allows 128 exact label ids per request; a 255-option choice needs more, so the
    build raises the cap to 512.
  - The checkpoint's config asks for bidirectional attention inside each image
    (`use_bidirectional_attention: "vision"`). vLLM applies that for Gemma4 but not for
    DiffusionGemma, whose `prepare_attn` never passes the image ranges, so image tokens would be
    prefilled causally. `docker/patches/vision_prefix_lm.py` passes them, reusing vLLM's own
    `compute_mm_prefix_ranges`. An upstream fix would make it unnecessary.
- Answer quality is the quality of DiffusionGemma 26B-A4B used in this mode. Evaluate it on your
  own tasks before you rely on it.

## Development

```bash
pip install -e '.[test]' && pytest
OPENJEV_LIVE_URL=http://127.0.0.1:8080 pytest tests/test_live.py   # end to end against a running server
```

Run the live checks after building an image and before a cutover. They cover every read
option, images, chat, and whichever encoder models the server lists.

## License

Apache-2.0. The DiffusionGemma weights are Apache-2.0 (NVIDIA / Google). Laya
(Nandakishor M / Convai Innovations) and Verdict (Heman10x) are Apache-2.0, weights and code;
`openjev/encoders.py` adapts Verdict's prompt format and calibration from its repository.
