# Kimi K2 API client

A small command-line client for Moonshot AI's Kimi models. The Moonshot API is
OpenAI-compatible, so this uses the standard `openai` Python SDK pointed at
Moonshot's base URL.

## Setup (once)

```powershell
cd kimi-api
python -m venv .venv
.\.venv\Scripts\Activate.ps1       # on macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
copy .env.example .env             # macOS/Linux: cp .env.example .env
```

Then open `.env` and paste your key after `MOONSHOT_API_KEY=`. The key comes
from the API Keys page of the Moonshot console. `.env` is git-ignored so it
never ends up in a commit.

If your key came from `platform.moonshot.cn` rather than `platform.moonshot.ai`,
uncomment `MOONSHOT_BASE_URL` in `.env` and set it to `https://api.moonshot.cn/v1`.

## Run

```powershell
python kimi.py --models                 # sanity check: key works, see model ids
python kimi.py "what is a dot product"  # one question, streamed answer
python kimi.py                          # interactive chat with memory
python kimi.py --model kimi-k2-thinking # pick a specific model
```

With no `--model`, the script picks the newest Kimi K2 it finds in your key's
model list. Thinking models show their reasoning dimmed before the answer.

## How it relates to `../learning`

`learning/model_playground.py` talks to the local Qwen model through
text-generation-webui's OpenAI-compatible endpoint. This client speaks the
exact same protocol to a hosted model. Same request shape, different URL and
an auth header. That is the whole difference between "local" and "cloud" LLM
serving from the client's point of view, which is the Phase 3 "hosted APIs vs
local" item on the learning path.
