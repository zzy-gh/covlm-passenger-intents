# Passenger intents including urgency info for CoVLM / InterDrive

InterDrive r1-r46 with passenger intents whose destination follows from the
words, keeping each vehicle's original urgency tier. 175 vehicles, one sentence
each.

## Install

```bash
bash apply.sh /path/to/covlm-agent-main/CoLMDriver-main
```

Then add one line to the `cov2v` section of
`simulation/leaderboard/team_code/agent_config/covlm.yaml`, next to `jpeg_quality`:

```yaml
    driver_intents_path: simulation/leaderboard/team_code/agent_config/driver_intents_nav.yaml
```

Also check the model, because upstream's default no longer resolves for newer
API keys — `gemini-2.5-flash` answers `404 NOT_FOUND ... no longer available to
new users`. Any `gemini-3.x` model works:

```yaml
    model: gemini-3.5-flash-lite
```

## Run

From the CoLMDriver-main root, in the CARLA environment, CARLA already running.
All 46 routes:

```bash
# args: cuda  port  method  latency  scenario_type  [route ids]
ROUTE_SUFFIX=_nav bash scripts/eval/eval_mode.sh 0 2000 covlm ideal Interdrive_no_npc

# baseline, no passenger utterances
COV2V_DISABLE_DRIVER_INTENTS=1 ROUTE_SUFFIX=_nav \
    bash scripts/eval/eval_mode.sh 0 2000 covlm ideal Interdrive_no_npc
```

Results go to `results/results_driving_<tag>/`.

Notes:
- `ROUTE_SUFFIX` is what this patch adds to `eval_mode.sh`; without it the
  script behaves exactly as upstream and runs the original routes.
- To go back to the original utterances, point `driver_intents_path` at
  `driver_intents.yaml`, or delete the line to run with none.

## What this installs

| path | |
|---|---|
| `add/.../agent_config/driver_intents_nav.yaml` | new — the 175 sentences, keyed by route directory and vehicle |
| `add/.../data/Interdrive/r*_nav/` | new — 46 route directories matching the sentences |
| `replace/.../cov2v_bridge.py` | **required** — loads `driver_intents_path` per scenario, keyed by `ROUTES_DIR` |
| `replace/cov2v/prompting.py` | **required** — drops `urgency level` / `priority bonus` from the prompt when they are 0 |
| `replace/cov2v/gemini_negotiator.py` | **required on a gemini-3.x model** — upstream hardcodes `thinking_budget=0`, which 3.x rejects with `400 INVALID_ARGUMENT` on every call; this picks `thinking_level="low"` for 3.x and leaves 2.x untouched |
| `replace/scripts/eval/eval_mode.sh` | adds `ROUTE_SUFFIX`, one line; without it the script is unchanged |
| `replace/.../covlm_agent.py` | recommended — uses the loaded intents, logs `priority_score` |

`apply.sh` backs up each replaced file as `<name>.orig_backup`. `covlm.yaml` is
not overwritten (your API keys live there), and `diffs/` has the per-file diffs
against the untouched repo — use those instead of `replace/` if your checkout
has its own changes to any of these files. Authentication is unchanged from
upstream: set `gemini_api_key` as usual.

Also upstream's own, and unrelated to these intents: `python_bin` in
`covlm.yaml` points at the author's home directory, so set it to your
`cov2v-gemini` environment, and `external_paths/carla_root` has to be a symlink
to a CARLA 0.9.10 install.
