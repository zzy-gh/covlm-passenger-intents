#!/bin/bash
# Install the navigation-intent dataset and its code changes into a CoVLM
# (CoLMDriver-main) checkout.
#
#   bash apply.sh /path/to/covlm-agent-main/CoLMDriver-main
#
# Files in replace/ overwrite existing ones; the originals are kept as
# <name>.orig_backup unless one is already there. Files in add/ are new and
# nothing upstream is touched by them. covlm.yaml is NOT written: it holds your
# own API keys and python_bin, so add the one line by hand (see README.md).

set -euo pipefail

HERE=$(cd "$(dirname "$0")" && pwd)
TARGET=${1:?usage: bash apply.sh /path/to/CoLMDriver-main}
TARGET=$(cd "$TARGET" && pwd)

if [ ! -d "$TARGET/simulation/leaderboard/team_code" ] || [ ! -d "$TARGET/cov2v" ]; then
    echo "error: $TARGET is not a CoLMDriver-main checkout" >&2
    exit 1
fi

echo "== replacing modified files"
while IFS= read -r rel; do
    src="$HERE/replace/$rel"
    dst="$TARGET/$rel"
    if [ ! -f "$dst" ]; then
        echo "  warn: $rel does not exist in the target, copying anyway"
    elif [ ! -f "$dst.orig_backup" ]; then
        cp "$dst" "$dst.orig_backup"
    fi
    mkdir -p "$(dirname "$dst")"
    cp "$src" "$dst"
    echo "  $rel"
done < <(cd "$HERE/replace" && find . -type f | sed 's|^\./||')

echo "== adding new files"
while IFS= read -r rel; do
    mkdir -p "$(dirname "$TARGET/$rel")"
    cp "$HERE/add/$rel" "$TARGET/$rel"
done < <(cd "$HERE/add" && find . -type f | sed 's|^\./||')
echo "  driver_intents_nav.yaml"
echo "  $(cd "$HERE/add/simulation/leaderboard/data/Interdrive" && ls -d *_nav | wc -l) route directories under data/Interdrive/"

cat <<'EOF'

== one manual step left
In simulation/leaderboard/team_code/agent_config/covlm.yaml, add this line to
the cov2v section (next to jpeg_quality) -- diffs/covlm.yaml.patch is the same
change:

  driver_intents_path: simulation/leaderboard/team_code/agent_config/driver_intents_nav.yaml

On a newly created API key, also set a reachable model, because upstream's
gemini-2.5-flash default answers 404 NOT_FOUND:

  model: gemini-3.5-flash-lite

Then run a scenario (see README.md):
  CUDA_VISIBLE_DEVICES=0 bash scripts/eval/eval_driving.sh \
      2 2000 r1_town05_ins_c_nav covlm_agent covlm covlm_nav Interdrive_no_npc
EOF
