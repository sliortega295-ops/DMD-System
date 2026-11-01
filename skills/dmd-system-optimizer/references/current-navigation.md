# Current recipes first, historical experience second

    python3 "$SKILL/scripts/dmd_knowledge.py" navigate --query "SD1.5 DMD2 resident tree fused Adam" --repo /project/DMD-System
    python3 "$SKILL/scripts/dmd_knowledge.py" navigate --query "Official Self-Forcing FSDP2 R6"
    python3 "$SKILL/scripts/dmd_knowledge.py" recall --brief --query "FLUX Qwen Image frozen base sharing"

`recall` includes `current_implementations` separately from historical cases. An
explicit DMD checkout is read fresh. Standalone installations use a generated,
source-pinned snapshot marked `BUNDLED_SNAPSHOT_NOT_CURRENT_TREE`. Missing/broken
explicit checkouts are reported, not silently replaced by a success from old data.
No package import, model loading, network request or test execution occurs.

Each family links its actual reference/optimized/other recipes, factory/profile
implementation paths, applicable registry metadata and related test paths.
`other` includes manual/native presets and templates: it does not mean slower or
unoptimized. `has_optimized_recipe=false` means no explicitly named/profile-bound
optimized recipe was found, not that the framework lacks optimization mechanisms.
Related generic capabilities require checking the chosen recipe's actual guards.
Tests are navigation only, never a new PASS. A stored profile's numerical REVIEW
is preserved, and no historical timing becomes current performance.

## Keep one source of truth

The generated index reads existing `training/catalog.py`, `official_profiles.py`,
`strategy_catalog.py`, `workloads/catalog.json` and `recipes/*/training*.json`.
After changing those sources, regenerate (do not hand-edit the bundled index):

    python3 "$SKILL/scripts/dmd_current.py" --repo /project/DMD-System \
      --revision "$(git -C /project/DMD-System rev-parse HEAD)" \
      --output "$SKILL/assets/current-navigation.json"

Use a reviewed source version. Recorded hashes identify the input bytes, including
uncommitted changes; a Git revision alone is not proof of a clean tree. The CPU test
compares generated metadata with the real registry/recipe bytes. New models need
actual registered recipes, not another hardcoded list in the Skill.

## Expand experience only when needed

    python3 "$SKILL/scripts/dmd_knowledge.py" recall --brief --query "缓存失效" --case-limit 3
    python3 "$SKILL/scripts/dmd_knowledge.py" recall --brief --query "缓存失效" --case-offset 3
    python3 "$SKILL/scripts/dmd_knowledge.py" recall --brief --query "缓存失效" --all-cases

Offsets apply independently to each outcome within each selected mechanism.
Keep the same query, context and authorized stores when paging. `case --id` and
unabridged `recall` retain complete observations, provenance and correction history.
The default index includes omitted counts and correction warnings, not hundreds
of old paragraphs. Use Codex's available native memory/history for personal
continuity and project logs for actual experimental facts; no private index is
required to use these public tools.
