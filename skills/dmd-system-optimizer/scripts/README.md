# Optimizer helpers: ownership and maintenance

The public commands stay in `dmd_knowledge.py` and `dmd_perf.py`. No model,
runner, server or experiment approval layer is added.

| Module | Owns |
| --- | --- |
| `dmd_knowledge.py` | CLI, public-pack validation, grouped recall and presentation |
| `dmd_memory.py` | Private records, scoped corrections, explicit session import |
| `dmd_conditions.py` | Normalize supplied conditions and compare them without guessing facts |
| `dmd_sources.py` | Pinned source references, source search and static inventory |
| `dmd_navigation.py` | Case-to-code symbols, integration and fallback navigation |
| `dmd_retrieval.py` | Lexical ranking and symptom routes |
| `dmd_perf.py` | Read-only arithmetic and shared file/input utilities |

Dependencies point from CLI composition to helpers, not back into the CLI.
Existing helper imports from `dmd_knowledge` remain available for local callers.
Public source identities have one declaration in `assets/experience-source-index.json`
under `source_files`, keyed by stable `source_id`. The experience pack,
implementation map and discovery rows retain their own locators, spans and
qualifications, not copies of repository/revision/path/hash/URLs. Use
`dmd_sources.load_public_data(path)` to read these assets: it expands references
and derives pinned GitHub URLs and span anchors; legacy inline assets still work.
`recall`, `case`, `locate`, `sources` and `render_knowledge.py` expose the same
expanded fields and links. Code-only declarations do not become discovery rows.
Keep `current-navigation.json` separate: these pinned historical identities do
not declare current-tree applicability or runtime validation.

Private `search --limit` limits records after ranking all matching mechanisms;
`recall --limit` intentionally limits mechanism packets, with omitted routes
reported. Neither command approves an implementation.

Session imports keep original provenance when the same message appears in a
copied log. `imported` counts accepted messages; `created` and `already_present`
distinguish new records from repeats. Damaged selected rows are counted as
ignored; user/tool/reasoning content is not imported. Run the stdlib suite:

    python -m unittest discover -s skills/dmd-system-optimizer/tests -v
