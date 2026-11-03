"""Check public entry points without importing model libraries."""
from __future__ import annotations

import json
from pathlib import Path
import re
from urllib.parse import unquote, urlsplit

import pytest

ROOT = Path(__file__).resolve().parents[1]


def test_actual_documentation_file_budget():
    """Count actual maintained pages, including the versioned contracts."""
    pages = sorted((ROOT / 'docs').rglob('*.md'))
    assert 30 <= len(pages) <= 40, f'actual docs/**/*.md count is {len(pages)}'


PAGES = (
    'README.md', 'AGENTS.md', 'CONTRIBUTING.md', 'docs/README.md', 'recipes/README.md',
    'docs/architecture/repository_organization.md',
    'docs/runbooks/native_planner_workflow.md', 'docs/workloads/open_role_pools.md',
    'docs/workloads/hardware_native_planning.md', 'docs/contracts/helios_native.md',
    'docs/guides/native_owner_backends.md',
    'docs/guides/performance_validation.md',
    'third_party/helios_compat/README.md',
    'skills/dmd-system-optimizer/SKILL.md',
)
PAGES = tuple(dict.fromkeys((*PAGES, *(str(path.relative_to(ROOT))
    for directory in ('docs', 'recipes')
    for path in sorted((ROOT / directory).rglob('*.md'))))))


def links(page):
    return re.findall(r'\]\(([^)\s]+)\)', (ROOT / page).read_text(encoding='utf-8'))


def anchors(path):
    text = path.read_text(encoding='utf-8')
    result = set(re.findall(r'<a id="([^"]+)"></a>', text))
    # Shell/Python comments inside examples are not Markdown headings.
    prose = re.sub(r'^```[^\n]*\n.*?^```', '', text, flags=re.M | re.S)
    counts = {}
    for heading in re.findall(r'^#{1,6} +(.+)$', prose, re.M):
        slug = re.sub(r'[^\w\- ]', '', heading.lower()).replace(' ', '-')
        ordinal = counts.get(slug, 0)
        counts[slug] = ordinal + 1
        result.add(slug + (f'-{ordinal}' if ordinal else ''))
    return result


@pytest.mark.parametrize('page', PAGES)
def test_public_entry_links_resolve(page):
    for href in links(page):
        target = urlsplit(href)
        if target.scheme or target.netloc:
            continue
        path = ((ROOT / page).parent / unquote(target.path)) if target.path else ROOT / page
        assert path.resolve().is_relative_to(ROOT), (page, href)
        assert path.exists(), f'{page} links to missing {href}'
        if target.fragment and path.suffix == '.md':
            assert unquote(target.fragment) in anchors(path), (page, href)


def test_consolidated_layout_counts_actual_pages_not_only_guides():
    counts = {directory: len(list((ROOT / 'docs' / directory).rglob('*.md')))
              for directory in ('architecture', 'contracts', 'guides', 'runbooks', 'status', 'workloads')}
    assert counts == dict(architecture=1, contracts=6, guides=9, runbooks=2, status=0, workloads=11)


@pytest.mark.parametrize('page,sections,requirements', [
    ('native_output',
     'WAN_NATIVE_OUTPUT_V1 HELIOS_NATIVE_OUTPUT_V1 HELIOS_GAN_NATIVE_OUTPUT_V1 PRIMARY_EXPORT_STAGING_V1 W1_W2_NATIVE_STREAM_HORIZON_V1',
     ('(4/3) I4 - (1/3) ones(4,4)', 'first eight bytes big-endian',
      'primary_only=True', 'G102/C510/EMA510', '2..100')),
    ('helios_native',
     'helios_gan_native_history_v1 helios_gan_native_history_checkpoint_cpu_v1 helios_nogan_native_history_v1 helios_nogan_native_history_safe_v1 helios_nogan_native_checkpoint_inputs_cpu_v1 helios_nogan_safe_stream_v1 helios_original_bounded_requests_v1 original_helios_record_epochs_v1',
     ('7,593,984,000', '60,751,872,000', '649,920,512',
      'FSDP2_ALIGNED_MATERIALIZATION_WITH_ORIGINAL_REDUCE_AND_STEP', '80 * cycles')),
    ('h3_native',
     'h3_attention_tuple_capture_v2 minimax_h3_bounded_requests_v1 original_h3_ar_capacity_v2 original_h3_ar_continuation_v1 original_h3_ar_native_v1 original_h3_ar_native_v2 original_h3_ar_record_epochs_v1',
     ('37,727', 'num_frames % 17 == 5', 'host_limits',
      '[2,32,94]', 'exclude_frozen_parameters=True', '800 logical draws')),
    ('livewan_native',
     'original_livewan_reference_training_v1 original_livewan_teacher_cohorts_v1 original_livewan_teacher_epochs_v1',
     ('[16,21,46,80]', '1 <= N <= 64', 'Gk/C5k/EMA0', '80 * cycles')),
    ('state_retention',
     'frozen_teacher_construction_handoff_v1 wan_rollout_shared_scratch_v1 wan_tail6_retention_v1 wan_tail8_retention_v1',
     ('cpu-fsdp2-frozen-teacher-v1', 'shared-scratch.v1', 'Critic24–29', 'blocks22..29', '20.324208')),
])
def test_consolidated_contracts_retain_versioned_sections_and_unique_constraints(page, sections, requirements):
    path = ROOT / 'docs/contracts' / (page + '.md')
    content = path.read_text()
    assert set(sections.split()) <= anchors(path)
    for value in requirements:
        assert value in content, (page, value)


def test_public_documentation_keeps_results_not_experiment_process_reports():
    assert not list((ROOT / 'docs/status').glob('*.md'))
    assert not list((ROOT / 'docs/workloads/evidence').glob('*.md'))
    assert not list((ROOT / 'docs/experiments').rglob('*.md'))
    assert not (ROOT / 'docs/guides/release_notes.md').exists()
    assert not (ROOT / 'artifacts/experiments').exists()


def test_one_performance_page_contains_current_three_baseline_comparisons():
    page = (ROOT / 'docs/guides/performance_validation.md').read_text()
    for title in ('FSDP2 baseline', 'Resident / offload baseline', 'RealHF-like baseline'):
        assert f'### {title}' in page
    for value in ('116.254', '7.514', '3.475', '50.739', '30.959', '255.671', 'H100'):
        assert value in page
    assert '113.59' not in page
    readme = (ROOT / 'README.md').read_text()
    assert 'docs/guides/performance_validation.md' in readme
    assert not re.search(r'^\|.*\|\s*\d+\.\d+\s*(?:s|×)?\s*\|', readme, re.M)


def test_current_usage_pages_do_not_repeat_historical_review_status():
    pages = [ROOT / 'README.md', *(ROOT / 'docs').rglob('*.md'),
             *(ROOT / 'recipes').rglob('*.md')]
    for path in pages:
        assert not re.search(r'\bREVIEW\b', path.read_text()), str(path.relative_to(ROOT))


def test_readme_and_recipe_index_cover_discoverable_training_families():
    families = set()
    for path in (ROOT / 'recipes').glob('*/training*.json'):
        value = json.loads(path.read_text(encoding='utf-8'))
        if isinstance(value, dict) and value.get('schema_version') == 'distillschedule.training.v1':
            families.add(path.parent.name)
    assert families
    for page in ('README.md', 'recipes/README.md'):
        targets = {(ROOT / page).parent.joinpath(urlsplit(link).path).resolve()
                   for link in links(page) if not urlsplit(link).scheme}
        missing = [family for family in sorted(families)
                   if (ROOT / 'recipes' / family / 'README.md').resolve() not in targets]
        assert not missing, f'{page} omits training families: {missing}'


def test_package_metadata_points_to_maintained_project_resources():
    try:
        import tomllib
    except ModuleNotFoundError:
        import tomli as tomllib
    project = tomllib.loads((ROOT / 'pyproject.toml').read_text())['project']
    base = 'https://github.com/sliortega295-ops/DMD-System'
    assert project['urls']['Repository'] == base
    assert project['urls']['Issues'] == base + '/issues'
    assert project['urls']['Documentation'] == base + '/blob/main/docs/README.md'


def test_readme_cpu_example_uses_the_installed_cli():
    readme = (ROOT / 'README.md').read_text()
    blocks = re.findall(r'```sh\n(.*?)```', readme, re.S)
    example = next(block for block in blocks if 'distillschedule.examples.tiny_av' in block)
    for command in ('data prepare', 'train', 'export'):
        assert f'distillctl {command}' in example
    assert '--resume "$CP"' in example
    assert 'torchrun' not in example
    assert (ROOT / 'src/distillschedule/examples/tiny_av.py').is_file()


@pytest.mark.parametrize('page', (
    'docs/architecture/repository_organization.md',
))
def test_architecture_entry_links_have_installed_targets(page):
    try:
        import tomllib
    except ModuleNotFoundError:
        import tomli as tomllib
    declared = tomllib.loads((ROOT / 'pyproject.toml').read_text())['tool']['setuptools']['data-files']
    installed = {(Path(destination) / path.name).as_posix()
                 for destination, patterns in declared.items()
                 for pattern in patterns for path in ROOT.glob(pattern)}
    for href in links(page):
        target = urlsplit(href)
        if target.scheme or target.netloc or not target.path:
            continue
        relative = ((ROOT / page).parent / unquote(target.path)).resolve().relative_to(ROOT)
        assert (Path('share/distillschedule') / relative).as_posix() in installed, (page, href)
