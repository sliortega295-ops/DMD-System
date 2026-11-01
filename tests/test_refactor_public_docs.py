"""Check public entry points without importing model libraries."""
from __future__ import annotations

import json
from pathlib import Path
import re
from urllib.parse import unquote, urlsplit

import pytest

ROOT = Path(__file__).resolve().parents[1]
PAGES = (
    'README.md', 'CONTRIBUTING.md', 'docs/README.md', 'recipes/README.md',
    'docs/architecture/framework.md', 'docs/architecture/repository_organization.md',
    'skills/dmd-system-optimizer/SKILL.md',
)


def links(page):
    return re.findall(r'\]\(([^)\s]+)\)', (ROOT / page).read_text(encoding='utf-8'))


@pytest.mark.parametrize('page', PAGES)
def test_public_entry_links_resolve(page):
    for href in links(page):
        target = urlsplit(href)
        if target.scheme or target.netloc or not target.path:
            continue
        path = (ROOT / page).parent / unquote(target.path)
        assert path.resolve().is_relative_to(ROOT), (page, href)
        assert path.exists(), f'{page} links to missing {href}'


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
    'docs/architecture/framework.md', 'docs/architecture/repository_organization.md',
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
