from pathlib import Path


def test_gan_nsight_docs_keep_profiles_out_of_formal_speedup() -> None:
    root = Path(__file__).resolve().parents[1]
    text = (root / "docs/design/gan_nsight_notes.md").read_text()
    assert "diagnostic only" in text
    assert "cudaProfilerStart" in text
    assert "cudaProfilerStop" in text
    assert "controller ranks `[0,8]`" in text
