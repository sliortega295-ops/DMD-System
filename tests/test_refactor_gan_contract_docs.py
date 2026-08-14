from pathlib import Path


def test_gan_contract_documents_use_post750_ema_semantics() -> None:
    root = Path(__file__).resolve().parents[1]
    contract = (root / "docs/contracts/routea_gan_full5_world16.md").read_text()
    runbook = (root / "docs/runbooks/baai_routea_gan_world16.md").read_text()
    assert "EMA optimization_step 800" in contract
    assert "G201 / C1005 / EMA optimization_step 801" in contract
    assert "no EMA" in contract
    assert "optimization_step=800" in runbook
    assert "one C update → no EMA" in runbook
