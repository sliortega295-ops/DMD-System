"""CPU-only execution of the H100 launcher's real placement policy."""
from pathlib import Path
import subprocess
import unittest


ROOT = Path(__file__).resolve().parents[1]
LAUNCHER = ROOT / "scripts/distillschedule/launch_wan13b_hardware_two_cycle_node.sh"


class H100ResidentBaselineLauncherTest(unittest.TestCase):
    def policy(self, placement):
        source = LAUNCHER.read_text()
        admission = source[source.index('case "$PLACEMENT"'):]
        admission = admission[:admission.index("esac") + 4]
        policy = source[source.index('if [[ "$PLACEMENT"'):]
        policy = policy[:policy.index("\nfi") + 3]
        script = ('set -euo pipefail\nPLACEMENT="$1"\n' + admission + "\n" + policy
                  + '\nprintf "%s %s\\n" "${NCCL_ALGO:-AUTO}" '
                    '"$DISTILLSCHEDULE_WAN13B_GENERATOR_NOGRAD_LORA_FUSED_EPILOGUE"\n')
        return subprocess.run(["bash", "-c", script, "policy", placement],
                              text=True, capture_output=True)

    def test_plain_resident_is_unoptimized_reference(self):
        result = self.policy("resident")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout.strip(), "AUTO 0")

    def test_fsdp_reference_is_unchanged(self):
        result = self.policy("world16-fsdp")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout.strip(), "AUTO 0")

    def test_winner_keeps_registered_optimizations(self):
        result = self.policy("resident-async128-tree-chunk2048")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout.strip(), "Tree 1")

    def test_unknown_placement_is_rejected(self):
        self.assertEqual(self.policy("resident-unknown").returncode, 2)


if __name__ == "__main__":
    unittest.main()
