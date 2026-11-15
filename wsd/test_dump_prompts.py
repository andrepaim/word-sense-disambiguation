import json
import os
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent


def test_dump_prompts_writes_model_prompts(tmp_path):
    """Run the script as a user would, so import-time breakage (a renamed module attribute) fails here too."""
    (tmp_path / "corpus.txt").write_text("She held a test tube at the bank.\nThe bass player adjusted the bass.\n")
    out = tmp_path / "prompts.jsonl"

    subprocess.run(
        [sys.executable, str(REPO / "scripts" / "dump_prompts.py"), str(tmp_path / "*.txt"), str(out)],
        check=True, cwd=REPO, env={**os.environ, "PYTHONPATH": str(REPO)}, timeout=600,
    )

    prompts = [json.loads(line)["prompt"] for line in out.read_text().splitlines()]
    assert len(prompts) >= 6  # content words of both sentences
    for prompt in prompts:
        assert prompt.count("*") == 2  # exactly one marked word or expression
        assert prompt.endswith("[unused0] [MASK]")
        assert "none of the above" in prompt
