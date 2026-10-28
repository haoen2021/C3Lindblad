from pathlib import Path


def test_examples_do_not_build_disk_or_streaming_arnoldi():
    examples = Path(__file__).resolve().parents[1] / "examples"
    retired = (
        "checkpointed_arnoldi_resolvent",
        "JaxPanelArnoldiOrthogonalizer",
        "mmap_mode=",
        "_run_source_cpu",
    )
    for path in examples.rglob("*.py"):
        text = path.read_text()
        for name in retired:
            assert name not in text, f"{path.relative_to(examples)} still uses {name}"
