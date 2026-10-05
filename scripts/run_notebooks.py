"""Execute examples with this Python, without modifying user kernel settings."""
import argparse
import json
from pathlib import Path
import sys
import tempfile

import nbformat
from nbclient import NotebookClient
from jupyter_client import KernelManager
from jupyter_client.kernelspec import KernelSpecManager


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--with-hm3d", action="store_true", help="include the local dataset examples")
    parser.add_argument("--notebook", help="select an example by filename or glob, e.g. '04_*.ipynb'")
    parser.add_argument("--output-dir", type=Path, help="keep executed notebooks in this directory")
    args = parser.parse_args()
    examples = Path(__file__).resolve().parents[1] / "examples"
    paths = sorted(examples.glob(args.notebook or "*.ipynb"))
    paths = [path for path in paths if path.is_file() and path.suffix == ".ipynb"]
    if not args.with_hm3d:
        paths = [path for path in paths if "hm3d" not in path.stem.lower()]
    if not paths:
        parser.error(f"No matching notebooks in {examples}; HM3D examples require --with-hm3d")
    with tempfile.TemporaryDirectory(prefix="graphmap-notebooks-") as directory:
        temporary = Path(directory)
        kernel_dir = temporary / "kernels" / "graphmap-test"
        kernel_dir.mkdir(parents=True)
        (kernel_dir / "kernel.json").write_text(json.dumps({
            "argv": [sys.executable, "-m", "ipykernel_launcher", "-f", "{connection_file}"],
            "display_name": "graphmap test interpreter", "language": "python",
        }))
        output = args.output_dir or temporary / "executed"
        output.mkdir(parents=True, exist_ok=True)
        for path in paths:
            notebook = nbformat.read(path, as_version=4)
            assert notebook.cells[0].cell_type == "code"
            assert notebook.cells[0].source == "%load_ext autoreload\n%autoreload 3 --print --log"
            manager = KernelManager(
                kernel_name="graphmap-test",
                kernel_spec_manager=KernelSpecManager(kernel_dirs=[str(kernel_dir.parent)]),
            )
            client = NotebookClient(notebook, km=manager, timeout=180,
                                    resources={"metadata": {"path": str(examples)}})
            print(f"Executing {path.name} with {sys.executable}", flush=True)
            try:
                client.execute()
            finally:
                # A supplied KernelManager is caller-owned in nbclient.
                if manager.has_kernel:
                    manager.shutdown_kernel(now=True)
                manager.cleanup_resources()
            nbformat.write(notebook, output / path.name)
            print(f"PASS {path.name}", flush=True)
        print(f"Executed {len(paths)} notebooks")
        if args.output_dir:
            print(f"Outputs: {output.resolve()}")


if __name__ == "__main__":
    main()
