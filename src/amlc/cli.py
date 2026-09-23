"""`amlc <command> ...` shortcut for the module entry points."""

import importlib
import sys

COMMANDS = {
    "baseline": "amlc.baseline",
    "download": "amlc.data.downloader",
    "embed": "amlc.features.embed",
    "vlm": "amlc.inference.vlm",
    "merge": "amlc.inference.shard",
    "attach-images": "amlc.data.io",
    "validate": "amlc.submission.validator",
    "finetune": "amlc.training.finetune_text",
}


def main() -> None:
    if len(sys.argv) < 2 or sys.argv[1] not in COMMANDS:
        print("usage: amlc {" + ",".join(COMMANDS) + "} ...  (add -h for command help)")
        sys.exit(2)
    importlib.import_module(COMMANDS[sys.argv[1]]).main(sys.argv[2:])
