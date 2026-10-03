# Repository maintenance

The root README is the entry point for the research data, analysis methods and reproduction commands.

The versioned materials are `code/`, `configs/`, `data/`, `docs/`, `manifests/`, `results/reference/`, the environment specifications, and the root documentation and license files. `.gitignore` keeps the local manuscript, working environments, caches and generated runs local. `.gitattributes` preserves file bytes across operating systems so that the SHA-256 manifests also verify after cloning.

The scientific workflow writes each complete reproduction under its selected output directory. `results/reference/` contains the fixed comparison outputs. `manifests/validated_run.json` records the verified repository snapshot.

After updating data or code, refresh the file manifests, reproduce into a fresh directory, and save the validated summary:

```powershell
uv run --no-sync python code/build_manifests.py
uv run --no-sync python code/run_all.py --output-dir results/reproduced_update
uv run --no-sync python code/verify.py --output-dir results/reproduced_update
Copy-Item results/reproduced_update/run_summary.json manifests/validated_run.json
```

Review the repository changes with `git status --short` and `git diff --check` before committing.
