# ComfyUI render run example

Copy `run.yaml` into a run created with `kura render new --slug example`, then set the workflow and promptset paths. Put an API-format workflow exported from ComfyUI in `workflows/`; UI workflow JSON is not accepted by `/prompt`.

```bash
kura render compile <run-id>
kura render launch <run-id> --dry-run
kura render launch <run-id>
```

Configure `workflow_patches` with existing API workflow node IDs and input paths before launch.
