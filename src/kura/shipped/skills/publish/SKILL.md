---
name: publish
description: Publish a LoRA trained with Kura to Hugging Face or ModelScope. Use to choose the checkpoint to publish, write the model card, add sample images, check the license, and upload.
---

# Publish

Upload a trained LoRA with its model card and sample images to Hugging Face or ModelScope,
under the user's account.

## Flow

```text
ask what to publish, where, and how
   ↓
build the upload folder (the LoRA, the model card, sample images)
   ↓
kura check secrets <folder>     no secrets in it
kura check artifacts <folder>   which weight files it holds
   ↓ show it to the user ─→ yes
upload (with the user's own login)
```

## Who decides

| | Decides |
|---|---|
| Kura | whether the folder holds secrets, and which weight files it holds |
| You | the folder and the model card, the checks, the upload |
| The user | which checkpoint, where, public or private, the license, which sample images |

## Steps

### 1. Ask
- Which run and which checkpoint (from the outputs `kura run status <run-id>` lists).
- Where (Hugging Face or ModelScope), the repository name, public or private.
- The license. The base model's license (non-commercial, say) carries over to the LoRA: check
  the card or the base model's page and tell the user.

### 2. Build the upload folder
- Use `artifacts/publish/<repository name>/`, holding:
  - the LoRA file (a copy; never move the original)
  - `README.md` (the model card)
  - sample images (from Kura render runs; the user picks them)
- The model card says: the base model, how to use it (trigger word, LoRA strength), the training
  settings (backend, rank and alpha, steps), what the dataset was, the license.
- Do not include the dataset unless the user asks for it explicitly.

### 3. Check it and get a yes
- `kura check secrets artifacts/publish/<repository name>`: nothing may be found; fix anything
  it finds.
- `kura check artifacts artifacts/publish/<repository name>`: it lists the weight files. Only
  the LoRA you mean to upload may be listed (it always lists that one).
- Show it by the `writing` skill: where it goes, public or private, the files, the model card.
- Wait for the user's yes.

### 4. Upload
- Hugging Face:
  ```sh
  hf repo create <user>/<repository name> --private   # or --public
  hf upload <user>/<repository name> artifacts/publish/<repository name>
  ```
- ModelScope: the same with ModelScope's official tool.
- If the user is not logged in, ask them to log in from their own terminal (`hf auth login` for
  Hugging Face). Never ask for a token or write one into a file.
- Give the repository's URL when it is up.
