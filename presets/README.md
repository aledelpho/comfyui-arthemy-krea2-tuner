# presets/

Where the **🟪🟨💾 Preset Saver** writes and the **🟪🟨📂 Preset Loader** reads. Drop a `.json`
preset here and it appears in the Loader's dropdown after a ComfyUI restart.

## Shipped

* **`Arthemy_Comics_Preset.json`** — the Arthemy Comics tuning: gains across the model and the
  text encoder, four style-compass rotations, and six scoped 5D injections embedded in the
  file. Self-contained: it needs no source LoRA to reproduce.

## Your own presets

Everything else in this folder is git-ignored — presets are personal, and one that embeds its
5D modifiers can run to tens of megabytes. To ship another one with the repository, add an
allow-line for it in the root `.gitignore`; the line has to name a file that actually exists,
or `git add .` skips it without saying so.
