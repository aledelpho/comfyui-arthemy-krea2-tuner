# Example Workflows

## `Arthemy_Krea2_Tuner_Workflow.json`

The full bench: every node in the suite, wired and laid out, with a README note next to each
one. Drag the file onto the ComfyUI canvas to open it.

How it is arranged:

* **Left** — loaders, prompt, latent size, and the Reset Patcher.
* **Right** — image generation. Queue it to see what a change actually did.
* **Middle** — the working area. Copy any node up from the banks below and chain it here.
* **Bottom rows** — the banks, grouped: Model Tuners · CLIP Tuners · LoRA · Rotators ·
  LoRA-to-5D · Presets · Savers & Baker.

Most nodes arrive bypassed (Ctrl+B) so the graph runs out of the box. Un-bypass the one you
want to try.

### Suggested first passes

1. **Read the baseline.** Fixed seed and prompt, nothing enabled, generate once. Add a
   **Model Visualizer** to see that nothing is patched yet.
2. **One scalar.** Enable the **Model Tuner**, move a single block group, generate again.
   Its `info` output ends in `Patches: N` — that is your confirmation the change landed.
3. **One rotation.** Enable the **Model Axis Rotator** at 10-15°, `depth_reach = Default`,
   one dial at a time. Same energy, different direction: it changes character rather than
   strength.
4. **Save it.** Wire the **Preset Saver** at the end of the chain. Everything you enabled
   comes back from a few KB of JSON.

> [!TIP]
> Any PNG generated from a ComfyUI graph carries that graph in its metadata — drag a
> generated image back onto the canvas to reopen the exact setup that produced it.
