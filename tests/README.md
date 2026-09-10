# Test Suite

The tests in this directory ensure structural integrity, memory safety, ComfyUI contract compliance, and numerical correctness.

## Running Tests

### 1. Offline / Standalone (No GPU, No ComfyUI)
Tests that do not require PyTorch tensor computations use `stub_env.py` to fake `comfy` and `torch` modules:

```bash
python tests/test_node_contracts.py
python tests/test_reset_patcher.py
python tests/test_channel_roundtrip.py
python tests/test_prune.py
python tests/test_prune_scope.py
python tests/test_display_names.py
python tests/test_saver_e2e.py
```

### 2. With Real PyTorch (ComfyUI Python Interpreter)
Tests that verify real tensor operations, SVD / rotations, model baking, or FP8 companion scale conversions require a real `torch` runtime. Run them using ComfyUI’s Python executable:

```bash
<path-to-comfyui-python> tests/test_baker.py
<path-to-comfyui-python> tests/test_channel_scale.py
<path-to-comfyui-python> tests/test_channel_profile.py
<path-to-comfyui-python> tests/test_savers_patching.py
```

Or run all tests in sequence:
```powershell
Get-ChildItem -Path tests -Filter "test_*.py" | ForEach-Object {
    & "<path-to-comfyui-python>" $_.FullName
}
```

## Test Coverage Map

| Test | What It Pins | Requires PyTorch |
| :--- | :--- | :---: |
| `test_node_contracts.py` | Node registration, unique display names, `FUNCTION` reachable from `INPUT_TYPES`, return arity, default selections, and Preset Loader replay targets. | No |
| `test_reset_patcher.py` | Reset Patcher clears only its own patches/recipes and leaves `backup`, `hook_backup`, and `object_patches` intact. | No |
| `test_channel_roundtrip.py` | Channel Magnitude tunings survive preset save/load and are not misidentified as external LoRAs. | No |
| `test_prune.py` | 5D scope pruning and aliasing mechanism preventing payload collisions. | No |
| `test_saver_e2e.py` | Preset Saver writes scoped payloads and verifies alias resolution on reload. | No |
| `test_display_names.py` | Node display naming mappings and legacy aliases. | No |
| `test_baker.py` | Model Baker folds patches directly into module weights without clearing unfoided patches. | **Yes** |
| `test_channel_scale.py` | Lightweight channel adapters match exact dense delta math across axes. | **Yes** |
| `test_channel_profile.py` | Channel profile cache invalidation and matrix fingerprinting. | **Yes** |
| `test_savers_patching.py` | Savers correctly calculate patched weights and recognize FP8 companion scale spellings. | **Yes** |
