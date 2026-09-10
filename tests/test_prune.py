"""Round-trip tests for the 5D scope pruning added to the Preset Saver."""
import json, sys
import stub_env, importlib
m = importlib.import_module("Arthemy_Krea2_Tuner")

FAIL = []
def check(cond, label, extra=""):
    (print(f"  ok   {label}") if cond else (FAIL.append(label), print(f"  FAIL {label} {extra}")))

# ----------------------------------------------------------------- fake payload
def make_payload(order=64, storage="dct"):
    layers = {}
    def add(name, domain, d_out, d_in):
        layers[name] = {"domain": domain, "d_out": d_out, "d_in": d_in,
                        "singular_values": [1.0, .5, .3, .2, .1],
                        "u_dct": [[0.01] * order for _ in range(5)],
                        "v_dct": [[0.01] * order for _ in range(5)]}
    for b in range(28):                       # 28 UNet blocks x 4 tensors
        add(f"blocks.{b}.attn.qkv", "model", 9216, 3072)
        add(f"blocks.{b}.attn.proj", "model", 3072, 3072)
        add(f"blocks.{b}.mlp.fc1", "model", 12288, 3072)
        add(f"blocks.{b}.mlp.fc2", "model", 3072, 12288)
    for name in ("first.proj", "last.proj", "txtfusion.linear", "tmlp.fc"):
        add(name, "model", 3072, 3072)        # block-less sections
    for l in range(36):                       # CLIP layers
        add(f"text_model.encoder.layers.{l}.self_attn.q_proj", "clip", 1280, 1280)
        add(f"text_model.encoder.layers.{l}.mlp.fc1", "clip", 5120, 1280)
    return {"version": "5.0", "storage": storage, "source_lora": "Comics.safetensors",
            "rank": 5, "harmonic_order": order, "normalized": True, "layers": layers}

def runtime_selection(payload, domain, target, subs, sub_tensor="All Sub-Tensors"):
    """Exactly what synthesize_5d_patches_from_dct would keep for this recipe."""
    tuner = m.Arthemy5DCLIPTuner() if domain == "clip" else m.Arthemy5DTuner()
    selected = m.resolve_target_map_entry(tuner.TARGET_MAP, target,
                                          lambda: set(range(0, 60 if domain == "clip" else 28)))
    filt = tuner._layer_filter(selected, subs, whole_model=target.strip().lower().startswith("all"),
                               sub_tensor=sub_tensor)
    return {n for n, e in payload["layers"].items()
            if e.get("domain", "model") == domain and filt(n)}

def kb(p): return len(json.dumps(p)) / 1024.0

# ----------------------------------------------------------------- 1. single block
print("\n1. single model block scope")
full = make_payload()
scopes = [("model", "  ↳ Block_3B (11)", "All Components", "All Sub-Tensors")]
pruned, stats = m.prune_5d_payload_to_scopes(full, scopes)
check(pruned is not full, "payload was pruned")
check(set(pruned["layers"]) == {f"blocks.11.{s}" for s in
      ("attn.qkv", "attn.proj", "mlp.fc1", "mlp.fc2")}, "kept exactly block 11",
      sorted(pruned["layers"])[:6])
check(stats["kept"] == 4 and stats["total"] == len(full["layers"]), "stats", stats)
check(runtime_selection(full, *scopes[0]) == runtime_selection(pruned, *scopes[0]),
      "EQUIVALENCE: same layers synthesised from pruned as from full")
print(f"     {kb(full):.0f} KB -> {kb(pruned):.0f} KB  ({kb(full)/kb(pruned):.1f}x smaller)")

# ----------------------------------------------------------------- 2. group label
print("\n2. group label (Block_3 = 10-14) + union with a second scope")
scopes = [("model", "Block_3 (All 10-14)", "All Components", "All Sub-Tensors"),
          ("model", "  ↳ Block_6A (24)", "All Components", "All Sub-Tensors")]
pruned, stats = m.prune_5d_payload_to_scopes(full, scopes)
idx = {int(n.split(".")[1]) for n in pruned["layers"]}
check(idx == {10, 11, 12, 13, 14, 24}, "union of both scopes", idx)
check(all(runtime_selection(full, *s) == runtime_selection(pruned, *s) for s in scopes),
      "EQUIVALENCE for every scope")

# ----------------------------------------------------------------- 3. sub-components
print("\n3. sub-component scope")
scopes = [("model", "  ↳ Block_3B (11)", "ATTN (WQ/WK/WV/WO/Gate)", "All Sub-Tensors")]
pruned, _ = m.prune_5d_payload_to_scopes(full, scopes)
check(set(pruned["layers"]) == {"blocks.11.attn.qkv", "blocks.11.attn.proj"}, "ATTN only",
      sorted(pruned["layers"]))
check(runtime_selection(full, *scopes[0]) == runtime_selection(pruned, *scopes[0]), "EQUIVALENCE")

# ----------------------------------------------------------------- 4. clip / cross-domain
print("\n4. domains")
scopes = [("clip", "  ↳ Layer_1A (0)", "All Components", "All Sub-Tensors")]
pruned, _ = m.prune_5d_payload_to_scopes(full, scopes)
check(all(e["domain"] == "clip" for e in pruned["layers"].values()), "no model layers survive a clip-only preset")
check(runtime_selection(full, *scopes[0]) == runtime_selection(pruned, *scopes[0]), "EQUIVALENCE")
mixed = [("model", "  ↳ Block_3B (11)", "All Components", "All Sub-Tensors"), ("clip", "  ↳ Layer_1A (0)", "All Components", "All Sub-Tensors")]
pruned, _ = m.prune_5d_payload_to_scopes(full, mixed)
check(len({e["domain"] for e in pruned["layers"].values()}) == 2, "both domains kept when both are used")
check(all(runtime_selection(full, *s) == runtime_selection(pruned, *s) for s in mixed), "EQUIVALENCE")

# ----------------------------------------------------------------- 5. whole-model passthrough
print("\n5. whole-model / unknown labels keep their whole domain")
for lbl in ("All Blocks (0-27)", "Something The Map Never Had"):
    pruned, stats = m.prune_5d_payload_to_scopes(full, [("model", lbl, "All Components", "All Sub-Tensors")])
    model_full = {n for n, e in full["layers"].items() if e["domain"] == "model"}
    check(set(pruned["layers"]) == model_full, f"every model layer kept for '{lbl}'")
    check(not any(e["domain"] == "clip" for e in pruned["layers"].values()),
          f"unused CLIP half still dropped for '{lbl}'")
    check(runtime_selection(full, "model", lbl, "All Components")
          == runtime_selection(pruned, "model", lbl, "All Components"), f"EQUIVALENCE for '{lbl}'")
pruned, _ = m.prune_5d_payload_to_scopes(full, [("model", "All Blocks (0-27)", "All Components", "All Sub-Tensors")])
check(runtime_selection(full, "model", "All Blocks (0-27)", "All Components")
      >= {"first.proj", "txtfusion.linear"}, "block-less sections still reachable at All Blocks")

# ----------------------------------------------------------------- 6. no-match safety
print("\n6. a scope that matches nothing keeps the payload whole")
tiny = {"storage": "dct", "harmonic_order": 64,
        "layers": {"blocks.3.attn.qkv": {"domain": "model", "d_out": 8, "d_in": 8,
                                          "singular_values": [1.0], "u_dct": [[0.1]], "v_dct": [[0.1]]}}}
pruned, _ = m.prune_5d_payload_to_scopes(tiny, [("model", "  ↳ Block_6D (27)", "All Components", "All Sub-Tensors")])
check(pruned is tiny, "payload untouched rather than emptied")

# ----------------------------------------------------------------- 7. fidelity freeze
print("\n7. fidelity is frozen at the full payload's worst vector")
m.dct_fidelity = lambda order, n: (float(order) / float(n)) ** 0.5   # the engine's formula
full_f = m.payload_worst_fidelity(full)
pruned, _ = m.prune_5d_payload_to_scopes(full, [("model", "  ↳ Block_3B (11)", "ATTN (WQ/WK/WV/WO/Gate)", "All Sub-Tensors")])
check(abs(m.payload_worst_fidelity(pruned) - full_f) < 1e-6,
      "pruned payload still reports the full payload's fidelity",
      f"{m.payload_worst_fidelity(pruned)} vs {full_f}")
naive = m.dct_fidelity(64, max(max(e["d_out"], e["d_in"]) for e in pruned["layers"].values()))
check(naive > full_f, "and that figure is genuinely worse than the naive one", f"naive={naive:.3f} frozen={full_f:.3f}")

# ----------------------------------------------------------------- 8. alias round-trip
print("\n8. scoped alias")
tag = m._scope_tag([("model", "  ↳ Block_3B (11)", "All Components", "All Sub-Tensors")])
alias = f"Comics.json{m.SCOPE_ALIAS_SEP}{tag}"
check(m.scope_alias_base(alias) == "Comics.json", "alias resolves back to its base name", alias)
check(m.scope_alias_base("Comics.json") == "Comics.json", "a plain name is its own base")
check(m._scope_tag([("model", "A", "B", "C"), ("clip", "C", "D", "E")]) ==
      m._scope_tag([("clip", "C", "D", "E"), ("model", "A", "B", "C")]), "tag is order-independent")
long_tag = m._scope_tag([("model", f"  ↳ Block_{i}A ({i})", "All Components", "All Sub-Tensors") for i in range(9)])
check(len(long_tag) <= 48 and long_tag.isalnum(), "long scopes collapse to a short hash", long_tag)
m._EMBEDDED_MODIFIERS.clear()
check(m.get_cached_5d_dct(alias) is None, "unknown alias falls back to the base name (absent here)")
m._EMBEDDED_MODIFIERS[alias] = pruned
check(m.get_cached_5d_dct(alias) is pruned, "a registered alias resolves to the scoped payload")
check(m.register_embedded_modifiers({alias: pruned}, origin="test") == 1, "partial payload registers")

# ----------------------------------------------------------------- 9. real-world sizes
print("\n9. size on a raw-storage payload (the heavy case)")
import base64, os as _os
raw = make_payload(storage="raw")
for e in raw["layers"].values():
    e.pop("u_dct"); e.pop("v_dct")
    e["u_raw"] = base64.b64encode(_os.urandom(2 * 5 * min(e["d_out"], 4096))).decode()
    e["v_raw"] = base64.b64encode(_os.urandom(2 * 5 * min(e["d_in"], 4096))).decode()
pruned, stats = m.prune_5d_payload_to_scopes(raw, [("model", "  ↳ Block_3B (11)", "All Components", "All Sub-Tensors")])
check(m.payload_worst_fidelity(pruned) is None, "raw payloads carry no frozen fidelity")
print(f"     {kb(raw)/1024:.1f} MB -> {kb(pruned)/1024:.2f} MB  ({kb(raw)/kb(pruned):.0f}x smaller)")


# ----------------------------------------------------------------- 10. sub-tensor scopes
print("\n10. single sub-tensor scopes (the finest targeting)")
def make_named(order=8):
    layers = {}
    def add(n, d, o=3072, i=3072):
        layers[n] = {"domain": d, "d_out": o, "d_in": i, "singular_values": [1.0] * 5,
                     "u_dct": [[0.01] * order] * 5, "v_dct": [[0.01] * order] * 5}
    for b in range(28):
        for t in ("attn.wq", "attn.wk", "attn.wv", "attn.wo", "mlp.gate", "mlp.up", "mlp.down"):
            add(f"blocks.{b}.{t}", "model")
        add(f"blocks.{b}.attn.qkv", "model")            # a fused LoRA tensor
    for l in range(36):
        for t in ("self_attn.q_proj", "self_attn.k_proj", "mlp.up_proj", "mlp.down_proj"):
            add(f"text_model.encoder.layers.{l}.{t}", "clip", 1280, 1280)
    return {"version": "5.0", "storage": "dct", "harmonic_order": order,
            "normalized": True, "layers": layers}

named = make_named()
sc = [("model", "  ↳ Block_3B (11)", "All Components", "ATTN_wq_query")]
pruned, stats = m.prune_5d_payload_to_scopes(named, sc)
check(set(pruned["layers"]) == {"blocks.11.attn.wq"}, "one tensor of one block", sorted(pruned["layers"]))
check(runtime_selection(named, *sc[0]) == runtime_selection(pruned, *sc[0]), "EQUIVALENCE")
check("blocks.11.attn.qkv" not in pruned["layers"], "a fused qkv is not mistaken for WQ")

sc_clip = [("clip", "  ↳ Layer_1A (0)", "All Components", "ATTN_q_proj")]
pruned_c, _ = m.prune_5d_payload_to_scopes(named, sc_clip)
check(set(pruned_c["layers"]) == {"text_model.encoder.layers.0.self_attn.q_proj"},
      "CLIP surgeon labels match despite the map's '.weight' suffixes", sorted(pruned_c["layers"]))
check(runtime_selection(named, *sc_clip[0]) == runtime_selection(pruned_c, *sc_clip[0]), "EQUIVALENCE")

whole = m.prune_5d_payload_to_scopes(named, [("model", "  ↳ Block_3B (11)", "All Components", "All Sub-Tensors")])[0]
check(len(pruned["layers"]) * 8 == len(whole["layers"]),
      "8x lighter than the same block without a sub-tensor",
      f"{len(pruned['layers'])} vs {len(whole['layers'])}")
print(f"     whole modifier {kb(named):.0f} KB | block only {kb(whole):.1f} KB | "
      f"block+WQ {kb(pruned):.1f} KB  ({kb(named)/kb(pruned):.0f}x)")

sc_group = [("model", "Block_3 (All 10-14)", "MLP (Gate/Up/Down)", "MLP_down_proj")]
pruned_g, _ = m.prune_5d_payload_to_scopes(named, sc_group)
check(set(pruned_g["layers"]) == {f"blocks.{b}.mlp.down" for b in range(10, 15)},
      "sub_tensor intersects with sub_components across a block group", sorted(pruned_g["layers"]))
check(runtime_selection(named, *sc_group[0]) == runtime_selection(pruned_g, *sc_group[0]), "EQUIVALENCE")

sc_conflict = [("model", "  ↳ Block_3B (11)", "MLP (Gate/Up/Down)", "ATTN_wq_query")]
check(runtime_selection(named, *sc_conflict[0]) == set(), "contradictory widgets select nothing (and say so)")

sc_all = [("model", "All Blocks (0-27)", "All Components", "ATTN_wq_query")]
pruned_a, _ = m.prune_5d_payload_to_scopes(named, sc_all)
check(set(pruned_a["layers"]) == {f"blocks.{b}.attn.wq" for b in range(28)},
      "one tensor across every block", len(pruned_a["layers"]))
check(runtime_selection(named, *sc_all[0]) == runtime_selection(pruned_a, *sc_all[0]), "EQUIVALENCE")

print("\n" + ("ALL PASS" if not FAIL else f"{len(FAIL)} FAILURES: {FAIL}"))
sys.exit(1 if FAIL else 0)
