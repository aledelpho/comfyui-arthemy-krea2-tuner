"""The node labels are a contract with the user's node search; the class keys are a
contract with every saved workflow. This asserts both: the rotator family reads as one
family, and no rename ever leaks into a class key."""
import collections, sys, inspect
import stub_env, importlib
m = importlib.import_module("Arthemy_Krea2_Tuner")

dn, cn = m.NODE_DISPLAY_NAME_MAPPINGS, m.NODE_CLASS_MAPPINGS
FAIL = []
def chk(c, label, extra=""):
    print(f"  {'ok  ' if c else 'FAIL'} {label}{'' if c else '  ' + str(extra)}")
    if not c:
        FAIL.append(label)

print("\nrotator family: <Domain> <how you aim it> Rotator")
EXPECTED = {
    "ArthemyKrea2ModelRotator":      "\U0001F7EA\U0001F33F Model Axis Rotator",
    "ArthemyKrea2CLIPRotator":       "\U0001F7E8\U0001F33F CLIP Axis Rotator",
    "ArthemyKrea2LatentSpaceRotator": "\U0001F7EA\U0001F9ED Model Compass Rotator",
    "ArthemyKrea2CLIPSpaceRotator":  "\U0001F7E8\U0001F9ED CLIP Compass Rotator",
    "ArthemyKrea2ModelChaosRotator": "\U0001F7EA\U0001F300 Model Chaos Rotator",
    "ArthemyKrea2CLIPChaosRotator":  "\U0001F7E8\U0001F300 CLIP Chaos Rotator",
}
for key, label in EXPECTED.items():
    chk(dn.get(key) == label, f"{key} -> {label}", f"got {dn.get(key)!r}")
    # The class key is what ComfyUI writes into a saved workflow and what the JS panel
    # keys off (ROTATOR_NODE_NAMES, isCompass via "SpaceRotator"). A rename that reaches
    # this dict silently orphans every workflow that used the node.
    chk(key in cn, f"{key} still registered under its original class key")

print("\ninvariants")
chk(all(str(v)[:1] in "\U0001F7EA\U0001F7E8\U0001FA77" for v in dn.values()),
    "every label keeps a family emoji prefix (arthemyFamilyColor reads it)")
dupes = [n for n, c in collections.Counter(dn.values()).items() if c > 1 and "Legacy" not in n]
chk(not dupes, "no two nodes share a label", dupes)
chk(all(n.endswith("Rotator") for k, n in dn.items() if "Rotator" in k and "Legacy" not in n),
    "every rotator label ends in 'Rotator'")
chk("Style Compass" not in " ".join(dn.values()), "the old 'Style Compass' label is gone")

print("\ninfo headlines match the labels")
src = inspect.getsource(m._StyleCompassMixin._compass) + inspect.getsource(m._DualRotationMixin._dual)
chk("Compass Rotator " in src, "compass headline renamed")
chk("Axis Rotator " in src, "axis headline renamed")

print("\n" + ("ALL PASS" if not FAIL else f"{len(FAIL)} FAILURES: {FAIL}"))
sys.exit(1 if FAIL else 0)
