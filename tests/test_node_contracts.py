"""Every registered node still satisfies the ComfyUI node contract.

Cheap to run, and it catches the class of mistake that only shows up as a red node in the
browser: a FUNCTION that no longer exists after a rename, a combo whose default is not one of
its own options, a RETURN_NAMES that drifted out of step with RETURN_TYPES, a display name
for a node that was deleted.
"""
import sys
import stub_env  # noqa: F401  - fakes torch and comfy so the module imports without ComfyUI
import importlib
import inspect

m = importlib.import_module("Arthemy_Krea2_Tuner")

FAIL = []
def chk(c, label, extra=""):
    if not c:
        FAIL.append(label)
        print(f"  FAIL {label}   <- {extra}")


CLS = m.NODE_CLASS_MAPPINGS
DISP = m.NODE_DISPLAY_NAME_MAPPINGS
print(f"\n{len(CLS)} registered nodes")

print("\n1. mappings agree in both directions")
chk(set(CLS) == set(DISP), "every class has a display name and vice versa",
    sorted(set(CLS) ^ set(DISP)))

print("\n1b. no two nodes share a display name")
import collections as _c
_by = _c.defaultdict(list)
for _k, _v in DISP.items():
    _by[_v].append(_k)
for _v, _ks in _by.items():
    # Two rows with the identical label in the node menu are indistinguishable; the label is
    # cosmetic (a saved workflow stores the class key), so a duplicate is free to fix.
    chk(len(_ks) == 1, f"display name is unique: {_v}", _ks)

print("\n2. FUNCTION exists and can receive every declared input")
for name, cls in CLS.items():
    fn_name = getattr(cls, "FUNCTION", None)
    chk(fn_name and hasattr(cls, fn_name), f"{name}.FUNCTION", fn_name)
    if not (fn_name and hasattr(cls, fn_name)):
        continue
    sig = inspect.signature(getattr(cls, fn_name))
    takes_kwargs = any(p.kind is p.VAR_KEYWORD for p in sig.parameters.values())
    spec = cls.INPUT_TYPES()
    declared = list(spec.get("required", {})) + list(spec.get("optional", {}))
    for key in declared:
        chk(takes_kwargs or key in sig.parameters, f"{name}: '{key}' reaches {fn_name}()")
    # a parameter with no default that INPUT_TYPES never supplies can never be called
    for pname, p in sig.parameters.items():
        if pname == "self" or p.kind in (p.VAR_KEYWORD, p.VAR_POSITIONAL):
            continue
        if p.default is inspect.Parameter.empty:
            chk(pname in declared, f"{name}: required parameter '{pname}' is declared")

print("\n3. RETURN_TYPES and RETURN_NAMES line up")
for name, cls in CLS.items():
    rt = getattr(cls, "RETURN_TYPES", ())
    rn = getattr(cls, "RETURN_NAMES", None)
    chk(isinstance(rt, tuple), f"{name}.RETURN_TYPES is a tuple", type(rt))
    if rn is not None:
        chk(len(rt) == len(rn), f"{name}: {len(rt)} types vs {len(rn)} names")

print("\n4. no combo is empty, and every default is one of its own options")
for name, cls in CLS.items():
    spec = cls.INPUT_TYPES()
    for section in ("required", "optional"):
        for key, decl in (spec.get(section) or {}).items():
            if not (isinstance(decl, (tuple, list)) and decl):
                continue
            opts = decl[0]
            if not isinstance(opts, (list, tuple)):
                continue          # a type name like "MODEL", not a combo
            chk(len(opts) > 0, f"{name}.{key}: combo has options")
            default = (decl[1] or {}).get("default") if len(decl) > 1 and isinstance(decl[1], dict) else None
            if default is not None and opts:
                chk(default in opts, f"{name}.{key}: default is selectable", repr(default))

print("\n5. CATEGORY is set (otherwise the node hides in the root menu)")
for name, cls in CLS.items():
    chk(bool(getattr(cls, "CATEGORY", "")), f"{name}.CATEGORY")

print("\n6. every recipe family the suite writes is known to the reset / clear path")
written = set()
src = open(m.__file__.replace(".pyc", ".py"), encoding="utf-8").read()
import re
for mo in re.finditer(r'append_recipe\([^,]+,\s*"([a-z0-9_]+)"', src):
    written.add(mo.group(1))
for mo in re.finditer(r'model_options\[\s*"(arthemy_[a-z0-9_]+)"\s*\]\s*=', src):
    written.add(mo.group(1))
for mo in re.finditer(r'RECIPE_KEY\s*=\s*"(arthemy_[a-z0-9_]+)"', src):
    written.add(mo.group(1))
for fam in sorted(written):
    chk(fam in m.ARTHEMY_RECIPE_KEYS, f"'{fam}' is in ARTHEMY_RECIPE_KEYS")
print(f"     families written: {sorted(written)}")

print("\n7. every node the Preset Loader replays through actually exists")
# The loader reproduces a recipe by CALLING a node. A rename inside the suite therefore breaks
# it as a NameError / AttributeError at load time, in a code path no unit test walks - which is
# exactly how `ArthemyKrea2ChaosRotator().rotate_chaos()` survived: neither name existed.
import ast as _ast
_tree = _ast.parse(src)
_loader = [n for n in _ast.walk(_tree) if isinstance(n, _ast.FunctionDef) and n.name == "load_preset"]
chk(len(_loader) == 1, "load_preset found in the source", len(_loader))
for _call in _ast.walk(_loader[0]) if _loader else []:
    if not (isinstance(_call, _ast.Call) and isinstance(_call.func, _ast.Attribute)):
        continue
    inner = _call.func.value
    if not (isinstance(inner, _ast.Call) and isinstance(inner.func, _ast.Name)
            and inner.func.id.startswith("Arthemy")):
        continue
    cls_name, meth = inner.func.id, _call.func.attr
    cls = getattr(m, cls_name, None)
    chk(cls is not None, f"loader calls a class that exists: {cls_name}")
    if cls is not None:
        chk(hasattr(cls, meth), f"loader calls {cls_name}.{meth}()")
        if hasattr(cls, meth):
            chk(getattr(cls, "FUNCTION", meth) == meth,
                f"{cls_name}.{meth}() is the node's own FUNCTION", getattr(cls, "FUNCTION", None))

print("\nALL PASS" if not FAIL else f"\n{len(FAIL)} FAILED")
sys.exit(1 if FAIL else 0)
