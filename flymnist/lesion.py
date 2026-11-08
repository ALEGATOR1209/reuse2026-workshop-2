"""Knockouts: silence groups of neurons or cut the connections between the hemispheres.

A rule = what to hit (a region, an anatomical structure, or cell types by name),
which side (left / right / both) and how much (10-100% of those neurons, picked at
random but reproducibly). All rules are combined into one set of silenced neurons.
"""

import fnmatch

import numpy as np

# (key, label, category, how to select): category is only used to group the menu.
#   ("region", i)            -> display region i (see server.REGIONS)
#   ("neuropil", "ME")       -> neurons whose FlyWire cell_class mentions that neuropil
#   ("class", [...])         -> FlyWire cell_class in list
#   ("super", [...])         -> FlyWire super_class in list
GROUPS = [
    ("whole", "Whole brain", "Regions", ("all", None)),
    ("eye", "Eye input (stimulated neurons)", "Regions", ("region", 0)),
    ("optic", "Optic lobe", "Regions", ("region", 1)),
    ("vpn", "Visual projection neurons", "Regions", ("region", 2)),
    ("central", "Central brain", "Regions", ("region", 3)),
    ("descending", "Descending neurons (brain → body)", "Regions", ("region", 4)),
    ("ascending", "Ascending neurons (body → brain)", "Regions", ("super", ["ascending", "sensory_ascending"])),
    ("lamina", "Lamina", "Optic lobe structures", ("neuropil", "LA")),
    ("medulla", "Medulla", "Optic lobe structures", ("neuropil", "ME")),
    ("lobula", "Lobula", "Optic lobe structures", ("neuropil", "LO")),
    ("lobula_plate", "Lobula plate", "Optic lobe structures", ("neuropil", "LOP")),
    ("kc", "Mushroom body: Kenyon cells", "Central brain structures", ("class", ["Kenyon_Cell"])),
    ("mbon", "Mushroom body output neurons (MBON)", "Central brain structures", ("class", ["MBON"])),
    ("dan", "Dopaminergic neurons (DAN)", "Central brain structures", ("class", ["DAN"])),
    ("al", "Antennal lobe (local + projection neurons)", "Central brain structures", ("class", ["ALLN", "ALPN"])),
    ("lh", "Lateral horn", "Central brain structures", ("class", ["LHLN", "LHCENT"])),
    ("cx", "Central complex", "Central brain structures", ("class", ["CX"])),
    ("olfactory", "Olfactory receptor neurons", "Sensory", ("class", ["olfactory"])),
    ("gustatory", "Taste neurons", "Sensory", ("class", ["gustatory"])),
    ("mechano", "Mechanosensory neurons", "Sensory", ("class", ["mechanosensory"])),
]


def cross_hemisphere_mask(ann, pre, post):
    """True for every connection between a left-side and a right-side neuron."""
    side = ann.side.to_numpy()
    s = np.select([side == "left", side == "right"], [1, 2], 0)
    return ((s[pre] == 1) & (s[post] == 2)) | ((s[pre] == 2) & (s[post] == 1))


class Lesions:
    def __init__(self, ann, region, pre, post):
        self.n = len(ann)
        self.side = ann.side.to_numpy()
        self.cell_type = ann.cell_type.fillna("").to_numpy()
        self._types_lower = np.char.lower(self.cell_type.astype(str))
        tokens = ann.cell_class.fillna("").str.split(r"[>.]")
        cell_class = ann.cell_class.to_numpy()
        super_class = ann.super_class.to_numpy()

        self.groups = {}
        for key, label, cat, (kind, arg) in GROUPS:
            if kind == "all":
                m = np.ones(self.n, bool)
            elif kind == "region":
                m = region == arg
            elif kind == "neuropil":
                m = tokens.apply(lambda t: arg in t).to_numpy()
            elif kind == "class":
                m = np.isin(cell_class, arg)
            else:
                m = np.isin(super_class, arg)
            self.groups[key] = (label, cat, m)

        self.cross = cross_hemisphere_mask(ann, pre, post)

    def options(self):
        cnt = lambda m: {"both": int(m.sum()), "left": int((m & (self.side == "left")).sum()),
                         "right": int((m & (self.side == "right")).sum())}
        return {"groups": [{"key": k, "label": lab, "category": cat, "neurons": cnt(m)} for k, (lab, cat, m) in self.groups.items()],
                "cross_hemisphere_connections": int(self.cross.sum())}

    def _base(self, rule):
        if rule.get("group"):
            if rule["group"] not in self.groups:
                raise ValueError(f"unknown group {rule['group']}")
            return self.groups[rule["group"]][2]
        patterns = [p.strip().lower() for p in str(rule.get("cell_types", "")).split(",") if p.strip()]
        if not patterns:
            raise ValueError("give a group or cell type names (e.g. Tm1, LPLC2, DN*)")
        uniq = np.unique(self._types_lower)
        hits = [t for t in uniq if t and any(fnmatch.fnmatchcase(t, p) for p in patterns)]
        return np.isin(self._types_lower, hits)

    def resolve(self, rule):
        m = self._base(rule).copy()
        if rule.get("side", "both") in ("left", "right"):
            m &= self.side == rule["side"]
        frac = float(rule.get("fraction", 1.0))
        idx = np.flatnonzero(m)
        if frac < 1.0 and len(idx):
            rng = np.random.default_rng(int(rule.get("seed", 0)))
            keep = rng.permutation(idx)[: int(round(frac * len(idx)))]
            m = np.zeros(self.n, bool)
            m[keep] = True
        return m

    def build(self, rules, cut_hemispheres):
        """Returns (silenced mask, cut-synapse mask or None, per-rule neuron counts)."""
        silenced = np.zeros(self.n, bool)
        per_rule = []
        for r in rules:
            m = self.resolve(r)
            per_rule.append(int(m.sum()))
            silenced |= m
        return silenced, (self.cross if cut_hemispheres else None), per_rule
