import torch


class TabularDomainProjector:
    """
    Projects perturbed tabular samples back onto the *feasible* data domain.

    Real-valued L_p balls are a poor threat model for tabular NIDS features:
    many fields are discrete encodings (protocol, service, state), binary
    flags, or counts, and deployments may pin operational metadata as
    immutable. A domain-constrained adversary must perturb only what a real
    attacker could realistically change.

    Given per-column valid levels (in the same scaled space as the model
    input), `project` enforces:
      - binary columns: snapped to {0, 1}
      - discrete/categorical columns: snapped to the nearest observed level
      - immutable columns: restored to the original sample value
      - all columns: clamped to [0, 1]

    The valid levels are derived from the processed training data (see
    `build_projector_from_dataframe`), so no magic constants are encoded here.
    """

    def __init__(self, levels, feature_cols, binary=(), discrete=(), immutable=()):
        """
        Args:
            levels: dict {column name -> sorted 1D tensor of valid values}
            feature_cols: ordered list of feature column names (model input order)
            binary: column names snapped to {0, 1}
            discrete: column names snapped to the nearest valid level
            immutable: column names restored to the original value
        """
        self.feature_cols = list(feature_cols)
        self.col_idx = {c: i for i, c in enumerate(self.feature_cols)}

        missing = (set(binary) | set(discrete) | set(immutable)) - set(self.feature_cols)
        if missing:
            raise KeyError(f"domain spec references unknown columns: {sorted(missing)}")

        self.binary = [self.col_idx[c] for c in binary]
        self.discrete = [self.col_idx[c] for c in discrete]
        self.immutable = [self.col_idx[c] for c in immutable]

        self.levels = {
            self.col_idx[c]: torch.as_tensor(v, dtype=torch.float32).sort().values
            for c, v in levels.items()
            if c in self.col_idx
        }

    def _snap_to_levels(self, values, levels):
        """Snap each value to the nearest entry of a sorted 1D tensor of levels."""
        if levels.numel() == 1:
            return values.new_full(values.shape, levels.item())
        pos = torch.searchsorted(levels, values.contiguous())
        hi = pos.clamp(max=levels.numel() - 1)
        lo = (pos - 1).clamp(min=0)
        lo_val, hi_val = levels[lo], levels[hi]
        return torch.where((values - lo_val).abs() <= (hi_val - values).abs(), lo_val, hi_val)

    def project(self, x_adv, x_orig):
        """Map a perturbed batch back onto the feasible domain. Shapes: (B, d)."""
        x = x_adv.clone()
        for j in self.binary:
            x[:, j] = torch.round(x[:, j]).clamp(0.0, 1.0)
        for j in self.discrete:
            levels = self.levels.get(j)
            if levels is not None:
                x[:, j] = self._snap_to_levels(x[:, j], levels)
        for j in self.immutable:
            x[:, j] = x_orig[:, j]
        return x.clamp(0.0, 1.0)

    def __call__(self, x_adv, x_orig):
        return self.project(x_adv, x_orig)


def build_projector_from_dataframe(df, feature_cols, spec):
    """
    Build a TabularDomainProjector whose valid levels are the observed values
    of each constrained column in the processed dataset.

    Args:
        df: processed pandas DataFrame
        feature_cols: ordered feature column names (model input order)
        spec: dict with optional keys "binary", "discrete", "immutable"
              (lists of column names)
    """
    binary = list(spec.get("binary", []))
    discrete = list(spec.get("discrete", []))
    immutable = list(spec.get("immutable", []))

    levels = {}
    for col in discrete:
        if col not in df.columns:
            raise KeyError(f"domain spec references unknown column: {col}")
        levels[col] = sorted(df[col].dropna().unique().tolist())

    return TabularDomainProjector(
        levels=levels,
        feature_cols=feature_cols,
        binary=binary,
        discrete=discrete,
        immutable=immutable,
    )
