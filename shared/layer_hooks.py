"""Paper-aligned residual-stream hook sequence for each architecture."""

from __future__ import annotations

from shared.component import Component
from shared.models import canonical_slug


def residual_components(model) -> list[Component]:
    """Return the n_layers+1 residual slots used by patching experiments.

    Qwen3 DeepStack additions happen between decoder blocks. Using the next
    block's input captures those additions; the final slot remains the output
    of the last block. Legacy models retain the original post-embed + per-block
    output convention.
    """
    n_layers = model.cfg.n_layers
    if canonical_slug(model.model_name) == "qwen3vl":
        return (
            [Component("resid_pre", layer=layer) for layer in range(n_layers)]
            + [Component("resid_post", layer=n_layers - 1)]
        )
    return (
        [Component("resid_pre", layer=0)]
        + [Component("resid_post", layer=layer) for layer in range(n_layers)]
    )
