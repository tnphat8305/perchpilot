"""Export C1 policy or C5 belief weights as little-endian f32 for the Rust node."""
import argparse
import hashlib
import json
from pathlib import Path
from tempfile import NamedTemporaryFile

import torch

from perch_core import check_policy_meta


def policy_layers(state, module):
    names = sorted({key.rsplit(".", 1)[0] for key in state
                    if key.startswith("actor.") and key.endswith(".weight")},
                   key=lambda name: int(name.split(".")[-1]))
    if not names or not all(f"{name}.bias" in state for name in names):
        raise ValueError("expected a sequential linear actor with biases")
    if module is None:
        # best.pt from rl/ppo.py uses ELU. The checkpoint holds only parameters.
        return [{"name": name, "activation": "elu" if i < len(names) - 1 else "linear"}
                for i, name in enumerate(names)]
    modules = list(module.actor.named_children())
    names_from_module = []
    layers = []
    for idx, layer in modules:
        kind = layer.original_name.lower()
        if kind == "linear":
            name = f"actor.{idx}"
            names_from_module.append(name)
            layers.append({"name": name, "activation": "linear"})
        elif kind in ("elu", "tanh", "relu") and layers and layers[-1]["activation"] == "linear":
            layers[-1]["activation"] = kind
        else:
            raise ValueError(f"unsupported policy actor module {layer.original_name}")
    if names_from_module != names:
        raise ValueError("policy actor tensor layout does not match its modules")
    return layers


def export(source, kind, out):
    meta = json.loads(source.with_name(f"{kind}.json").read_text())
    module = torch.jit.load(str(source), map_location="cpu").eval() if source.suffix == ".ts" else None
    state = module.state_dict() if module is not None else torch.load(source, map_location="cpu", weights_only=True)
    if kind == "policy":
        check_policy_meta(meta)
        layers = policy_layers(state, module)
        names = ["norm.mean", "norm.var"] + [f"{layer['name']}.{param}" for layer in layers
                                             for param in ("weight", "bias")]
        metadata = {"policy": meta, "layers": layers}
    else:
        names = ["mean", "std", "mask", "temp"] + [
            f"{layer}.{param}" for layer in ("conv.0", "conv.2", "conv.4", "head.1", "head.3")
            for param in ("weight", "bias")]
        metadata = {"belief": meta, "temperature": state["temp"].item(), "mask": state["mask"].tolist()}

    with source.open("rb") as stream:
        source_sha = hashlib.file_digest(stream, "sha256").hexdigest()
    filename = f"{kind}.{source_sha}.bin"
    out.mkdir(parents=True, exist_ok=True)
    tensors = {}
    with NamedTemporaryFile(dir=out, delete=False) as stream:
        tmp = Path(stream.name)
        try:
            for name in names:
                value = state[name].detach().cpu().numpy().astype("<f4", copy=False)
                tensors[name] = {"shape": list(value.shape), "offset": stream.tell()}
                stream.write(value.tobytes(order="C"))
            stream.flush()
            tmp.replace(out / filename)
        finally:
            tmp.unlink(missing_ok=True)
    header = dict(version=1, kind=kind, file=filename, source=source.name, source_sha256=source_sha,
                  tensors=tensors, metadata=metadata)
    with NamedTemporaryFile(mode="w", dir=out, delete=False) as stream:
        json.dump(header, stream, indent=2, allow_nan=False)
        stream.write("\n")
    path = out / f"{kind}.rust.json"
    Path(stream.name).replace(path)
    return path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("kind", choices=("policy", "belief"))
    parser.add_argument("source", type=Path, help="policy.ts, belief.ts or best.pt, next to the model metadata JSON")
    parser.add_argument("--out", type=Path, help="output directory (defaults to the source run directory)")
    args = parser.parse_args()
    print(export(args.source, args.kind, args.out or args.source.parent))


if __name__ == "__main__":
    main()
