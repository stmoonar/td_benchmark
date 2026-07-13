"""Row-match analysis for c-group verify dumps.

Usage:
  python tools/debug_row_match.py results/verify_dump_c1/c1_rank0.pt
"""

import sys

import torch


def row_cos(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    a_n = a / a.norm(dim=1, keepdim=True).clamp(min=1e-12)
    b_n = b / b.norm(dim=1, keepdim=True).clamp(min=1e-12)
    return (a_n * b_n).sum(dim=1)


def main() -> int:
    blob = torch.load(sys.argv[1], map_location="cpu")
    out = blob["output"]
    gl = blob["golden_local"]
    gf = blob["golden_full"]
    print(f"scheme={blob['scheme']} rank={blob['rank']} output={tuple(out.shape)} golden_full={tuple(gf.shape)}")
    print(f"output |max|={out.abs().max():.6f}  golden_local |max|={gl.abs().max():.6f}")

    direct = row_cos(out, gl)
    print(f"[direct]     per-row cos: mean={direct.mean():.4f}  min={direct.min():.4f}  frac(cos>0.99)={(direct > 0.99).float().mean():.4f}")

    out_n = out / out.norm(dim=1, keepdim=True).clamp(min=1e-12)
    gf_n = gf / gf.norm(dim=1, keepdim=True).clamp(min=1e-12)
    sim = out_n @ gf_n.T
    best_cos, best_idx = sim.max(dim=1)
    print(f"[best-match] per-row cos: mean={best_cos.mean():.4f}  min={best_cos.min():.4f}  frac(cos>0.99)={(best_cos > 0.99).float().mean():.4f}")
    print(f"[best-match] matched indices unique={best_idx.unique().numel()}/{out.shape[0]}")
    print(f"[best-match] first 16 matched idx: {best_idx[:16].tolist()}")
    expected_start = blob["rank"] * out.shape[0]
    identity = (best_idx == torch.arange(expected_start, expected_start + out.shape[0])).float().mean()
    print(f"[best-match] frac(idx == rank-slice identity)={identity:.4f}")

    scale = (out.flatten() @ gl.flatten()) / (gl.flatten() @ gl.flatten()).clamp(min=1e-12)
    print(f"[H3 check]   least-squares scale(output vs golden_local)={scale:.4f} (1.0 = no scaling issue)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
