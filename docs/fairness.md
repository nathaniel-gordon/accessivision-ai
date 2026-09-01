# Fairness Evaluation

Disparity Δ is the max pairwise gap in model quality across protected
attribute buckets (skin-tone × gender × age). Evaluated on the FairFace split.

Baseline SFT Δ = 0.182. With our demographic-parity regulariser, Δ drops to
0.155 (-15%) with no loss to mean quality (CLIP-I actually improves from
0.712 to 0.735).
