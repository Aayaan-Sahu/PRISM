"""ONNX export stub (deployment step, after a model is trained).

The streaming graph must expose LSTM states as explicit inputs/outputs:

    inputs : spec_frame (1, 1, F) as real/imag pair, spk_cond (1, spk_dim),
             h0..h2, c0..c2 (1, 1, hidden)
    outputs: masked frame + updated states

Notes for the implementer:
  * export the post-FiLM graph with the conditioning vector e precomputed at
    enrollment time (spk_mlp runs once per speaker, not per frame);
  * complex tensors are not ONNX-friendly — split spec into (real, imag)
    channels and apply the mask as 2x2 real multiplication;
  * verify parity against StreamingSession on random input before shipping.
"""

raise NotImplementedError("ONNX export is a post-training deployment task — see notes above")
