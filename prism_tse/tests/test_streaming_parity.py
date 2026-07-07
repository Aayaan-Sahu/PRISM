"""The load-bearing test: offline forward must equal hop-by-hop streaming."""

import torch

from prism_tse.config import Config
from prism_tse.models.separator import PrismTSE, count_params
from prism_tse.models.stft import CausalStft
from prism_tse.streaming import StreamingSession

torch.manual_seed(0)


def _setup():
    cfg = Config()
    stft = CausalStft(cfg.audio.n_fft, cfg.audio.hop)
    model = PrismTSE(cfg).eval()
    return cfg, stft, model


def test_param_budget():
    _, _, model = _setup()
    n = count_params(model)
    assert 1.5e6 < n < 3.5e6, f"param count drifted: {n:,}"


def test_stft_roundtrip():
    _, stft, _ = _setup()
    x = torch.randn(2, 4096)
    y = stft.istft(stft.stft(x), 4096)
    assert torch.allclose(x, y, atol=1e-5), (x - y).abs().max()


def test_cola():
    _, stft, _ = _setup()
    w2 = stft.window**2
    # product window (analysis*synthesis = hann) sums to 1 at hop = n_fft/2
    assert torch.allclose(w2[: stft.hop] + w2[stft.hop :], torch.ones(stft.hop), atol=1e-6)


def test_offline_equals_streaming():
    cfg, stft, model = _setup()
    n = 16 * cfg.audio.hop
    x = torch.randn(1, n) * 0.1
    emb = torch.nn.functional.normalize(torch.randn(1, cfg.model.emb_dim), dim=-1)

    with torch.no_grad():
        est_spec, _ = model(stft.stft(x), emb)
        y_offline = stft.istft(est_spec, n).squeeze(0)

    session = StreamingSession(model, stft, emb.squeeze(0))
    y_stream = session.process(x.squeeze(0))

    assert y_offline.shape == y_stream.shape
    err = (y_offline - y_stream).abs().max()
    assert err < 1e-4, f"parity error {err}"


def test_state_carry_across_calls():
    """Processing two halves hop-by-hop with carried state == one pass."""
    cfg, stft, model = _setup()
    hop = cfg.audio.hop
    n = 8 * hop
    x = torch.randn(n) * 0.1
    emb = torch.nn.functional.normalize(torch.randn(cfg.model.emb_dim), dim=-1)

    full = StreamingSession(model, stft, emb)
    outs_full = [full.process_hop(x[k * hop : (k + 1) * hop]) for k in range(8)]

    split = StreamingSession(model, stft, emb)
    outs_split = [split.process_hop(x[k * hop : (k + 1) * hop]) for k in range(4)]
    outs_split += [split.process_hop(x[k * hop : (k + 1) * hop]) for k in range(4, 8)]

    for a, b in zip(outs_full, outs_split):
        assert torch.allclose(a, b, atol=1e-6)
