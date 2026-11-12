import pytest
import torch

from hyper_fsad import Features, HyperFSAD, sparsemax


def test_sparsemax_known_projection_and_distractor_rejection():
    z = torch.tensor([[0.8, 0.7, -0.5]], dtype=torch.float64)
    expected = torch.tensor([[0.55, 0.45, 0.0]], dtype=torch.float64)
    torch.testing.assert_close(sparsemax(z), expected)
    extended = torch.cat([z, torch.full((1, 100), -0.9)], dim=1)
    torch.testing.assert_close(sparsemax(extended)[:, :3], expected)
    assert torch.count_nonzero(sparsemax(extended)[:, 3:]) == 0


def test_simplex_kkt_and_shift_invariance():
    torch.manual_seed(4)
    z = torch.randn(7, 13, dtype=torch.float64)
    w = sparsemax(z)
    torch.testing.assert_close(w.sum(-1), torch.ones(7, dtype=z.dtype))
    assert (w >= 0).all()
    tau = (z - w).masked_fill(w == 0, float("nan" )).nanmean(-1, keepdim=True)
    torch.testing.assert_close(w, (z - tau).clamp_min(0))
    torch.testing.assert_close(w, sparsemax(z + 100))
    torch.testing.assert_close(w.T, sparsemax(z.T, dim=0))


def test_analytic_dual_branch_scores():
    # These vectors should give cosine distances of 0, 1 and 2.
    support = Features(torch.tensor([[[[1., 0.]]]]), torch.tensor([[[1., 0.]]]), (1, 1))
    vectors = torch.tensor([[1., 0.], [0., 1.], [-1., 0.]])
    query = Features(vectors[None, :, None], vectors[None], (1, 1))
    result = HyperFSAD().fit(support).predict(query)
    torch.testing.assert_close(result.image_score, torch.tensor([0., 1., 2.]))
    torch.testing.assert_close(result.patch_map[:, 0, 0], result.image_score)
    assert result.resize_maps((9, 7)).shape == (3, 9, 7)


def test_layer_average_max_pool_and_global_nearest_support():
    p = torch.tensor([[[[1., 0.]], [[0., 1.]]]]).repeat(2, 1, 1, 1)
    support = Features(p, p[:, :, 0], (1, 1))
    q = torch.tensor([[[[1., 0.], [-1., 0.]]], [[[0., 1.], [0., -1.]]]])
    query = Features(q, torch.tensor([[[1., 0.]], [[0., 1.]]]), (1, 2))
    result = HyperFSAD().fit(support).predict(query)
    torch.testing.assert_close(result.global_score, torch.zeros(1))
    torch.testing.assert_close(result.spatial_score, torch.ones(1))
    torch.testing.assert_close(result.image_score, torch.full((1,), 0.5))


def test_chunking_batching_and_support_permutation():
    torch.manual_seed(2)
    support = Features(torch.randn(4, 2, 6, 8), torch.randn(4, 2, 8), (2, 3))
    query = Features(torch.randn(4, 3, 6, 8), torch.randn(4, 3, 8), (2, 3))
    expected = HyperFSAD(chunk_size=1000).fit(support).predict(query)
    shuffled = Features(support.patches.flip(1, 2), support.cls.flip(1), support.grid)
    actual = HyperFSAD(chunk_size=2).fit(shuffled).predict(query)
    torch.testing.assert_close(actual.patch_map, expected.patch_map)
    torch.testing.assert_close(actual.image_score, expected.image_score)
    for i in range(3):
        single = Features(query.patches[:, i:i+1], query.cls[:, i:i+1], query.grid)
        torch.testing.assert_close(HyperFSAD().fit(support).predict(single).image_score, expected.image_score[i:i+1])


def test_reject_invalid_inputs():
    with pytest.raises(ValueError):
        sparsemax(torch.tensor([float("nan")]))
    with pytest.raises(ValueError):
        HyperFSAD(chunk_size=0)
    features = Features(torch.zeros(1, 1, 1, 2), torch.ones(1, 1, 2), (1, 1))
    with pytest.raises(RuntimeError):
        HyperFSAD().predict(features)
    with pytest.raises(ValueError):
        HyperFSAD().fit(features)
