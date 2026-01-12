from __future__ import annotations

import string

import numpy as np
import pytest

from projector_utils import decomp


class TestTSVD:
    def test_tsvd_ng(self) -> None:
        with pytest.raises(ValueError, match=r".*between 1 and arr.ndim - 1.*"):
            decomp.tsvd(np.eye(3), 0)
        with pytest.raises(ValueError, match=r".*between 1 and arr.ndim - 1.*"):
            decomp.tsvd(np.eye(3), 2)

    @pytest.mark.parametrize("nu", [1, 2, 3, 4])
    def test_tsvd(self, rng: np.random.Generator, nu: int) -> None:
        shapes = (2, 3, 4, 5, 6)
        arr = rng.normal(size=shapes)
        u, s, v = decomp.tsvd(arr, nu)
        ind_u = "".join(string.ascii_uppercase[:nu])
        ind_v = "".join(string.ascii_uppercase[nu : arr.ndim])
        usv = np.asarray(np.einsum(f"{ind_u}x,x,{ind_v}x->{ind_u}{ind_v}", u, s, v))
        np.testing.assert_allclose(arr, usv)
