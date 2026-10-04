import pytest
pytest.importorskip("sklearn")

from slflood.sklearn import SublevelFloodBifiltration

from multipers.ml.mma import FilteredComplex2MMA
import torch
from sklearn.pipeline import Pipeline

def test_sklearn():
    X = torch.rand((10,10000,3))

    pipeline = Pipeline([("slf", SublevelFloodBifiltration(100, 0.2, use_triton=True)),
                         # ("mma", FilteredComplex2MMA(n_jobs=-1))
                         ])

    Y = pipeline.fit_transform(X)