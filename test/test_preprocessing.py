import numpy as np
from vispyx.preprocessing import apply_clahe

def test_apply_clahe_shape():
    dummy_image = np.random.randint(0, 256, size=(128, 128), dtype=np.uint8)
    result = apply_clahe(dummy_image)
    assert result.shape == dummy_image.shape

def test_apply_clahe_type():
    dummy_image = np.random.randint(0, 256, size=(128, 128), dtype=np.uint8)
    result = apply_clahe(dummy_image)
    assert isinstance(result, np.ndarray)