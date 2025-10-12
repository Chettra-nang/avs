import numpy as np
from highway_datacollection.collection.collector import SynchronizedCollector


def test_grayscale_normalisation_transpose():
    # Create a small synchronized collector instance (we only need the _process_binary_observation method)
    c = SynchronizedCollector()

    # Create a fake grayscale observation in (C, W, H) format with distinct dims
    C, W, H = 4, 8, 6
    arr = np.arange(C * W * H, dtype=np.uint8).reshape((C, W, H))

    processed = {}
    # Use modality config stub with storage_enabled True
    class _Cfg:
        storage_enabled = True

    cfg = _Cfg()

    # Call the internal method
    c._process_binary_observation(arr, processed, 'GrayscaleObservation', cfg)

    # After processing, grayscale_shape should be (C, H, W)
    shape = tuple(processed.get('grayscale_shape', []))
    assert shape == (C, H, W), f"Expected shape (C,H,W) after normalisation, got {shape}"

    # Decode blob using BinaryArrayEncoder to confirm content matches transposed array
    from highway_datacollection.storage.encoders import BinaryArrayEncoder
    encoder = BinaryArrayEncoder()
    decoded = encoder.decode(processed['grayscale_blob'], tuple(processed['grayscale_shape']), processed['grayscale_dtype'])

    assert decoded.shape == (C, H, W)
    # spot-check a few elements
    assert decoded[0, 0, 0] == arr[0, 0, 0]
    assert decoded[-1, -1, -1] == arr[-1, -1, -1]
