"""
Unit tests for binary array encoders.
"""

import pytest
import numpy as np
from highway_datacollection.storage.encoders import BinaryArrayEncoder


class TestBinaryArrayEncoder:
    """Test binary array encoding and decoding functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.encoder = BinaryArrayEncoder()
    
    def test_encode_decode_simple_array(self):
        """Test encoding and decoding a simple array."""
        # Create test array
        original = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32)
        
        # Encode
        encoded = self.encoder.encode(original)
        
        # Verify encoded structure
        assert 'blob' in encoded
        assert 'shape' in encoded
        assert 'dtype' in encoded
        assert encoded['shape'] == [2, 3]
        assert encoded['dtype'] == 'float32'
        assert isinstance(encoded['blob'], bytes)
        assert len(encoded['blob']) > 0
        
        # Decode
        decoded = self.encoder.decode(
            encoded['blob'], 
            tuple(encoded['shape']), 
            encoded['dtype']
        )
        
        # Verify reconstruction
        np.testing.assert_array_equal(original, decoded)
        assert decoded.dtype == original.dtype
        assert decoded.shape == original.shape
    
    def test_encode_decode_different_dtypes(self):
        """Test encoding/decoding with different data types."""
        test_cases = [
            (np.array([1, 2, 3], dtype=np.int32), np.int32),
            (np.array([1.0, 2.0, 3.0], dtype=np.float64), np.float64),
            (np.array([True, False, True], dtype=np.bool_), np.bool_),
            (np.array([1, 2, 3], dtype=np.uint8), np.uint8)
        ]
        
        for original, expected_dtype in test_cases:
            encoded = self.encoder.encode(original)
            decoded = self.encoder.decode(
                encoded['blob'],
                tuple(encoded['shape']),
                encoded['dtype']
            )
            
            np.testing.assert_array_equal(original, decoded)
            assert decoded.dtype == expected_dtype
    
    def test_encode_decode_multidimensional_array(self):
        """Test encoding/decoding multidimensional arrays."""
        # Create 3D array (like occupancy grid)
        original = np.random.rand(10, 10, 3).astype(np.float32)
        
        encoded = self.encoder.encode(original)
        decoded = self.encoder.decode(
            encoded['blob'],
            tuple(encoded['shape']),
            encoded['dtype']
        )
        
        np.testing.assert_array_almost_equal(original, decoded, decimal=6)
        assert decoded.shape == original.shape
        assert decoded.dtype == original.dtype
    
    def test_encode_empty_array(self):
        """Test encoding empty arrays."""
        original = np.array([], dtype=np.float32)
        
        encoded = self.encoder.encode(original)
        decoded = self.encoder.decode(
            encoded['blob'],
            tuple(encoded['shape']),
            encoded['dtype']
        )
        
        np.testing.assert_array_equal(original, decoded)
        assert decoded.dtype == original.dtype
    
    def test_encode_none_array(self):
        """Test encoding None values."""
        encoded = self.encoder.encode(None)
        
        assert encoded['blob'] == b''
        assert encoded['shape'] == []
        assert encoded['dtype'] == 'float32'
        
        decoded = self.encoder.decode(b'', (), 'float32')
        assert decoded.shape == ()
        assert decoded.dtype == np.float32
        assert decoded == 0.0
    
    def test_encode_multiple_arrays(self):
        """Test encoding multiple arrays at once."""
        arrays = {
            'occ': np.random.rand(5, 5).astype(np.float32),
            'gray': np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
        }
        
        encoded = self.encoder.encode_multiple(arrays)
        
        # Check all expected keys are present
        expected_keys = [
            'occ_blob', 'occ_shape', 'occ_dtype',
            'gray_blob', 'gray_shape', 'gray_dtype'
        ]
        for key in expected_keys:
            assert key in encoded
        
        # Verify shapes and dtypes
        assert encoded['occ_shape'] == [5, 5]
        assert encoded['occ_dtype'] == 'float32'
        assert encoded['gray_shape'] == [64, 64, 3]
        assert encoded['gray_dtype'] == 'uint8'
    
    def test_decode_multiple_arrays(self):
        """Test decoding multiple arrays."""
        # Create test arrays
        original_arrays = {
            'occ': np.random.rand(5, 5).astype(np.float32),
            'gray': np.random.randint(0, 255, (32, 32), dtype=np.uint8)
        }
        
        # Encode
        encoded = self.encoder.encode_multiple(original_arrays)
        
        # Decode
        decoded_arrays = self.encoder.decode_multiple(encoded, ['occ', 'gray'])
        
        # Verify reconstruction
        np.testing.assert_array_almost_equal(
            original_arrays['occ'], 
            decoded_arrays['occ'], 
            decimal=6
        )
        np.testing.assert_array_equal(
            original_arrays['gray'], 
            decoded_arrays['gray']
        )
    
    def test_decode_shape_mismatch_error(self):
        """Test error handling for shape mismatches."""
        original = np.array([[1, 2], [3, 4]], dtype=np.int32)
        encoded = self.encoder.encode(original)
        
        # Try to decode with wrong shape
        with pytest.raises(ValueError, match="Shape mismatch"):
            self.encoder.decode(
                encoded['blob'],
                (3, 2),  # Wrong shape
                encoded['dtype']
            )
    
    def test_dtype_conversion(self):
        """Test automatic dtype conversion when needed."""
        original = np.array([1.0, 2.0, 3.0], dtype=np.float64)
        encoded = self.encoder.encode(original)
        
        # Decode with different dtype
        decoded = self.encoder.decode(
            encoded['blob'],
            tuple(encoded['shape']),
            'float32'  # Different dtype
        )
        
        assert decoded.dtype == np.float32
        np.testing.assert_array_almost_equal(original, decoded, decimal=6)
    
    def test_large_array_handling(self):
        """Test handling of large arrays (memory efficiency)."""
        # Create a reasonably large array
        large_array = np.random.rand(100, 100, 10).astype(np.float32)
        
        encoded = self.encoder.encode(large_array)
        decoded = self.encoder.decode(
            encoded['blob'],
            tuple(encoded['shape']),
            encoded['dtype']
        )
        
        np.testing.assert_array_almost_equal(large_array, decoded, decimal=6)
        assert decoded.shape == large_array.shape
        assert decoded.dtype == large_array.dtype