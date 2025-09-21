"""
Binary array encoders for efficient storage.
"""

from typing import Dict, Any, Optional
import numpy as np
import io


class BinaryArrayEncoder:
    """Encode numpy arrays as binary blobs with metadata."""
    
    def __init__(self, compression: Optional[str] = None):
        """
        Initialize encoder with optional compression.
        
        Args:
            compression: Compression method ('gzip', 'lz4', etc.) or None
        """
        self._compression = compression
    
    def encode(self, array: np.ndarray) -> Dict[str, Any]:
        """
        Encode array as binary blob with metadata.
        
        Args:
            array: Numpy array to encode
            
        Returns:
            Dictionary with blob, shape, and dtype information
        """
        if array is None:
            return {
                'blob': b'',
                'shape': [],
                'dtype': 'float32'
            }
        
        # Convert array to bytes
        buffer = io.BytesIO()
        np.save(buffer, array, allow_pickle=False)
        blob = buffer.getvalue()
        
        return {
            'blob': blob,
            'shape': list(array.shape),
            'dtype': str(array.dtype)
        }
    
    def decode(self, blob: bytes, shape: tuple, dtype: str) -> np.ndarray:
        """
        Decode binary blob back to numpy array.
        
        Args:
            blob: Binary data
            shape: Original array shape
            dtype: Original array dtype
            
        Returns:
            Reconstructed numpy array
        """
        if not blob:
            if not shape:  # Empty shape means scalar
                return np.array(0, dtype=dtype)
            else:
                return np.zeros(shape, dtype=dtype)
        
        # Load array from bytes
        buffer = io.BytesIO(blob)
        array = np.load(buffer, allow_pickle=False)
        
        # Verify shape and dtype match
        if list(array.shape) != list(shape):
            raise ValueError(f"Shape mismatch: expected {shape}, got {array.shape}")
        
        if str(array.dtype) != dtype:
            array = array.astype(dtype)
        
        return array
    
    def encode_single(self, array: np.ndarray, prefix: str = "") -> Dict[str, Any]:
        """
        Encode a single array with optional prefix.
        
        Args:
            array: Numpy array to encode
            prefix: Optional prefix for the keys
            
        Returns:
            Dictionary with blob, shape, and dtype information
        """
        encoded = self.encode(array)
        
        if prefix:
            return {
                f"{prefix}_blob": encoded['blob'],
                f"{prefix}_shape": encoded['shape'],
                f"{prefix}_dtype": encoded['dtype']
            }
        else:
            return encoded
    
    def encode_multiple(self, arrays: Dict[str, np.ndarray]) -> Dict[str, Any]:
        """
        Encode multiple arrays with prefixed keys.
        
        Args:
            arrays: Dictionary of arrays to encode
            
        Returns:
            Dictionary with encoded data for all arrays
        """
        result = {}
        
        for key, array in arrays.items():
            encoded = self.encode(array)
            result[f"{key}_blob"] = encoded['blob']
            result[f"{key}_shape"] = encoded['shape']
            result[f"{key}_dtype"] = encoded['dtype']
        
        return result
    
    def decode_multiple(self, data: Dict[str, Any], array_keys: list) -> Dict[str, np.ndarray]:
        """
        Decode multiple arrays from encoded data.
        
        Args:
            data: Dictionary containing encoded array data
            array_keys: List of array keys to decode
            
        Returns:
            Dictionary of decoded arrays
        """
        result = {}
        
        for key in array_keys:
            blob = data.get(f"{key}_blob", b'')
            shape = data.get(f"{key}_shape", [])
            dtype = data.get(f"{key}_dtype", 'float32')
            
            result[key] = self.decode(blob, tuple(shape), dtype)
        
        return result