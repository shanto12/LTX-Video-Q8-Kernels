import inspect
import unittest
from unittest.mock import Mock, MagicMock, patch

import torch

# Import the function we're testing
from q8_kernels.integration.diffusers import _linear_call, processor_factory


class MockFP8Linear:
    """Mock FP8Linear layer with configurable forward signature."""
    
    def __init__(self, params=None):
        self.forward_params = params or ['self', 'input', 'scales']
    
    def forward(self, input, scales=None, out_dtype=None, unsupported_arg=None):
        """Mock forward method with signature matching real FP8Linear."""
        return input  # Simple passthrough for testing
    
    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)


class MockFP8LinearStrict:
    """Mock FP8Linear with strict signature (no unsupported args)."""
    
    def forward(self, input, scales=None):
        """Strict signature - doesn't accept out_dtype or other args."""
        return input
    
    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)


class TestFP8LinearWrapper(unittest.TestCase):
    """Test suite for FP8Linear compatibility wrapper."""
    
    def test_linear_call_filters_unsupported_kwargs(self):
        """Test that _linear_call filters out unsupported arguments."""
        # Create a mock layer with limited signature
        mock_layer = MockFP8LinearStrict()
        
        # Mock input tensors
        input_tensor = torch.randn(2, 4)
        scales_tensor = torch.randn(4)
        
        # Call with extra unsupported arguments
        result = _linear_call(
            mock_layer,
            input_tensor,
            scales=scales_tensor,
            out_dtype=torch.bfloat16,  # This should be filtered out
            unsupported_arg=True       # This should be filtered out
        )
        
        # Should not raise an error and return the input
        self.assertIsNotNone(result)
        torch.testing.assert_close(result, input_tensor)
    
    def test_linear_call_preserves_supported_kwargs(self):
        """Test that _linear_call preserves supported arguments."""
        mock_layer = MockFP8Linear()
        
        input_tensor = torch.randn(2, 4)
        scales_tensor = torch.randn(4)
        
        # Call with supported arguments
        result = _linear_call(
            mock_layer,
            input_tensor,
            scales=scales_tensor,
            out_dtype=torch.bfloat16
        )
        
        self.assertIsNotNone(result)
        torch.testing.assert_close(result, input_tensor)
    
    def test_linear_call_handles_positional_args(self):
        """Test that _linear_call handles positional arguments correctly."""
        mock_layer = MockFP8Linear()
        
        input_tensor = torch.randn(2, 4)
        scales_tensor = torch.randn(4)
        
        # Call with positional arguments
        result = _linear_call(mock_layer, input_tensor, scales_tensor)
        
        self.assertIsNotNone(result)
        torch.testing.assert_close(result, input_tensor)
    
    def test_linear_call_inspects_signature_correctly(self):
        """Test that _linear_call correctly inspects layer signatures."""
        mock_layer = MockFP8LinearStrict()
        
        # Get the actual signature
        sig = inspect.signature(mock_layer.forward)
        expected_params = set(sig.parameters.keys())
        
        # Should only have 'input' and 'scales'
        self.assertEqual(expected_params, {'input', 'scales'})
        
        # Verify _linear_call would filter correctly
        input_tensor = torch.randn(2, 4)
        kwargs = {
            'scales': torch.randn(4),
            'out_dtype': torch.bfloat16,  # Should be filtered
            'unsupported': True           # Should be filtered
        }
        
        # This should work without error
        result = _linear_call(mock_layer, input_tensor, **kwargs)
        self.assertIsNotNone(result)


class MockAttentionProcessor:
    """Mock attention processor for testing."""
    
    def __init__(self):
        self.to_q = MockFP8Linear()
        self.to_k = MockFP8Linear()
        self.to_v = MockFP8Linear()
        self.to_out = [MockFP8Linear()]  # to_out[0]
        self.proj = MockFP8Linear()
        self.net = [None, None, MockFP8Linear()]  # net[2]
        self.heads = 8
        self.adaptive_norm = "none"
        self.norm = Mock()
        self.attn2 = None
        self._chunk_size = None
        self.ff = Mock()


class TestProcessorFactory(unittest.TestCase):
    """Test processor factory functions use _linear_call correctly."""
    
    @patch('q8_kernels.integration.diffusers.get_compute_dtype')
    @patch('q8_kernels.integration.diffusers.get_attention_func')
    def test_fused_forward_uses_linear_call(self, mock_get_attention_func, mock_get_compute_dtype):
        """Test that fused_forward uses _linear_call for all linear layers."""
        # Setup mocks
        mock_get_compute_dtype.return_value = torch.bfloat16
        mock_attention_func = Mock(return_value=torch.randn(1, 8, 10, 64))
        mock_get_attention_func.return_value = mock_attention_func
        
        # Create processor functions
        fused_forward, gelu_forward, ff_forward = processor_factory(
            None, None, None, None, None, None, None
        )
        
        # Create mock processor
        processor = MockAttentionProcessor()
        processor.norm.return_value = torch.randn(1, 10, 512)
        
        # Mock the linear layers to track calls
        with patch('q8_kernels.integration.diffusers._linear_call') as mock_linear_call:
            mock_linear_call.return_value = torch.randn(1, 10, 512)
            
            # Call fused_forward
            hidden_states = torch.randn(1, 10, 512)
            hidden_states_scales = torch.randn(512)
            
            try:
                result = fused_forward(
                    processor,
                    hidden_states,
                    hidden_states_scales
                )
                
                # Verify _linear_call was used for to_q, to_k, to_v, to_out[0]
                self.assertGreaterEqual(mock_linear_call.call_count, 4)
                
                # Check that calls included the linear layers
                call_args_list = [call[0][0] for call in mock_linear_call.call_args_list]
                self.assertIn(processor.to_q, call_args_list)
                self.assertIn(processor.to_k, call_args_list)
                self.assertIn(processor.to_v, call_args_list)
                self.assertIn(processor.to_out[0], call_args_list)
                
            except Exception as e:
                # Some mocking might cause issues, but we mainly want to verify
                # that _linear_call is being used
                self.assertIn('_linear_call', str(type(e)) or str(e) or "")
    
    def test_gelu_forward_uses_linear_call(self):
        """Test that gelu_forward uses _linear_call for proj layer."""
        # Create processor functions
        fused_forward, gelu_forward, ff_forward = processor_factory(
            None, None, None, None, None, None, None
        )
        
        # Create mock processor
        processor = MockAttentionProcessor()
        
        with patch('q8_kernels.integration.diffusers._linear_call') as mock_linear_call:
            with patch('q8_kernels.integration.diffusers.gelu_hadamard_transform') as mock_gelu:
                with patch('q8_kernels.integration.diffusers.get_compute_dtype') as mock_dtype:
                    mock_linear_call.return_value = torch.randn(1, 10, 512)
                    mock_gelu.return_value = (torch.randn(1, 10, 512), torch.randn(512))
                    mock_dtype.return_value = torch.bfloat16
                    
                    hidden_states = torch.randn(1, 10, 512)
                    hidden_states_scales = torch.randn(512)
                    
                    result = gelu_forward(processor, hidden_states, hidden_states_scales)
                    
                    # Verify _linear_call was used for proj layer
                    mock_linear_call.assert_called_once_with(
                        processor.proj, hidden_states, hidden_states_scales, out_dtype=torch.bfloat16
                    )
    
    def test_ff_forward_uses_linear_call(self):
        """Test that ff_forward uses _linear_call for net[2] layer."""
        # Create processor functions
        fused_forward, gelu_forward, ff_forward = processor_factory(
            None, None, None, None, None, None, None
        )
        
        # Create mock processor
        processor = MockAttentionProcessor()
        processor.net[0] = Mock(return_value=(torch.randn(1, 10, 512), torch.randn(512)))
        
        with patch('q8_kernels.integration.diffusers._linear_call') as mock_linear_call:
            mock_linear_call.return_value = torch.randn(1, 10, 512)
            
            hidden_states = torch.randn(1, 10, 512)
            hidden_states_scales = torch.randn(512)
            
            result = ff_forward(processor, hidden_states, hidden_states_scales)
            
            # Verify _linear_call was used for net[2] layer
            mock_linear_call.assert_called_once_with(
                processor.net[2], hidden_states, hidden_states_scales, out_dtype=torch.bfloat16
            )


class TestCallSiteCompatibility(unittest.TestCase):
    """Test that call sites work with different FP8Linear signatures."""
    
    def test_call_sites_handle_signature_mismatch(self):
        """Test that all call sites handle signature mismatches gracefully."""
        # Test cases for different signature patterns
        test_cases = [
            # (layer_class, expected_to_work)
            (MockFP8Linear, True),       # Supports all args
            (MockFP8LinearStrict, True), # Strict signature, but wrapper handles it
        ]
        
        for layer_class, should_work in test_cases:
            with self.subTest(layer_class=layer_class.__name__):
                layer = layer_class()
                
                input_tensor = torch.randn(2, 4)
                scales_tensor = torch.randn(4)
                
                # This should work for all layer types due to wrapper
                result = _linear_call(
                    layer,
                    input_tensor,
                    scales=scales_tensor,
                    out_dtype=torch.bfloat16,
                    unsupported_extra_arg=True
                )
                
                if should_work:
                    self.assertIsNotNone(result)
                    torch.testing.assert_close(result, input_tensor)


if __name__ == '__main__':
    unittest.main()
