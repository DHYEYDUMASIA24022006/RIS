"""Test CIFAR-10 data handling, model forward/backward, and training sanity check."""

import os
import sys
import unittest
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fl.models import CIFAR10Net, get_model
from fl.data_partition import load_dataset_with_validation


class TestCIFAR10Training(unittest.TestCase):
    """Stage 2: CIFAR-10 model and local training test."""

    def setUp(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def test_model_forward_shape(self):
        model = get_model('cifar10', self.device)
        dummy_input = torch.randn(8, 3, 32, 32, device=self.device)
        output = model(dummy_input)
        self.assertEqual(output.shape, (8, 10))

    def test_loss_and_gradients(self):
        model = get_model('cifar10', self.device)
        optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
        criterion = nn.CrossEntropyLoss()

        dummy_input = torch.randn(4, 3, 32, 32, device=self.device)
        dummy_target = torch.tensor([0, 3, 5, 9], device=self.device, dtype=torch.long)

        output = model(dummy_input)
        loss = criterion(output, dummy_target)
        self.assertTrue(torch.isfinite(loss).item())

        optimizer.zero_grad()
        loss.backward()

        for name, param in model.named_parameters():
            if param.requires_grad:
                self.assertIsNotNone(param.grad, f"Missing gradient for {name}")
                self.assertTrue(torch.any(param.grad != 0).item(), f"Zero gradient for {name}")

    def test_overfit_tiny_batch(self):
        """Sanity check: model can overfit 4 images within 40 steps."""
        model = get_model('cifar10', self.device)
        optimizer = optim.Adam(model.parameters(), lr=0.01)
        criterion = nn.CrossEntropyLoss()

        torch.manual_seed(42)
        dummy_input = torch.randn(4, 3, 32, 32, device=self.device)
        dummy_target = torch.tensor([0, 1, 2, 3], device=self.device, dtype=torch.long)

        initial_loss = criterion(model(dummy_input), dummy_target).item()

        for _ in range(120):
            optimizer.zero_grad()
            out = model(dummy_input)
            loss = criterion(out, dummy_target)
            loss.backward()
            optimizer.step()

        final_loss = criterion(model(dummy_input), dummy_target).item()
        self.assertLess(final_loss, initial_loss * 0.1, "Model failed to overfit tiny batch")

    def test_model_size_bits(self):
        model = get_model('cifar10', self.device)
        total_params = sum(p.numel() for p in model.parameters())
        model_bits = total_params * 32
        print(f"\n[Info] CIFAR10 model params: {total_params:,}, size: {model_bits / 1e6:.2f} Mbits ({model_bits / 8 / 1024:.1f} KB)")


if __name__ == '__main__':
    unittest.main()
