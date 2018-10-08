import unittest
from torch.autograd import gradcheck
from quantize import *

class TestQuantize(unittest.TestCase):
    def test_softmin_back(self):
        x = (torch.rand(20, 10, dtype=torch.double, requires_grad=True),)
        self.assertTrue(gradcheck(softmin, x))

    def test_quantize(self):
        x = 2*torch.rand((100, 100), dtype=torch.double) - 1
        centers = 2*torch.arange(200, dtype=torch.double)*1/200 - 1
        qx = quantize(x, centers, 0.1)
        ax = (x * 100).round() / 100
        print(x)
        print(qx)
        self.assertTrue(torch.mean(torch.abs(qx - ax)) < 1/400)

if __name__ == '__main__':
    unittest.main()
