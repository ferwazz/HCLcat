import unittest
import numpy as np
from HCLcat import *


class Test_HCLcat(unittest.TestCase):
    def setUp(self):
        """
        Crea variables que se usarán en las pruebas
        """

        self.LightCurve: np.array = np.array([1, 2, 3, 4, 5])


    def test_snr(self):
        output = snr(self.LightCurve)
        self.assertAlmostEqual(output, 2.1213)


if __name__ == "__main__":
    unittest.main()
