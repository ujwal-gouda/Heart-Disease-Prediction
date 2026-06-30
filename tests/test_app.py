import unittest

from app import app


class AppTests(unittest.TestCase):
    def setUp(self):
        self.client = app.test_client()

    def test_home_page_renders(self):
        response = self.client.get("/")
        self.assertEqual(response.status_code, 200)

    def test_prediction_endpoint(self):
        response = self.client.post(
            "/",
            data={
                "age": "55",
                "sex": "Male",
                "restingBP": "140",
                "cholesterol": "220",
                "fastingBS": "0",
                "maxHR": "130",
                "exerciseAngina": "No",
                "oldPeak": "1.2",
                "stSlope": "Flat",
                "chestPainType": "ASY",
                "restingECG": "Normal",
            },
            follow_redirects=True,
        )
        self.assertEqual(response.status_code, 200)
        self.assertIn(b"Probability", response.data)


if __name__ == "__main__":
    unittest.main()
