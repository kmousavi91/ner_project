import unittest
from fastapi.testclient import TestClient
from apifast import app

class TestAPI(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(app)

    def test_extract_entities(self):
        response = self.client.post("/extract", json={"text": "Asus Laptop mit 512GB SSD in silber"})
        self.assertEqual(response.status_code, 200)

        data = response.json()
        labels = [ent["label"] for ent in data["entities"]]

        self.assertIn("BRAND_MODEL", labels)
        self.assertIn("STORAGE", labels)
        self.assertIn("COLOR", labels)

if __name__ == "__main__":
    unittest.main()
