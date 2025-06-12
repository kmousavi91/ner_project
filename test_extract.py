import unittest
from fastapi.testclient import TestClient
from apifast import app

client = TestClient(app)

class TestEntityExtraction(unittest.TestCase):

    def test_basic_entities(self):
        text = "Asus 250-G8 Laptop mit 512GB SSD in mattschwarz"
        response = client.post("/extract", json={"text": text})
        self.assertEqual(response.status_code, 200)

        data = response.json()
        labels = [(ent["label"], ent["text"].lower()) for ent in data["entities"]]

        self.assertIn(("COLOR", "mattschwarz"), labels)
        self.assertTrue(any(l == "STORAGE" and "ssd" in t for l, t in labels))
        self.assertTrue(any(l == "BRAND_MODEL" and "asus" in t for l, t in labels))

    def test_irrelevant_text(self):
        text = "Keine relevante Information hier."
        response = client.post("/extract", json={"text": text})
        self.assertEqual(response.status_code, 200)

        data = response.json()
        self.assertEqual(len(data["entities"]), 0)

if __name__ == "__main__":
    unittest.main(verbosity=2)

