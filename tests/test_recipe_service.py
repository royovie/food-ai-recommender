from unittest.mock import patch
from backend.recipe_service import get_recipes

@patch("backend.recipe_service.requests.get")
def test_get_recipes(mock_get):
    mock_get.return_value.status_code = 200
    mock_get.return_value.json.return_value = {
        "results": [
            {
                "id": 1,
                "title": "Pizza Recipe",
                "image": "pizza.jpg"
            }
        ]
    }

    recipes = get_recipes("pizza")

    assert len(recipes) == 1
    assert recipes[0]["title"] == "Pizza Recipe"