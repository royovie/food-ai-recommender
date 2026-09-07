import os
import requests
from dotenv import load_dotenv

load_dotenv(override=True)

API_KEY = os.getenv("SPOONACULAR_API_KEY")

print("API key loaded:", bool(API_KEY))
print("API key ending:", API_KEY[-4:] if API_KEY else "NONE")

BASE_URL = "https://api.spoonacular.com/recipes/complexSearch"


def get_recipes(food_name, number=5):
    params = {
        "apiKey": API_KEY,
        "query": food_name.replace("_", " "),
        "number": number,
        "addRecipeInformation": True
    }

    response = requests.get(
        BASE_URL,
        params=params,
        timeout=10
    )

    response.raise_for_status()

    data = response.json()

    recipes = []

    for item in data.get("results", []):
        recipes.append({
            "id": item.get("id"),
            "title": item.get("title"),
            "image": item.get("image"),
            "source_url": item.get("sourceUrl")
        })

    return recipes
