import streamlit as st
import requests

API_URL = "http://127.0.0.1:8000/predict"
RECIPE_URL = "http://127.0.0.1:8000/recipes"

st.set_page_config(
    page_title="AI Food Recipe Recommender",
    page_icon="🍕",
    layout="wide"
)

st.title("🍽️ AI Food Recipe Recommender")

uploaded_file = st.file_uploader(
    "Upload a food image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:

    st.image(
        uploaded_file,
        caption="Uploaded image",
        width=400
    )

    if st.button("Analyze Food"):

        files = {
            "file": (
                uploaded_file.name,
                uploaded_file.getvalue(),
                uploaded_file.type
            )
        }

        response = requests.post(
            API_URL,
            files=files,
            timeout=30
        )

        response.raise_for_status()

        data = response.json()

        # Save API results in session state
        st.session_state["prediction"] = data["prediction"]
        st.session_state["confidence"] = data["confidence"]
        st.session_state["recipes"] = data["recipes"]
        st.session_state["top_predictions"] = data.get(
            "top_predictions",
            []
        )


# Show results even after Streamlit reruns
if "prediction" in st.session_state:

    prediction = st.session_state["prediction"]
    confidence = st.session_state["confidence"]

    if prediction == "uncertain":
        st.warning(
            "The model is not confident about this image. "
            "Enter the food name below to search for recipes."
        )
    else:
        st.success("Food identified!")

    st.subheader(
        f"Prediction: {prediction.replace('_', ' ').title()}"
    )

    st.write(
        f"Confidence: **{confidence}%**"
    )

    # Optional: show top 3 predictions
    top_predictions = st.session_state.get(
        "top_predictions",
        []
    )

    if top_predictions:
        st.write("Top predictions:")

        for item in top_predictions:
            food_name = item["food"].replace("_", " ").title()
            score = item["confidence"]

            st.write(
                f"- {food_name}: {score}%"
            )

    refined_query = st.text_input(
        "Want more specific recipes?",
        value=(
            ""
            if prediction == "uncertain"
            else prediction.replace("_", " ")
        ),
        key="refined_query"
    )

    if st.button("Find Recipes"):

        if refined_query.strip():

            recipe_response = requests.get(
                RECIPE_URL,
                params={
                    "query": refined_query
                },
                timeout=30
            )

            recipe_response.raise_for_status()

            recipe_data = recipe_response.json()

            st.session_state["recipes"] = (
                recipe_data["recipes"]
            )

        else:
            st.warning(
                "Enter a food name before searching."
            )


# Display recipes
if "recipes" in st.session_state:

    recipes = st.session_state["recipes"]

    if recipes:

        st.divider()

        st.header("Recommended Recipes")

        columns = st.columns(3)

        for index, recipe in enumerate(recipes):

            column = columns[index % 3]

            with column:

                if recipe.get("image"):
                    st.image(
                        recipe["image"],
                        use_container_width=True
                    )

                st.subheader(
                    recipe["title"]
                )

                if recipe.get("source_url"):
                    st.link_button(
                        "View Recipe",
                        recipe["source_url"]
                    )