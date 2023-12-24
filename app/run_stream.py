import os
import sys
import streamlit as st
import pickle
from text_processing import tokenize
from starting_verb_extractor import StartingVerbExtractor
import time

# Add the parent directory of the script to the Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Load the trained model using pickle
with open("models/classifier.pkl", "rb") as model_file:
    loaded_model = pickle.load(model_file)

category_names = [
    "related", "request", "offer", "aid_related", "medical_help",
    "medical_products", "search_and_rescue", "security", "military",
    "child_alone", "water", "food", "shelter", "clothing", "money",
    "missing_people", "refugees", "death", "other_aid",
    "infrastructure_related", "transport", "buildings", "electricity",
    "tools", "hospitals", "shops", "aid_centers", "other_infrastructure",
    "weather_related", "floods", "storm", "fire", "earthquake", "cold",
    "other_weather", "direct_report",
]

# Function to check if 5 hours have passed
@st.cache(ttl=60 * 60 * 5, suppress_st_warning=True)  # 60 seconds * 60 minutes * 5 hours
def check_time():
    return time.time()

# Streamlit app
@st.cache(ttl=60 * 60 * 5, suppress_st_warning=True)  # 60 seconds * 60 minutes * 5 hours
def main():
    # Get user input
    user_input = st.text_input("Enter a message:")

    # Examples as a text area
    example_inputs = (
        "1. Please, we need tents and water. We are in Silo, Thank you!\n"
        "2. I am in Croix-des-Bouquets. We have health issues. They ( workers ) are in Santo 15. ( an area in Croix-des-Bouquets )\n"
        "3. Good evening, is the earthquake end?\n"
        "4. People from Dal blocked since Wednesday in Carrefour, we having water shortage, food and medical assistance."
    )

    classify_button = st.button("Classify")

    # Check if 5 hours have passed, if yes, rerun the code
    if check_time():
        if classify_button:
            if not user_input:
                st.warning("Please enter a message.")
            else:
                # Preprocess the user input using the tokenize function
                processed_input = " ".join(tokenize(user_input))

                # Use the loaded model to predict categories
                predicted_categories = loaded_model.predict([processed_input])

                # Display the predicted categories in a table
                st.subheader("Predicted categories:")
                if any(predicted_categories[0]):
                    selected_categories = [
                        category_names[i]
                        for i in range(len(predicted_categories[0]))
                        if predicted_categories[0][i] == 1
                    ]

                    # Create a table with one column for predicted categories
                    table_data = {"Predicted Categories": selected_categories}
                    st.table(table_data)
                else:
                    st.warning("No categories predicted for the given input.")

        st.text_area("Sample input examples:", value=example_inputs, height=200)

    # Add footer
    st.markdown("<br><br> <br><br> <br><br>", unsafe_allow_html=True)  # Adding space
    st.markdown("Made by Kapil Wankhede (Sanghamitra Tech)")


if __name__ == "__main__":
    main()
