import streamlit as st
import pickle
from text_processing import tokenize
from starting_verb_extractor import StartingVerbExtractor
import time

# Load the trained model using pickle
with open("models/classifier.pkl", "rb") as model_file:
    loaded_model = pickle.load(model_file)

category_names = [
    "related",
    "request",
    "offer",
    "aid_related",
    "medical_help",
    "medical_products",
    "search_and_rescue",
    "security",
    "military",
    "child_alone",
    "water",
    "food",
    "shelter",
    "clothing",
    "money",
    "missing_people",
    "refugees",
    "death",
    "other_aid",
    "infrastructure_related",
    "transport",
    "buildings",
    "electricity",
    "tools",
    "hospitals",
    "shops",
    "aid_centers",
    "other_infrastructure",
    "weather_related",
    "floods",
    "storm",
    "fire",
    "earthquake",
    "cold",
    "other_weather",
    "direct_report",
]

# Function to check if 5 hours have passed
@st.cache(ttl=60 * 60 * 5)  # 60 seconds * 60 minutes * 5 hours
def check_time():
    return time.time()

# Streamlit app
def main():
    # Check if 5 hours have passed, if yes, rerun the code
    if check_time():
        st.title("Disaster Message Classifier")

        # Load image
        image_path = "app/ai_kapil.png"
        st.image(image_path, use_column_width=True)

        # Problem statement
        st.markdown(
            """
            This app predicts categories for disaster-related messages using a trained machine learning model.
            Enter a message, and click the 'Classify' button to see predicted categories.

            ### How It Works:
            - **Enter a Message:** Type a message related to disaster response.
            - **Click 'Classify':** The app uses a trained model to predict relevant categories.
            - **View Predictions:** See the predicted categories based on the input message.

            The goal is to assist disaster response teams in quickly assessing and acting upon incoming messages.
            """
        )

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

        if classify_button:
            if not user_input:
                st.warning("Please enter a message.")
            else:
                processed_input = " ".join(tokenize(user_input))
                predicted_categories = loaded_model.predict([processed_input])

                st.subheader("Predicted categories:")
                if any(predicted_categories[0]):
                    selected_categories = [
                        category_names[i]
                        for i in range(len(predicted_categories[0]))
                        if predicted_categories[0][i] == 1
                    ]

                    table_data = {"Predicted Categories": selected_categories}
                    st.table(table_data)
                else:
                    st.warning("No categories predicted for the given input.")

        st.text_area("Sample input examples:", value=example_inputs, height=200)

        # Add footer
        st.markdown("<br><br> <br><br> <br><br>", unsafe_allow_html=True)  # Adding space
        st.markdown("Made by Kapil Wankhede (Sanghamitra Tech)")


if __name__ == "__main__":
    while True:
        main()
        time.sleep(5 * 60 * 60)  # Sleep for 5 hours (in seconds)
