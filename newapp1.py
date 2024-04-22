import streamlit as st
import pandas as pd
import joblib

# Load the saved models
model_min = joblib.load('model_min.pkl')
model_max = joblib.load('model_max.pkl')

# Load the data to get the list of states and categories
data = pd.read_csv("Maindata.csv")
states = data['State'].unique()
categories = data['Category'].unique()

# Function to predict budget
def predict_budget(state_input, category_input):
    # Predict the budget
    X_pred = pd.DataFrame({
        # Include all possible categories and states
        'Category_Adventure/Scenic': [0],  # Set to 0 as default
        'Category_Amusement Park/Recreation': [0],  # Set to 0 as default
        # Include other category columns similarly
        'State_Assam': [0],  # Set to 0 as default
        'State_Other_State': [0],  # Set to 0 as default
    })

    # If the selected category/state is in the prediction input, set its corresponding column to 1
    if 'Category_' + category_input in X_pred.columns:
        X_pred['Category_' + category_input] = 1
    if 'State_' + state_input in X_pred.columns:
        X_pred['State_' + state_input] = 1

    # Ensure all columns in X_pred are consistent with the model's feature names
    model_features = model_min.feature_names_in_
    X_pred = X_pred.reindex(columns=model_features, fill_value=0)

    min_budget_pred = model_min.predict(X_pred)[0]
    max_budget_pred = model_max.predict(X_pred)[0]
    return min_budget_pred, max_budget_pred

# Streamlit web application
def main():
    st.title("TRAVEL VISTA - TAILORED DESTINATIONS DISCOVERY")
    st.sidebar.title("Navigation")

    # Navigation bar
    page = st.sidebar.selectbox("Go to", ["Home", "Attraction Finder", "Budget Prediction"])

    if page == "Home":
        st.subheader("Travel Destination Discovery")
        st.write("This app helps you find top attractions and predict travel budgets.")

    elif page == "Attraction Finder":
        st.subheader("Attraction Finder")
        # Load your DataFrame
        df = pd.read_csv("Maindata.csv")

        # Get unique values for state and category
        unique_states = df['State'].unique()
        unique_categories = df['Category'].unique()

        # User inputs
        state_input = st.selectbox('Select State:', unique_states)
        category_input = st.selectbox('Select Category:', unique_categories)

        # Filter the DataFrame based on user input for state and category
        filtered_df = df[(df['State'] == state_input) & (df['Category'] == category_input)]

        # Sort the filtered DataFrame by 'Sentiment_Score' in ascending order
        sorted_df = filtered_df.sort_values(by='Sentiment_Score', ascending=True)

        # Take the top 10 rows from the sorted DataFrame
        top_10_places = sorted_df.head(10)

        # Display the attraction names
        if not top_10_places.empty:
            st.subheader('Top 10 Attractions:')
            for index, row in top_10_places.iterrows():
                st.write(row['Attraction Name'])
        else:
            st.write('No attractions found for the selected state and category.')

    elif page == "Budget Prediction":
        st.subheader("Budget Prediction")
        # User inputs
        state_input = st.selectbox("Select State", states)
        category_input = st.selectbox("Select Category", categories)

        # Predict budget
        min_budget_pred, max_budget_pred = predict_budget(state_input, category_input)

        # Display results
        st.write(f"Predicted Minimum Budget: ${min_budget_pred:.2f}")
        st.write(f"Predicted Maximum Budget: ${max_budget_pred:.2f}")

if __name__ == "__main__":
    main()


