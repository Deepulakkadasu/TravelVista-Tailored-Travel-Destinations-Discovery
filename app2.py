import streamlit as st
import pandas as pd
import joblib

# Load the main DataFrame
data = pd.read_csv("Maindata.csv")

# Load the saved models
model_min = joblib.load('new_model_min.pkl')
model_max = joblib.load('new_model_max.pkl')

# Function to get available categories based on the selected state
def get_available_categories(state_input):
    available_categories = data[data['State'] == state_input]['Category'].unique()
    return available_categories
def predict_budget(state_input, category_input):
    # Get the list of feature names used during model training
    model_features = model_min.feature_names_in_

    # Prepare the input data
    all_categories = data['Category'].unique()
    all_states = data['State'].unique()
    X_pred = pd.DataFrame(0, index=[0], columns=model_features)

    # Set the selected category and state columns to 1
    if 'Category_' + category_input in X_pred.columns:
        X_pred['Category_' + category_input] = 1
    if 'State_' + state_input in X_pred.columns:
        X_pred['State_' + state_input] = 1

    # Fill any missing columns with zeros
    missing_columns = set(X_pred.columns) - set(model_features)
    for col in missing_columns:
        X_pred[col] = 0

    # Predict budget
    min_budget_pred = model_min.predict(X_pred)[0]
    max_budget_pred = model_max.predict(X_pred)[0]
    return min_budget_pred, max_budget_pred


# Streamlit web application
def main():
    st.title("Travel Vista")

    # Navigation bar
    page = st.sidebar.selectbox("Go to", ["Home", "Attraction Finder", "Budget Prediction", "Place Budgets"])

    if page == "Home":
        st.subheader("Travel Destination Discovery")
        st.write("This app helps you find top attractions and predict travel budgets.")

    elif page == "Attraction Finder":
        st.subheader("Attraction Finder")
        # Load your DataFrame
        df = pd.read_csv("Maindata.csv")

        # User inputs
        state_input = st.selectbox('Select State:', data['State'].unique())

        # Get available categories for the selected state
        available_categories = get_available_categories(state_input)
        category_input = st.selectbox('Select Category:', available_categories)

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
        state_input = st.selectbox("Select State", data['State'].unique())
        
        # Get available categories for the selected state
        available_categories = get_available_categories(state_input)
        category_input = st.selectbox("Select Category", available_categories)

        # Predict budget
        min_budget_pred, max_budget_pred = predict_budget(state_input, category_input)

        # Display results
        st.write(f"Predicted Minimum Budget: {min_budget_pred:.2f}")
        st.write(f"Predicted Maximum Budget: {max_budget_pred:.2f}")


    elif page == "Place Budgets":
        st.subheader("Place Budgets")
        # User inputs
        state_input = st.selectbox("Select State", data['State'].unique())
        categories_in_state = get_available_categories(state_input)
        category_input = st.selectbox("Select Category", categories_in_state)

        # Filter the DataFrame based on user input for state and category
        filtered_df = data[(data['State'] == state_input) & (data['Category'] == category_input)]

        # Display the places along with their maximum and minimum budgets
        if not filtered_df.empty:
            st.write(filtered_df[['Attraction Name', 'Min_Budget', 'Max_Budget']])
        else:
            st.write('No places found for the selected state and category.')

if __name__ == "__main__":
    main()
