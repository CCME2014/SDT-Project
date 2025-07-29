import pandas as pd
import plotly.express as px
import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt
import statsmodels.api as sm
from d_manager import DataCleaner, prepare_data

st.title('Understanding Demand and Listing Duration')
st.write('This code utilizes Streamlit for creating a user interface with sections, titles, and data visualizations, to explore the relationship between days listed and vehicle price using the data set, "vehicles_us.csv". The variables date_posted and model are not used in this analysis.')

# Load the Vehicle data for processing
df = pd.read_csv('vehicles_us.csv')

# Drop unwanted columns
columns_to_drop = ['date_posted', 'model']
df = df.drop(columns=columns_to_drop)

# Fill missing values
df.fillna(method='ffill', inplace=True)

# Create a section displaying descriptive statistics
st.header('Distributions of Variables')

# Create a figure with two subplots
fig1, (ax1, ax2) = plt.subplots(nrows=2, ncols=1, figsize=(8, 6))

# Plot the histogram for the independent variable (days_listed)
sns.histplot(df['days_listed'], kde=True, ax=ax1)
ax1.set_title('Histogram and Boxplot of Days Listed')

# Plot the boxplot for the independent variable (days_listed)
sns.boxplot(x=df['days_listed'], ax=ax2)
plt.subplots_adjust(bottom=0.1)
st.pyplot(fig1)

# Calculate the correlation between days listed and price
dp_correlation = df['price'].corr(df['days_listed'])
st.write(f"Correlation between price and days listed: {dp_correlation:.2f}")

# Section for analyzing relationships and interactions
st.header('Exploring Interactions and Relationships')

# Bar chart for days listed by a selected variable
option = st.selectbox("Select a variable:", 
                      ('model_year', 'is_4wd', 'cylinders', 'condition', 'fuel', 'transmission', 'type', 'paint_color'))

fig2, ax = plt.subplots()
df[option] = df[option].astype('category')  # Ensure 'option' is a categorical variable
sns.barplot(x=option, y='days_listed', data=df, ax=ax)
ax.set_title(f'Days Listed by {option}')
st.pyplot(fig2)

# Regression Model Section
tabs = st.tabs(["Variable Selection", "Model Results"])

with tabs[0]:  # "Variable Selection"
    include_interactions = st.checkbox("Include Interaction Terms?", value=False)

    selected_independent_variables = st.multiselect(
        'Select Independent Variables for the OLS Regression Model:', 
        df.columns, default='days_listed')

    if include_interactions and len(selected_independent_variables) >= 2:
        selected_interaction_variables = st.multiselect(
            'Select 2 Variables for Interaction Analysis (Optional):', 
            selected_independent_variables, max_selections=2)
    else:
        selected_interaction_variables = []

with tabs[1]:  # "Model Results"
    if selected_independent_variables:
        # Prepare data for regression
        data = df[selected_independent_variables].copy()

        # Handle categorical variables (e.g., using one-hot encoding)
        categorical_cols = data.select_dtypes(include=['object', 'category']).columns

        if len(categorical_cols) > 0:
            data = pd.get_dummies(data, columns=categorical_cols, drop_first=True)

        # Add interaction terms if specified
        if selected_interaction_variables:
            interaction_term = "*".join(selected_interaction_variables)
            data[interaction_term] = (
                df[selected_interaction_variables[0]].astype(float) * 
                df[selected_interaction_variables[1]].astype(float)
            )

        # Add constant term
        X = sm.add_constant(data, has_constant='add')
        y = df['price']

        # Fit the model
        model = sm.OLS(y, X).fit()

        # Display results
        if model:
            st.text(model.summary().as_text())
        else:
            st.write("Model fitting failed.")
    else:
        st.write("Please select variables in the 'Variable Selection' tab.")