"""
    This code utilizes Streamlit for creating a user interface with sections, titles, 
    and data visualizations to explore the relationship between days listed and vehicle 
    price using the data set, 'vehicles_us.csv'. The variables date_posted and model 
    are not used in this analysis.
"""

import pandas as pd
import plotly.express as px
import altair as alt
import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt
from d_manager import DataCleaner, prepare_data
import statsmodels.api as sm

st.title('Understanding Demand and Listing Duration')
st.write('This code utilizes Streamlit for creating a user interface with sections, titles, and data visualizations, to explore the relationship between days listed and vehicle price using the data set, "vehicles_us.csv". The variables date_posted and model are not used in this analysis.')

# Load the Vehicle data for processing
df = pd.read_csv('vehicles_us.csv')

# Drop the unwanted columns
columns_to_drop = ['date_posted', 'model']
df = df.drop(columns=columns_to_drop)

# Create a DataCleaner object
cleaner = DataCleaner(df)

# Clean all columns in the DataFrame
cleaner.clean_data(df.columns)

# Create a section displaying descriptive statistics about the independent variable (days_listed) and the dependent variable (price)
st.header('Distributions of Variables')

# Create a histogram of the dependent variable (price)
histo_price = cleaner.create_visualization('price', kind='histogram', title='Histogram of Vehicle Price')
st.write(histo_price)

# Create a figure with two subplots
fig1, (ax1, ax2) = plt.subplots(nrows=2, ncols=1, figsize=(8, 6))

# Plot the histogram for the independent variable (days_listed)
sns.histplot(df['days_listed'], kde=True, ax=ax1)
ax1.set_title('Histogram and Boxplot of Days Listed')
ax1.set_xlabel(None)

# Plot the boxplot for the independent variable (days_listed)
sns.boxplot(x=df['days_listed'], ax=ax2)
# ax2.set_title('Boxplot of Days Listed')
plt.subplots_adjust(bottom=0.1)  # Adjust spacing between subplots
st.pyplot(fig1)

# Create a section displaying bivariate scatter plots 
st.header('Exploring Price Trends')

# Create the bar graph for the price mean by days listed
bar_variables = cleaner.prepare_data_for_visualization('days_listed', 'price')
fig2 = px.scatter(bar_variables, x='days_listed', y='price_mean', error_y='price_std',
                 title='Average Price by Days Listed with Standard Deviation and Point Size Representing Price Data Quantity',
                 size='marker_size')
st.plotly_chart(fig2)

# Calculate the correlation between days listed and price
dp_correlation = df['price'].corr(df['days_listed'])
st.write(f"Correlation between price and days listed: {dp_correlation:.2f}")

# Create a section for analyzing the relationships and interactions
st.header('Exploring Interactions and Relationships')
col1, col2 = st.columns([1, 3]) 

# Create the scatter plot between days listed and the odometer
scatter_date_listed_odometer = cleaner.create_visualization(x_col='days_listed', y_col='odometer', 
                kind='scatter', title='Days Listed vs. Odometer')
st.write(scatter_date_listed_odometer)

# Create the bar chart for days listed and the selected variable
option = st.selectbox("Select a variable:",
                          ('model_year','is_4wd','cylinders','condition','fuel','transmission','type', 'paint_color'),
                          )
fig3, ax = plt.subplots()
df.fillna(method='ffill', inplace=True)  # Replace missing values with the previous value
df[option] = df[option].astype('category') # Ensure 'option' is a categorical variable
sns.barplot(x=option, y='days_listed', data=df, ax=ax)
ax.set_title(f'Days Listed by {option}')    
st.pyplot(fig3)
    
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
            st.write(model.summary())
        else:
            st.write("Model fitting failed.")
    else:
        st.write("Please select variables in the 'Variable Selection' tab.")

# with tabs[1]:  # "Model Results"
#     # # Prepare data based on selections
#     # if selected_independent_variables:
#     #     data = prepare_data(df.copy(), selected_independent_variables, selected_interaction_variables)
        
#     #     # Add constant term
#     #     X = sm.add_constant(data)
#     #     y = df['price']
        
#     #     # Fit the model
#     #     model = sm.OLS(y, X).fit()
            
#     #     # Display results within Streamlit
#     #     if model:
#     #         st.write(model.summary()) # Display model summary
#     #     else:
#     #         st.write("Model fitting failed.")
#     # else:
#     #     st.write("Please select your variables in the 'Variable Selection' tab.")
        
#     if selected_independent_variables:
#         # Prepare data, handling categorical variables
#         data = prepare_data(df.copy(), selected_independent_variables, selected_interaction_variables)
    
#         # Handle categorical variables (e.g., using one-hot encoding)
#         categorical_cols = data.select_dtypes(include=['object']).columns
#         data = pd.get_dummies(data, columns=categorical_cols)
    
#         # Add constant term
#         X = sm.add_constant(data)
#         y = df['price']
    
#         # Fit the model
#         model = sm.OLS(y, X).fit()
    
#         # Display results within Streamlit
#         if model:
#             st.write(model.summary())
#         else:
#             st.write("Model fitting failed.")
#     else:
#         st.write("Please select your variables in the 'Variable Selection' tab.")
