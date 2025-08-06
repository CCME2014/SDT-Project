"""
    This code utilizes Streamlit for creating a user interface with sections, titles, 
    and data visualizations to explore the relationship between days listed and vehicle 
    price using the data set, 'vehicles_us.csv'. The variables date_posted and model 
    are not used in this analysis.
"""

import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import altair as alt
import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt
from d_manager import DataCleaner
import statsmodels.api as sm
import itertools

st.title('Understanding Demand and Listing Duration')
st.write('This code utilizes Streamlit for creating a user interface with sections, titles, and data visualizations, to explore the relationship between days listed and vehicle price using the data set, "vehicles_us.csv". The variables date_posted and model are not used in this analysis.')

# Load the Vehicle data for processing
df = pd.read_csv('vehicles_us.csv')
go_category = ['model_year', 'cylinders', 'is_4wd', 'condition', 'fuel', 'transmission', 'type', 'paint_color']

# Drop the unwanted columns
columns_to_drop = ['date_posted', 'model']
df = df.drop(columns=columns_to_drop)

# Create a DataCleaner object
cleaner = DataCleaner(df, categories=go_category)

# Create a section displaying descriptive statistics about the independent variable (days_listed) and the dependent variable (price)
st.header('Distributions of Variables')

# Create a histogram of the dependent variable (price)
histo_price = cleaner.create_visualization('price', kind='histogram', title='Histogram of Vehicle Price')
st.write(histo_price)

# Create the figure with two subplots
fig = make_subplots(rows=2, cols=1, subplot_titles=("Histogram of Days Listed", "Boxplot of Days Listed"))

# Add the histogram to the first subplot
fig.add_trace(
    go.Histogram(x=df['days_listed'], histnorm='probability density'),
    row=1, col=1
)

# Add the boxplot to the second subplot
fig.add_trace(
    go.Box(x=df['days_listed']),
    row=2, col=1
)

# Customize the layout (optional)
fig.update_layout(height=600, width=800)

# Display the figure in Streamlit
st.plotly_chart(fig)

# Create a section displaying bivariate scatter plots 
st.header('Exploring Price Trends')

# Create the bar graph for the price mean by days listed
bar_variables = cleaner.prepare_data_for_visualization('days_listed', 'price')

# Ensure 'marker_size' is a standard float type (float64) and handle any potential NaNs
# The .fillna(0) is a safeguard, though unlikely to be needed for a count-derived column.
bar_variables['marker_size'] = bar_variables['marker_size'].astype(float).fillna(0)

fig2 = px.scatter(bar_variables, x='days_listed', y='price_mean', error_y='price_std',
                 title='Average Price by Days Listed with Standard Deviation and Point Size Representing Price Data Quantity',
                 size='marker_size')
st.plotly_chart(fig2)

# Calculate the correlation between days listed and price
dp_correlation = df['price'].corr(df['days_listed'])
st.write(f"Correlation between price and days listed: {dp_correlation:.2f}")

# Create a section for analyzing the relationships and interactions
st.header('Exploring Interactions and Relationships')

# Create the scatter plot between days listed and the odometer
scatter_date_listed_odometer = cleaner.create_visualization(x_col='days_listed', y_col='odometer', 
                kind='scatter', title='Days Listed vs. Odometer')
st.write(scatter_date_listed_odometer)

# Create the bar chart for days listed and the selected variable
option = st.selectbox("Select a variable:",
                          ('model_year','is_4wd','cylinders','condition','fuel','transmission','type', 'paint_color'),
                          )
# df.fillna(method='ffill', inplace=True)  # Replace missing values with the previous value
df[option] = df[option].astype('category') # Ensure 'option' is a categorical variable
fig3 = px.bar(df, x=option, y='days_listed', title=f'Days Listed by {option}')
st.plotly_chart(fig3)
    

###### Debugging BEGINNING ########
st.dataframe(df)

# Function to capture df.info() output
import io
import sys
def capture_df_info(dataframe):
    buffer = io.StringIO()  # Create a buffer to capture the output
    sys.stdout = buffer  # Redirect stdout to the buffer
    dataframe.info()  # This will write to the buffer instead of the console
    sys.stdout = sys.__stdout__  # Reset stdout to the original
    return buffer.getvalue()  # Return the content of the buffer

# Capture and display the formatted info
df_info_str = capture_df_info(df)

# Display the info in the Streamlit app using st.code() for better formatting
st.code(df_info_str, language='text')

###### Debugging END ########


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
        y = df['price'].copy()
        y.fillna(y.mean(), inplace=True)
    
        categorical_cols = data.select_dtypes(include=['object', 'category']).columns
        if len(categorical_cols) > 0:
            data = pd.get_dummies(data, columns=categorical_cols, drop_first=True)
    
        if selected_interaction_variables and len(selected_interaction_variables) == 2:
            var1_base = selected_interaction_variables[0]
            var2_base = selected_interaction_variables[1]
    
            cols_for_var1 = [col for col in data.columns if col == var1_base or col.startswith(f"{var1_base}_")]
            cols_for_var2 = [col for col in data.columns if col == var2_base or col.startswith(f"{var2_base}_")]
    
            for c1 in cols_for_var1:
                for c2 in cols_for_var2:
                    if c1 == c2:
                        continue
                    interaction_name = f"{c1}_x_{c2}"
                    data[interaction_name] = data[c1] * data[c2]
        elif selected_interaction_variables and len(selected_interaction_variables) != 2:
            st.warning("Please select exactly two variables for interaction analysis.")

 # --- FINAL, AGGRESSIVE CONVERSION TO NUMPY FLOAT ARRAYS ---
        # This is the most robust way to ensure statsmodels gets pure numeric data.
        try:
            X_final = data.astype(float).replace([np.inf, -np.inf], np.nan).fillna(0.0).to_numpy()
            y_final = y.astype(float).replace([np.inf, -np.inf], np.nan).fillna(0.0).to_numpy() # Ensure y is also clean
        except Exception as e:
            st.error(f"Error during final data conversion for OLS: {e}")
            st.stop()

        # Add constant term to the NumPy array
        X_final = sm.add_constant(X_final, has_constant='add')

        # Fit the model
        model = sm.OLS(y_final, X_final).fit()
        
        # Display results
        if model:
            st.write(model.summary())
        else:
            st.write("Model fitting failed.")
    else:
        st.write("Please select variables in the 'Variable Selection' tab.")