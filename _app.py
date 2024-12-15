"""
    This code utilizes Streamlit for creating a user interface with sections, titles, 
    and data visualizations to explore the relationship between days listed and vehicle 
    price using the data set, 'vehicles_us.csv'. The variables date_posted and model 
    are not used in this analysis.
"""

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import altair as alt
import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt
from d_manager import DataCleaner
import statsmodels.api as sm

st.title('Understanding Demand and Listing Duration')
st.write('This code utilizes Streamlit for creating a user interface with sections, titles, and data visualizations, to explore the relationship between days listed and vehicle price using the data set, "vehicles_us.csv". The variables date_posted and model are not used in this analysis.')

# Load the Vehicle data for processing
df = pd.read_csv('vehicles_us.csv')

# Drop the unwanted columns
columns_to_drop = ['date_posted', 'model']
df = df.drop(columns=columns_to_drop)

# Fill missing values
df.fillna(method='ffill', inplace=True)

# Create a DataCleaner object
cleaner = DataCleaner(df)

# Clean all columns in the DataFrame
cleaner.clean_data(df.columns)

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
    
# Regression Model Section
tabs = st.tabs(["Variable Selection", "Model Results"])
df['price'] = pd.to_numeric(df['price'], errors='coerce') # Handle potential errors during conversion
df['odometer'] = pd.to_numeric(df['odometer'], errors='coerce') # Handle potential errors during conversion
st.text(df.info()) 
st.write(df)
st.markdown(f"""
    **Data Information:**
    ```
    {str(df.info())}
    ```
    """)

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
        y = df['price']
        
        # 1. Handle potential non-numeric values in selected independent variables
        for col in data.columns:
            try:    
                data[col] = pd.to_numeric(data[col], errors='coerce') 
            except ValueError:
                pass  # If conversion fails, it's likely a categorical variable

        # 2. Handle categorical variables (e.g., using one-hot encoding)
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

        # Fit the model
        model = sm.OLS(y, X).fit()

        # Display results
        if model:
            st.write(model.summary())
        else:
            st.write("Model fitting failed.")
    else:
        st.write("Please select variables in the 'Variable Selection' tab.")