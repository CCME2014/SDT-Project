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
from .d_manager import DataCleaner, prepare_data
import statsmodels.api as sm

st.title('Understanding Demand and Listing Duration')
st.write('This code utilizes Streamlit for creating a user interface with sections, titles, and data visualizations to explore the relationship between days listed and vehicle price using the data set, 'vehicles_us.csv'. The variables date_posted and model are not used in this analysis.')

# Load the Vehicle data for processing
df = pd.read_csv('vehicles_us.csv')
df = DataCleaner(df)

# Create the collections of data frame variables
complete_list = ['model_year','is_4wd','cylinders','condition','fuel','transmission','type', 'paint_color', 'days_listed', 'price']
independent_variables = ['model_year','is_4wd','cylinders','condition','fuel','transmission','type', 'paint_color', 'days_listed'] 
variable_choice = ['model_year','is_4wd','cylinders','condition','fuel','transmission','type', 'paint_color']

# Create the data frames
df_price = df.clean_data('price')
df_days = df.clean_data('days_listed')
df_dp = df.clean_data(['days_listed', 'price'])
df_all = df.clean_data(complete_list)
df_date_listed_odometer = df.clean_data(['odometer', 'days_listed'])
df_variables = df.clean_data(variable_choice)

# Create a section displaying descriptive statistics about the independent variable (days_listed) and the dependent variable (price)
st.header('Distributions of Variables')

# Create a histogram of the dependent variable (price)
histo_price = df_price.create_visualization('price', kind='histogram', title='Histogram of Vehicle Price')
st.write(histo_price)

# Create a figure with two subplots
fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1, figsize=(8, 6))

# Plot the histogram for the independent variable (days_listed)
sns.histplot(df_days['days_listed'], kde=True, ax=ax1)
ax1.set_title('Histogram of Days Listed')

# Plot the boxplot for the independent variable (days_listed)
sns.boxplot(x=df_days['days_listed'], ax=ax2)
ax2.set_title('Boxplot of Days Listed')
st.pyplot(fig)

# Create a section displaying bivariate scatter plots 
st.header('Exploring Price Trends')

# Create the bar graph for the price mean by days listed
bar_variables = df_dp.prepare_data_for_visualization('days_listed', 'price').create_visualization(x_col='days_listed', y_col='price_mean', 
                 error_y='price_std', kind='scatter', title='Average Price by Days Listed with Standard Deviation and Point Size Representing Price Data Quanitity',
                 size='marker_size')
st.write(bar_variables)

# Calculate the correlation between days listed and price
dp_correlation = df_dp['price'].corr(df_dp['days_listed'])
st.write(f"Correlation between price and days listed: {dp_correlation:.2f}")

# Create a section for analyzing the relationships and interactions
st.header('Exploring Interactions and Relationships')
col1, col2 = st.columns(1,3) 

# Create the scatter plot between days listed and the odometer
with col1:
    scatter_date_listed_odometer = df_date_listed_odometer.create_visualization(x_col='days_listed', y_col='odometer', 
                kind='scatter', title='Days Listed vs. Odometer')

# Create the bar chart for days listed and the selected variable
with col2:
    option = st.selectbox("Select a variable:",
                          ('model_year','is_4wd','cylinders','condition','fuel','transmission','type', 'paint_color'),
                          )
    fig, ax = plt.subplots()
    sns.barplot(x=option, y='days_listed', data=df_variables, ax=ax)
    ax.set_title(f'Days Listed by {option}')    
    st.pyplot(fig)
    
# Create streamlit tabs for selecting the variables and displaying the regression model
with st.tabs():
    with st.tab("Variable Selection"):
        # Include a checkbox
        include_interactions = st.checkbox("Include Interaction Terms?", value=False)

        # Select independent variables
        selected_independent_variables = st.multiselect('Select Independent Variables for the OLS Regression Model:', 
                                                        independent_variables, default='days_listed')
        
        # Enable interaction selection only if at least 2 independent variables are selected
        if include_interactions and len(selected_independent_variables) >= 2:       
           selected_interaction_variables = st.multiselect('Select 2 Variables for an Interaction Analysis (Optional):', independent_variables,
                                                           max_selections=2)
        else:
            selected_interaction_variables = []  # Reset interaction selection if less than 2 independent variables chosen
  
        # Warning message
        if len(selected_interaction_variables) > 2:
            st.warning("Please select a maximum of 2 options for interaction analysis.")

    with st.tab("Model Results"):
        # Prepare data based on selections
        if selected_independent_variables:
            data = prepare_data(df_all.copy(), selected_independent_variables, selected_interaction_variables)
            
            # Add constant term
            X = sm.add_constant(data)
            y = df_all['price']
            
            # Fit the model
            model = sm.OLS(y, X).fit()
                
            # Display results within Streamlit
            if model:
                st.write(model.summary()) # Display model summary
            else:
                st.write("Model fitting failed.")
        else:
            st.write("Please select your variables in the 'Variable Selection' tab.")
