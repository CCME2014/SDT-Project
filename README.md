Understanding Demand and Listing Duration

This project utilizes Streamlit to create an interactive data exploration tool for understanding the relationship between days listed and vehicle price using the dataset vehicles_us.csv. The variables date_posted and model are not included in this analysis.

Getting Started

Ensure you have Python and the required libraries installed (pandas, streamlit, seaborn, plotly.express, matplotlib, statsmodels). You can install them using pip install pandas streamlit seaborn plotly.express matplotlib statsmodels.
Clone or download this repository.
Run the application using streamlit run app.py.

Code Breakdown

app.py: This file contains the core logic for the Streamlit application. It loads the data, performs cleaning and visualizations, and creates interactive elements to explore the relationship between days listed and price

d_manager.py: This file defines a class DataCleaner for data cleaning and preparation tasks. It includes methods for cleaning specific columns, calculating summary statistics, and creating visualizations.

requirements.txt: This file specifies the required Python libraries for running the application.

Data Exploration
The application provides different sections for exploring the data:

Distributions of Variables: This section displays histograms and boxplots of the day listed and price.

Exploring Price Trends: This section shows the average price by days listed with standard deviation and a scatter plot of days listed vs. price. The correlation between price and days listed is also displayed.

Exploring Interactions and Relationships: This section allows you to analyze the relationship between days listed and other vehicle attributes. You can create scatter plots and bar charts to explore various variables and their interactions.

Model Results: This section provides a tab for selecting independent variables, including an optional interaction term. It then attempts to fit an Ordinary Least Squares (OLS) regression model and display the results.
