import pandas as pd
import numpy as np
import plotly.express as px

# Create a class for cleaning and preparing data for analysis and visualization
class DataCleaner:
    """
    A class for cleaning and preparing data for analysis and visualization.

    Attributes:
        data (pandas.DataFrame): The DataFrame containing the data.

    Methods:
        __init__(self, data): Initializes the class with the given data.
        clean_data(self, column_names): Performs data cleaning based on the specified column names.
        tukey_five_number_summary(self, column): Calculates the Tukey 5-number summary for a specified column.
        prepare_data_for_visualization(self, group_col, value_col, scaling_factor=50): Prepares data for visualization by calculating mean, standard deviation, and count for a given value column grouped by a specified column.
        create_visualization(self, x_col, y_col=None, error_y=None, kind='histogram', title='', marker_size_col=None, height=500): Creates a histogram, scatter plot, or bar chart based on the specified parameters.
    """

    def __init__(self, data):
        self.data = data

    def clean_data(self, column_names):
        """
        Performs data cleaning based on the specified column names.

        Args:
            column_names (list): A list of column names to clean.
        """

        for column in column_names:
            if column == 'is_4wd':
                self.data[column] = self.data[column].fillna(0)
            elif column in ['date_posted']:
                self.data[column] = pd.to_datetime(self.data[column])
            elif column in ['model_year', 'cylinders', 'is_4wd', 'odometer']:
                self.data[column] = self.data[column].astype('Int64')
                if column in ['model_year', 'cylinders', 'is_4wd']:
                    self.data[column] = self.data[column].astype('object')

        return self.data

    def tukey_five_number_summary(self, column):
        """
        Calculates the Tukey 5-number summary for a specified column.

        Args:
            column (str): The name of the column to analyze.

        Returns:
            dict: A dictionary containing the minimum, Q1, median, Q3, and maximum values.
        """

        q1, q3 = np.percentile(self.data[column], [25, 75])
        iqr = q3 - q1
        lower_fence = q1 - 1.5 * iqr
        upper_fence = q3 + 1.5 * iqr

        return {
            'minimum': self.data[column].min(),
            'q1': q1,
            'median': self.data[column].median(),
            'q3': q3,
            'maximum': self.data[column].max(),
            'lower_fence': lower_fence,
            'upper_fence': upper_fence
        }

    def prepare_data_for_visualization(self, group_col, value_col, scaling_factor=50):
        """
        Prepares data for visualization by calculating mean, standard deviation, and count
        for a given value column grouped by a specified column.

        Args:
            group_col (str): The column to group the data by.
            value_col (str): The column to calculate statistics on.
            scaling_factor (int, optional): The factor to scale the marker size by. Default is 50.

        Returns:
            pandas.DataFrame: A DataFrame with calculated mean, standard deviation, count, and marker size columns.
        """

        # Calculate statistics
        mean_df = self.data.groupby(group_col)[value_col].mean().reset_index()
        std_df = self.data.groupby(group_col)[value_col].std().reset_index()
        count_df = self.data.groupby(group_col)[value_col].count().reset_index()

        # Merge dataframes
        merged_df = mean_df.merge(std_df, on=group_col, how='left')
        merged_df = merged_df.merge(count_df, on=group_col, how='left')
        merged_df = merged_df.rename(columns={
            value_col + '_x': value_col + '_mean',
            value_col + '_y': value_col + '_std',
            value_col: value_col + '_count'
        })

        # Calculate marker size
        merged_df['marker_size'] = merged_df[value_col + '_count'] * scaling_factor

        return merged_df

    def create_visualization(self, x_col, y_col=None, error_y=None, kind='histogram', title='', marker_size_col=None, height=500):
        """
        Creates a histogram, scatter plot, or bar chart based on the specified parameters.

        Args:
            x_col (str): The column to use for the x-axis.
            y_col (str, optional): The column to use for the y-axis (scatter plot only). Defaults to None.
            error_y (str, optional): The column to use for error bars (scatter plot only). Defaults to None.
            kind (str, optional): The type of visualization to create (histogram, scatter, bar). Defaults to 'histogram'.
            title (str, optional): The title of the visualization. Defaults to ''.
            marker_size_col (str, optional): The column to use for marker size (scatter plot only). Defaults to None.
            height (int, optional): The height of the visualization (in pixels). Defaults to 500.

        Returns:
            plotly.graph_objects.Figure: The generated visualization.
        """

        if kind == 'histogram':
            fig = px.histogram(self.data, x=x_col)
        elif kind == 'scatter':
            fig = px.scatter(self.data, x=x_col, y=y_col, error_y=error_y, title=title, size=marker_size_col)
        elif kind == 'bar':
            if y_col is None:
                y_col = x_col  # If y_col is not provided, use x_col for bar charts
            # Calculate the mean for the specified column
            mean_df = self.data.groupby(x_col)[y_col].mean().reset_index()

            # Use the mean dataframe for plotting the bar chart
            fig = px.bar(mean_df, x=x_col, y=y_col, title=title, height=height)
        return fig

# Create a function for preparing variables to fit a OLS Regression Model
def prepare_data(data, selected_variables, interaction_variables):
    if selected_variables:
        data = data[selected_variables]
        if interaction_variables:
            combined_interaction = "*".join(interaction_variables)
            selected_model_variables = selected_variables + [combined_interaction]
            data = data[selected_model_variables]
    return data
