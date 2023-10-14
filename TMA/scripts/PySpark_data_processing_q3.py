import logging
import os
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, sum, avg, min, max, when

# Constants
SCRIPTS_DIR = os.path.abspath(os.path.dirname(__file__))
DATA_DIR = os.path.join(SCRIPTS_DIR, "..", "data")
FLIGHTS_DATA_FILE_PATH = os.path.join(DATA_DIR, "flights_data_v2.csv")
PLANES_DATA_FILE_PATH = os.path.join(DATA_DIR, "planes_data_v2.csv")
LOGGING_LEVEL = logging.INFO
LOAD_DATA_ERROR_MESSAGE = "An error occurred while loading data: {}"
FILE_NOT_FOUND_MESSAGE = "The specified file does not exist: {}"


def configure_logging():
    """Configure logging settings and return a logger object.

    Returns
    -------
    logger : object
        Logger object for logging messages.

    Notes
    -----
    This function returns a logger object that may be used to log messages
    inside the program and initializes the logging parameters, including the logging level.

    The default logging level is set to the value of the constant LOGGING_LEVEL.
    """
    logging.basicConfig(level=LOGGING_LEVEL)
    return logging.getLogger(__name__)


def create_spark_session(app_name="TMA_Data_Analysis"):
    """
    Create and return a Spark session.

    Parameters
    ----------
    app_name : str, optional
        The name of the Spark application, by default "TMA_Data_Analysis".

    Returns
    -------
    SparkSession
        The Spark session object.

    Notes
    -----
    This function initializes a Spark session, which is the entry point for working with Spark functionality.
    """
    return SparkSession.builder.appName(app_name)\
        .config("spark.some.config.option", "some-value")\
        .getOrCreate()


def show_dataframe(df, max_rows=100, show_rows=20):
    """
    Show rows of a DataFrame with the option to limit the number of rows displayed.

    Parameters
    ----------
    df : DataFrame
        The DataFrame to be displayed.

    max_rows : int, optional
        The maximum number of rows to display. Default is 20.

    show_rows : int, optional
        The rows to display if records is above max rows. 

    Returns
    -------
    None

    Notes
    -----
    This function shows the rows from the DataFrame input.
    The DataFrame will only display the first "show_rows" rows if there are more rows than the specified "max_rows".
    The DataFrame will display all available rows without truncation if the number of rows is less than "max_rows".
    """
    if df.count() > max_rows:
        df.show(show_rows)
    else:
        df.show(df.count(), truncate=False)


def load_data(spark, logger, file_path):
    """
    Load data from CSV file into a Spark DataFrame.

    Parameters
    ----------
    spark : SparkSession
        The Spark session.
    logger : Logger
        Logger object for logging messages.
    file_path : str, optional
        The path to the CSV file to load.

    Returns
    -------
    DataFrame
        DataFrame containing the data.

    Raises
    ------
    FileNotFoundError
        If the specified file does not exist.

    Notes
    -----
    This function reads data from a CSV file and loads it into a Spark DataFrame.

    Note: The default logging level is set to the value of the constant LOGGING_LEVEL.
    """
    try:
        # Load data from csv
        df = spark.read.option("inferSchema", "true").option(
            "header", "true").csv(file_path)
        return df
    except Exception as e:
        if "Path does not exist" in str(e):
            logger.error(FILE_NOT_FOUND_MESSAGE.format(file_path))
            raise FileNotFoundError(f"File not found: {file_path}")

        logger.error(LOAD_DATA_ERROR_MESSAGE.format(str(e)))
        raise e


def process_missing_data(df, logger):
    """Process and analyze the loaded data.

    Parameters
    ----------
    df : DataFrame
        DataFrame containing the data.
    logger : object
        Logger object for logging messages.

    Returns
    -------
    DataFrame
        DataFrame containing the data.

    Raises
    ------
    Exception
        If an error occurs during data processing.

    Notes
    -----
    This function determines which rows in the supplied DataFrame have missing values,
    displays those rows, and removes them from the DataFrame. It details the total number of missing values,
    the cleaned DataFrame that is produced, and any problems that may have occurred.
    """
    try:
        columns_to_check = df.columns
        filter_condition = None

        # Loop through the list of columns and build a filter condition to check for null values.
        for column_name in columns_to_check:
            # Create a new condition to check if the column is null
            if filter_condition is None:
                filter_condition = col(column_name).isNull()
            # For subsequent columns, update the filter condition to include a check for null values
            else:
                filter_condition = filter_condition | col(column_name).isNull()

        missing_data_df = df.filter(filter_condition)

        logger.info("Sample rows in the DataFrame with Missing Value:")
        show_dataframe(missing_data_df)
        missing_occurrence = missing_data_df.count()
        logger.info(
            f"There are {missing_occurrence} rows with missing values in DataFrame.\n")

        clean_data_df = df.filter(~filter_condition)
        return clean_data_df
    except Exception as e:
        logger.error(f"An error occurred: {str(e)}")
        raise e


def count_by(df, grouped_columns, logger):
    """
    Count occurrences of rows by grouping columns.

    Parameters
    ----------
    df : DataFrame
        DataFrame containing the data.
    grouped_columns : list
        List of columns to group by.
    logger : object
        Logger object for logging messages.

    Returns
    -------
    DataFrame
        DataFrame with counts, sorted in descending order.

    Raises
    ------
    Exception
        If an error occurs during counting.

    Notes
    -----
    This function divides the data in the input DataFrame into groups according to the chosen columns,
    then counts the number of rows in each group. The outcome is a DataFrame with counts that is
    sorted in decreasing order according to the count values.
    """
    try:
        count_by_col = df.groupby(*grouped_columns).count()
        sorted_counts_df = count_by_col.orderBy("count", ascending=False)
        return sorted_counts_df
    except Exception as e:
        logger.error(f"An error occurred: {str(e)}")
        raise e


def percentage_by(df, grouped_columns, logger):
    """
    Calculate the percentage of occurrences by grouping columns.

    Parameters
    ----------
    df : DataFrame
        DataFrame containing the data.
    grouped_columns : list
        List of columns to group by.
    logger : object
        Logger object for logging messages.

    Returns
    -------
    DataFrame
        DataFrame with counts and percentages, sorted in descending order.

    Raises
    ------
    Exception
        If an error occurs during percentage calculation.

    Notes
    -----
    The function computes the number and percentage of occurrences within each group relative
    to the total number of rows in the DataFrame for the data in the input DataFrame, which is organized by the provided columns.
    The outcome is a DataFrame that is sorted in decreasing order using the percentage values and includes both counts and percentages.
    """
    try:
        total_flights = df.count()

        total_flights_by = df.groupby(*grouped_columns).count()
        col_percentage = total_flights_by.withColumn(
            "percentage", (total_flights_by["count"] / total_flights) * 100).orderBy("percentage", ascending=False)
        return col_percentage
    except Exception as e:
        logger.error(f"An error occurred: {str(e)}")
        raise e


def top_n_cat_by(df, grouped_columns, n, logger):
    """
    Find the top n category.

    Parameters
    ----------
    df : DataFrame
        DataFrame containing the data.
    grouped_columns : list
        List of columns to group by.
    n : int
        Number of top category to retrieve.
    logger : object
        Logger object for logging messages.

    Returns
    -------
    DataFrame
        DataFrame with the top category, sorted by count in descending order.

    Raises
    ------
    Exception
        If an error occurs during counting.

    Notes
    -----
    This function groups data from the specified DataFrame by the specified columns and counts the number of occurrences of each category.
    It returns a DataFrame containing the top n categories with the highest counts, sorted in descending order based on count values.
    """
    try:
        top_cat = df.groupBy(*grouped_columns).count().orderBy(
            "count", ascending=False).limit(n)
        return top_cat
    except Exception as e:
        logger.error(f"An error occurred: {str(e)}")
        raise e


def bottom_n_cat_by(df, grouped_columns, n, logger):
    """
    Find the bottom n category.

    Parameters
    ----------
    df : DataFrame
        DataFrame containing the data.
    grouped_columns : list
        List of columns to group by.
    n : int
        Number of bottom category to retrieve.
    logger : object
        Logger object for logging messages.

    Returns
    -------
    DataFrame
        DataFrame with the bottom category, sorted by count in ascending order.

    Raises
    ------
    Exception
        If an error occurs during counting.

    Notes
    -----
    This function groups data from the specified DataFrame by the specified columns and counts the number of occurrences of each category.
    It returns a DataFrame containing the last n categories with the highest number, sorted in ascending order based on count values.
    """
    try:
        bottom_cat = df.groupBy(*grouped_columns).count().orderBy(
            "count", ascending=True).limit(n)
        return bottom_cat
    except Exception as e:
        logger.error(f"An error occurred: {str(e)}")
        raise e


def sql_top_n_cat_by(df, grouped_columns, n, spark, logger):
    """
    Find the top n category.

    Parameters
    ----------
    df : DataFrame
        DataFrame containing the data.
    grouped_columns : list
        List of columns to group by.
    n : int
        Number of top category to retrieve.
    spark : SparkSession
        The Spark session.
    logger : object
        Logger object for logging messages.

    Returns
    -------
    DataFrame
        DataFrame with the top category, sorted by count in descending order.

    Raises
    ------
    Exception
        If an error occurs during counting.

    Notes
    -----
    This function groups data from the specified DataFrame by the specified columns and counts the number of occurrences of each category.
    It returns a DataFrame containing the top n categories with the highest counts, sorted in descending order based on count values.
    """
    try:
        df.createOrReplaceTempView("top_flights_planes")
        columns_str = ", ".join(grouped_columns)

        query = f"""
        SELECT {columns_str}, COUNT(*) AS count
        FROM top_flights_planes
        GROUP BY {columns_str}
        ORDER BY count DESC
        LIMIT {n}
        """

        sql_top_n_cat = spark.sql(query)
        return sql_top_n_cat
    except Exception as e:
        logger.error(f"An error occurred: {str(e)}")
        raise e


def sql_bottom_n_cat_by(df, grouped_columns, n, spark, logger):
    """
    Find the bottom n category.

    Parameters
    ----------
    df : DataFrame
        DataFrame containing the data.
    grouped_columns : list
        List of columns to group by.
    n : int
        Number of bottom category to retrieve.
    spark : SparkSession
        The Spark session.
    logger : object
        Logger object for logging messages.

    Returns
    -------
    DataFrame
        DataFrame with the bottom category, sorted by count in ascending order.

    Raises
    ------
    Exception
        If an error occurs during counting.

    Notes
    -----
    This function groups data from the specified DataFrame by the specified columns and counts the number of occurrences of each category.
    It returns a DataFrame containing the last n categories with the highest number, sorted in ascending order based on count values.
    """
    try:
        df.createOrReplaceTempView("bottom_flights_planes")
        columns_str = ", ".join(grouped_columns)

        query = f"""
        SELECT {columns_str}, COUNT(*) AS count
        FROM bottom_flights_planes
        GROUP BY {columns_str}
        ORDER BY count ASC
        LIMIT {n}
        """

        sql_bottom_n_cat = spark.sql(query)
        return sql_bottom_n_cat
    except Exception as e:
        logger.error(f"An error occurred: {str(e)}")
        raise e


def analyze_average_delay(df, column, delay_column, logger):
    """
    Analyze average departure/arrival delay.

    Parameters
    ----------
    df : DataFrame
        The DataFrame containing the data.
    column : str
        The column by which to group the data for analysis.
    delay_column : str
        The column representing departure/arrival delay.
    logger : object
        Logger object for logging messages.

    Returns
    -------
    DataFrame
        DataFrame containing the average departure/arrival delay.

    Raises
    ------
    Exception
        If an error occurs during the computation.

    Notes
    -----
    This function calculates the average departure/arrival delay for a specified column,
    groups the data by another column,and ranks the results in descending order based on the average delay.
    """
    try:
        suffix = delay_column.split('_')[0]
        new_column_name = f"average_{suffix}_delay"

        avg_departure_delay_by_column = df.groupBy(column).agg(avg(col(delay_column)).alias(
            new_column_name)).orderBy(new_column_name, ascending=False)
        return avg_departure_delay_by_column
    except Exception as e:
        logger.error(f"An error occurred during data analysis: {str(e)}")
        raise e


def analyze_positive_delay(df, column, delay_column, logger):
    """
    Analyze positive departure/arrival delay.

    Parameters
    ----------
    df : DataFrame
        The DataFrame containing the data.
    column : str
        The column by which to group the data for analysis.
    delay_column : str
        The column representing departure/arrival delay.
    logger : object
        Logger object for logging messages.

    Returns
    -------
    DataFrame
        DataFrame containing the analysis of positive departure/arrival delay.

    Raises
    ------
    Exception
        If an error occurs during the computation.

    Notes
    -----
    This function calculates the average positive departure/arrival delay for a specified column,
    groups the data by another column, and ranks the results in descending order based on the average positive delay.
    """
    try:
        suffix = delay_column.split('_')[0]
        new_column_name = f"average_positive_{suffix}_delay"

        # Filter for positive delay values before computing the average.
        avg_positive_delay_by_column = df.groupBy(column).agg(avg(when(col(delay_column) > 0, col(
            delay_column))).alias(new_column_name)).orderBy(new_column_name, ascending=False)
        return avg_positive_delay_by_column
    except Exception as e:
        logger.error(f"An error occurred during data analysis: {str(e)}")
        raise e


def analyze_negative_delay(df, column, delay_column, logger):
    """
    Analyze negative departure/arrival delay.

    Parameters
    ----------
    df : DataFrame
        The DataFrame containing the data.
    column : str
        The column by which to group the data for analysis.
    delay_column : str
        The column representing departure/arrival delay.
    logger : object
        Logger object for logging messages.

    Returns
    -------
    DataFrame
        DataFrame containing the analysis of negative departure/arrival delay.

    Raises
    ------
    Exception
        If an error occurs during the computation.

    Notes
    -----
    This function calculates the average  departure/arrival delay for a specified column,
    groups the data by another column, and ranks the results in descending order based on the average delay.
    """
    try:
        suffix = delay_column.split('_')[0]
        new_column_name = f"average_negative_{suffix}_delay"

        # Filter for negative delay values before computing the average.
        avg_negative_delay_by_column = df.groupBy(column).agg(avg(when(col(delay_column) < 0, col(
            delay_column))).alias(new_column_name)).orderBy(new_column_name, ascending=False)
        return avg_negative_delay_by_column
    except Exception as e:
        logger.error(f"An error occurred during data analysis: {str(e)}")
        raise e


def numeric_stats(df, group_by_column, numeric_column, logger):
    """
    Compute statistics for a numeric column grouped by another column.

    Parameters
    ----------
    df : DataFrame
        The input DataFrame containing the data.
    group_by_column : str
        The name of the column to group by.
    numeric_column : str
        The name of the numeric column to compute statistics for.
    logger : object
        Logger object for logging messages.

    Returns
    -------
    DataFrame
        A DataFrame containing statistics (average, minimum, and maximum) for the numeric column
        grouped by the specified column.

    Raises
    ------
    Exception
        If an error occurs during the computation.

    Notes
    -----
    This function calculates statistics (average, minimum, maximum) for a specified numeric column in the DataFrame.
    The statistics are computed based on groups formed by the values in the specified 'group_by_column.'
    The resulting DataFrame is ordered in descending order of the average of the numeric column.
    """
    try:
        col_stats = df.groupBy(group_by_column).agg(
            avg(numeric_column).alias(f"average_{numeric_column}"),
            min(numeric_column).alias(f"min_{numeric_column}"),
            max(numeric_column).alias(f"max_{numeric_column}")
        ).orderBy(f"average_{numeric_column}", ascending=False)
        return col_stats
    except Exception as e:
        logger.error(f"An error occurred: {str(e)}")
        raise e


def compute_flight_speed(df, distance, air_time, logger):
    """
    Calculate flight speed in miles per hour and add it as a new column.

    Parameters
    ----------
    df : DataFrame
        The input DataFrame containing flight data.
    distance : str
        The name of the column representing flight distance in miles.
    air_time : str
        The name of the column representing flight air time in minutes.
    logger : object
        Logger object for logging messages.

    Returns
    -------
    DataFrame
        A DataFrame with an additional column, "flight_speed (miles per hour)," representing the calculated
        flight speed for each record.

    Raises
    ------
    Exception
        If an error occurs during the computation.

    Notes
    -----
    This function calculates the flight speed (in miles per hour) by dividing the flight distance (in miles)
    by the flight air time (in minutes) and adds it as a new column to the input DataFrame.

    Note: Flight air time is converted to hours by dividing by 60 to obtain the speed in miles per hour.
    """
    try:
        df_add_speed = df.withColumn(
            "flight_speed (miles per hour)", (col(distance) / (col(air_time) / 60)))
        return df_add_speed
    except Exception as e:
        logger.error(f"An error occurred: {str(e)}")
        raise e


def shortest_n_flight_from_origin(df, flight_column, origin_column, origin_name, measurement, n, logger):
    """
    Find the shortest 'n' flights from a specific origin based on a measurement.

    Parameters
    ----------
    df : DataFrame
        The DataFrame containing the flight data.
    origin_column : str
        The name of the column representing the flight origin.
    origin_name : str
        The name of the origin for which to find the shortest flights.
    measurement : str
        The column name representing the measurement by which to find the shortest flights.
    n : int
        The number of shortest flights to retrieve.
    logger : object
        Logger object for logging messages.

    Returns
    -------
    DataFrame
        DataFrame containing the 'n' shortest flights from the specified origin based on the given measurement.

    Raises
    ------
    Exception
        If an error occurs during the computation.

    Notes
    -----
    This function filters the DataFrame to select flights originating from a specific location (origin_name).
    It then sorts these flights by the provided measurement column in ascending order and retrieves the top 'n' shortest flights.
    """
    try:
        origin = df.filter(df[origin_column] == origin_name)

        shortest_flight = origin.select(
            flight_column, origin_column, measurement).orderBy(measurement, ascending=True).limit(n)
        return shortest_flight
    except Exception as e:
        logger.error(f"An error occurred: {str(e)}")
        raise e


def longest_n_flight_from_origin(df, flight_column, origin_column, origin_name, measurement, n, logger):
    """
    Find the longest 'n' flights from a specific origin based on a measurement.

    Parameters
    ----------
    df : DataFrame
        The DataFrame containing the flight data.
    origin_column : str
        The name of the column representing the flight origin.
    origin_name : str
        The name of the origin for which to find the longest flights.
    measurement : str
        The column name representing the measurement by which to find the longest flights.
    n : int
        The number of longest flights to retrieve.
    logger : object
        Logger object for logging messages.

    Returns
    -------
    DataFrame
        DataFrame containing the 'n' longest flights from the specified origin based on the given measurement.

    Raises
    ------
    Exception
        If an error occurs during the computation.

    Notes
    -----
    This function filters the DataFrame to select flights originating from a specific location (origin_name).
    It then sorts these flights by the provided measurement column in ascending order and retrieves the top 'n' longest flights.
    """
    try:
        origin = df.filter(df[origin_column] == origin_name)

        longest_flight = origin.select(
            flight_column, origin_column, measurement).orderBy(measurement, ascending=False).limit(n)
        return longest_flight
    except Exception as e:
        logger.error(f"An error occurred: {str(e)}")
        raise e


def average_duration(df, carrier_column, carrier_name, origin_column, origin_name, measurement, logger):
    """
    Calculate the average flight duration for a specific carrier and origin.

    Parameters
    ----------
    df : DataFrame
        The input DataFrame containing flight data.
    carrier_column : str
        The name of the column representing the carrier.
    carrier_name : str
        The name of the carrier for which to calculate the average duration.
    origin_column : str
        The name of the column representing the origin airport.
    origin_name : str
        The name of the origin airport for which to calculate the average duration.
    measurement : str
        The name of the column representing the flight duration measurement in minutes.
    logger : object
        Logger object for logging messages.

    Returns
    -------
    DataFrame
        A DataFrame with the average flight duration for the specified carrier and origin.

    Raises
    ------
    Exception
        If an error occurs during the computation.

    Notes
    -----
    This function filters the input DataFrame to select flights operated by a specific carrier and originating from a
    specific airport. It then calculates the average flight duration (in minutes) for these flights.
    """
    try:
        carrier = df.filter(df[carrier_column] == carrier_name)
        origin = carrier.filter(df[origin_column] == origin_name)

        average_duration = origin.groupBy(df[carrier_column], df[origin_column]).agg(
            avg(measurement).alias(f"average_{measurement} (mins)"))
        return average_duration
    except Exception as e:
        logger.error(f"An error occurred: {str(e)}")
        raise e


def total_duration(df, carrier_column, carrier_name, origin_column, origin_name, measurement, logger):
    """
    Calculate the total flight duration for a specific carrier and origin.

    Parameters
    ----------
    df : DataFrame
        The input DataFrame containing flight data.
    carrier_column : str
        The name of the column representing the carrier.
    carrier_name : str
        The name of the carrier for which to calculate the total duration.
    origin_column : str
        The name of the column representing the origin airport.
    origin_name : str
        The name of the origin airport for which to calculate the total duration.
    measurement : str
        The name of the column representing the flight duration measurement in minutes.
    logger : object
        Logger object for logging messages.

    Returns
    -------
    DataFrame
        A DataFrame with the total flight duration for the specified carrier and origin.

    Raises
    ------
    Exception
        If an error occurs during the computation.

    Notes
    -----
    This function filters the input DataFrame to select flights operated by a specific carrier and originating from a
    specific airport. It then calculates the total flight duration (in hours) for these flights.

    Note: Flight air time is converted to hours by dividing by 60 to obtain the total flight duration in hours.
    """
    try:
        carrier = df.filter(df[carrier_column] == carrier_name)
        origin = carrier.filter(df[origin_column] == origin_name)

        total_duration_hours = origin.groupBy(df[carrier_column], df[origin_column]).agg(
            (sum(measurement) / 60).alias(f"total_{measurement} (hours)"))
        return total_duration_hours
    except Exception as e:
        logger.error(f"An error occurred: {str(e)}")
        raise e


def main():
    """Entry point of the script.

    Parameters
    ----------
    None

    Returns
    -------
    None

    Notes
    -----
    This function serves as the entry point of the script for processing flight data. It performs the following steps:
    1. Configures the logging settings and initializes a logger.
    2. Creates a Spark session for data processing.
    3. Loads flight data from a CSV file and processes it to handle missing values.
    4. Performs various data analyses.
    5. Displays and logs the analysis results.
    6. Stops the Spark session when processing is complete.
    """
    logger = configure_logging()
    spark = create_spark_session()

    try:
        flights_data_frame = load_data(spark, logger, FLIGHTS_DATA_FILE_PATH)
        logger.info("Sample rows in the flights DataFrame:")
        show_dataframe(flights_data_frame)
        occurrence = flights_data_frame.count()
        logger.info(f"There are {occurrence} occurences in the DataFrame.\n")
        logger.info(flights_data_frame.schema)

        clean_flights_data_df = process_missing_data(
            flights_data_frame, logger)
        logger.info("Sample rows in the cleaned flights DataFrame:")
        show_dataframe(clean_flights_data_df)
        clean_occurrence = clean_flights_data_df.count()
        logger.info(
            f"{clean_occurrence} rows remained after removing the rows with missing values.\n")

        flight_by_year_month = count_by(clean_flights_data_df, [
            "year", "month"], logger)
        show_dataframe(flight_by_year_month)

        flight_by_day = count_by(clean_flights_data_df, ["day"], logger)
        show_dataframe(flight_by_day)

        percentage_flight_by_carrier = percentage_by(
            clean_flights_data_df, ["carrier"], logger)
        show_dataframe(percentage_flight_by_carrier)

        flights_by_origin = count_by(clean_flights_data_df, ["origin"], logger)
        show_dataframe(flights_by_origin)

        flights_by_dest = count_by(clean_flights_data_df, ["dest"], logger)
        show_dataframe(flights_by_dest)

        top_10_planes = top_n_cat_by(
            clean_flights_data_df, ["tailnum"], 10, logger)
        show_dataframe(top_10_planes)

        flights_by_hour = count_by(clean_flights_data_df, ["hour"], logger)
        show_dataframe(flights_by_hour)

        avg_pos_dep_delay_by_carrier = analyze_positive_delay(
            clean_flights_data_df, "carrier", "dep_delay", logger)
        show_dataframe(avg_pos_dep_delay_by_carrier)

        avg_dep_delay_by_carrier = analyze_average_delay(
            clean_flights_data_df, "carrier", "dep_delay", logger)
        show_dataframe(avg_dep_delay_by_carrier)

        avg_dep_delay_by_month = analyze_average_delay(
            clean_flights_data_df, "month", "dep_delay", logger)
        show_dataframe(avg_dep_delay_by_month)

        avg_dep_delay_by_hour = analyze_average_delay(
            clean_flights_data_df, "hour", "dep_delay", logger)
        show_dataframe(avg_dep_delay_by_hour)

        avg_neg_dep_delay_by_carrier = analyze_negative_delay(
            clean_flights_data_df, "carrier", "dep_delay", logger)
        show_dataframe(avg_neg_dep_delay_by_carrier)

        avg_neg_dep_delay_by_month = analyze_negative_delay(
            clean_flights_data_df, "month", "dep_delay", logger)
        show_dataframe(avg_neg_dep_delay_by_month)

        avg_neg_dep_delay_by_hour = analyze_negative_delay(
            clean_flights_data_df, "hour", "dep_delay", logger)
        show_dataframe(avg_neg_dep_delay_by_hour)

        distance_stats = numeric_stats(
            clean_flights_data_df, "carrier", "distance", logger)
        show_dataframe(distance_stats)

        transformed_flights_df = compute_flight_speed(
            clean_flights_data_df, "distance", "air_time", logger)
        show_dataframe(transformed_flights_df)

        speed_stats = numeric_stats(
            transformed_flights_df, "carrier", "flight_speed (miles per hour)", logger)
        show_dataframe(speed_stats)

        shortest_flight_distance_PDX = shortest_n_flight_from_origin(
            transformed_flights_df, "flight", "origin", "PDX", "distance", 1, logger)
        show_dataframe(shortest_flight_distance_PDX)

        longest_flight_distance_SEA = longest_n_flight_from_origin(
            transformed_flights_df, "flight", "origin", "SEA", "air_time", 1, logger)
        show_dataframe(longest_flight_distance_SEA)

        average_duration_UA_SEA = average_duration(
            transformed_flights_df, "carrier", "UA", "origin", "SEA", "air_time", logger)
        show_dataframe(average_duration_UA_SEA)

        total_duration_UA_SEA = total_duration(
            transformed_flights_df, "carrier", "UA", "origin", "SEA", "air_time", logger)
        show_dataframe(total_duration_UA_SEA)

        planes_data_frame = load_data(spark, logger, PLANES_DATA_FILE_PATH)
        logger.info("Sample rows in the planes DataFrame:")
        show_dataframe(planes_data_frame)
        occurrence = planes_data_frame.count()
        logger.info(f"There are {occurrence} occurences in the DataFrame.\n")
        logger.info(planes_data_frame.schema)

        clean_planes_data_df = planes_data_frame.drop("speed")
        clean_planes_data_df = clean_planes_data_df.withColumnRenamed(
            "year", "plane_year")
        show_dataframe(clean_planes_data_df)

        flights_planes_df = transformed_flights_df.join(
            clean_planes_data_df, on=["tailnum"], how="inner")
        show_dataframe(flights_planes_df)
        occurrence = flights_planes_df.count()
        logger.info(f"There are {occurrence} occurences in the DataFrame.\n")

        clean_flights_planes_data_df = process_missing_data(
            flights_planes_df, logger)
        logger.info("Sample rows in the cleaned planes DataFrame:")
        show_dataframe(clean_flights_planes_data_df)
        clean_occurrence = clean_flights_planes_data_df.count()
        logger.info(
            f"{clean_occurrence} rows remained after removing the rows with missing values.\n")

        top_20_flights_planes = top_n_cat_by(clean_flights_planes_data_df, [
            "carrier", "model", "plane_year"], 20, logger)
        show_dataframe(top_20_flights_planes)

        bottom_20_flights_planes = bottom_n_cat_by(clean_flights_planes_data_df, [
            "carrier", "model", "plane_year"], 20, logger)
        show_dataframe(bottom_20_flights_planes)

        sql_top_20_flights_planes = sql_top_n_cat_by(clean_flights_planes_data_df, [
            "carrier", "model", "plane_year"], 20, spark, logger)
        show_dataframe(sql_top_20_flights_planes)

        sql_bottom_20_flights_planes = sql_bottom_n_cat_by(clean_flights_planes_data_df, [
            "carrier", "model", "plane_year"], 20, spark, logger)
        show_dataframe(sql_bottom_20_flights_planes)
    except Exception as e:
        logger.error(f"An error occurred: {str(e)}")
        raise e
    finally:
        if spark is not None:
            spark.stop()


if __name__ == "__main__":
    main()
