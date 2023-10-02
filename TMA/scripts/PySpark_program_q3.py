import logging
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, avg, when

# Constants
DATA_FILE_PATH = r"C:\Everything\SUSS\DE\ICT337\TMA\data\flights_data_v2.csv"
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
    This function initializes the logging settings, including the logging level,
    and returns a logger object that can be used for logging messages within the application.

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


def load_data(spark, logger, file_path=DATA_FILE_PATH):
    """
    Load data from CSV file into a Spark DataFrame.

    Parameters
    ----------
    spark : SparkSession
        The Spark session.
    logger : Logger
        Logger object for logging messages.
    file_path : str, optional
        The path to the CSV file to load, by default DATA_FILE_PATH.

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

    The default logging level is set to the value of the constant LOGGING_LEVEL.
    """
    try:
        # Load data from csv
        flights_df = spark.read.option("inferSchema", "true").option(
            "header", "true").csv(DATA_FILE_PATH)

        # Display sample data, number of occurrences, and schema
        logger.info("Sample rows in the flights_df DataFrame:")
        flights_df.show(5)
        flight_occurrence = flights_df.count()

        logger.info(f"There are {flight_occurrence} number of flights_df.\n")
        logger.info(flights_df.schema)
        return flights_df
    except Exception as e:
        if "Path does not exist" in str(e):
            logger.error(FILE_NOT_FOUND_MESSAGE.format(file_path))
            raise FileNotFoundError(f"File not found: {file_path}")

        logger.error(LOAD_DATA_ERROR_MESSAGE.format(str(e)))
        raise e


def process_missing_data(flights_df, logger):
    """Process and analyze the loaded data.

    Parameters
    ----------
    flights_df : DataFrame
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
    This function checks for missing values in the columns of the input DataFrame, identifies and displays
    rows with missing values, and removes those rows from the DataFrame. It provides information about
    the number of missing values, the resulting cleaned DataFrame, and any errors encountered during
    the process.
    """
    try:
        # Check for missing values in columns
        columns_to_check = flights_df.columns
        filter_condition = None

        for column_name in columns_to_check:
            if filter_condition is None:
                filter_condition = col(column_name).isNull()
            else:
                filter_condition = filter_condition | col(column_name).isNull()

        # Find and display rows with missing values
        missing_data_flights_df = flights_df.filter(filter_condition)

        logger.info(
            "Sample rows in the flights_df DataFrame with Missing Value:")
        missing_data_flights_df.show(5)

        missing_occurrence = missing_data_flights_df.count()
        logger.info(
            f"There are {missing_occurrence} rows with missing values in flights_df.\n")

        # Remove rows with missing values
        clean_data_flights_df = flights_df.filter(~filter_condition)

        logger.info("Sample rows in the cleaned flights_df DataFrame:")
        clean_data_flights_df.show(5)

        clean_flight_occurrence = clean_data_flights_df.count()

        logger.info(
            f"{clean_flight_occurrence} rows remained after removing the rows with missing values.\n")
        return clean_data_flights_df
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
    This function groups the data in the input DataFrame by the specified columns and counts the
    occurrences of rows within each group. The result is a DataFrame containing counts, sorted in
    descending order based on the count values.
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
    This function groups the data in the input DataFrame by the specified columns and calculates
    the number and percentage of occurrences within each group relative to the total number of rows in the
    DataFrame. The result is a DataFrame containing both counts and percentages, sorted in
    descending order based on the percentage values.
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


def top_cat_by(df, column, n, logger):
    """
    Find the top n category.

    Parameters
    ----------
    df : DataFrame
        DataFrame containing the data.
    column : str
        The column to group by and count.
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
    This function groups the data in the specified DataFrame by the given column and counts
    the occurrences of each category. It returns a DataFrame containing the top n categories
    with the highest counts, sorted in descending order based on the count values.
    """
    try:
        top_cat = df.groupBy(column).count().orderBy(
            "count", ascending=False).limit(n)

        return top_cat
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

    Notes
    -----
    This function calculates the average departure/arrival delay for a specified column,
    groups the data by another column, and orders the results in descending order based on the average delay.
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

    Notes
    -----
    This function calculates the average positive departure/arrival delay for a specified column,
    groups the data by another column, and orders the results in descending order based on the average positive delay.
    """
    try:
        suffix = delay_column.split('_')[0]
        new_column_name = f"average_positive_{suffix}_delay"

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

    Notes
    -----
    This function calculates the average negative departure/arrival delay for a specified column,
    groups the data by another column, and orders the results in ascending order based on the average negative delay.
    """
    try:
        suffix = delay_column.split('_')[0]
        new_column_name = f"average_negative_{suffix}_delay"

        avg_negative_delay_by_column = df.groupBy(column).agg(avg(when(col(delay_column) < 0, col(
            delay_column))).alias(new_column_name)).orderBy(new_column_name, ascending=True)

        return avg_negative_delay_by_column
    except Exception as e:
        logger.error(f"An error occurred during data analysis: {str(e)}")
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
        flights_data_frame = load_data(spark, logger)
        clean_data_flights_df = process_missing_data(
            flights_data_frame, logger)

        flight_by_year_month = count_by(clean_data_flights_df, [
            "year", "month"], logger)
        flight_by_year_month.show(5)

        flight_by_day = count_by(clean_data_flights_df, ["day"], logger)
        flight_by_day.show(5)

        percentage_flight_by_carrier = percentage_by(
            clean_data_flights_df, ["carrier"], logger)
        percentage_flight_by_carrier.show(5)

        flights_by_origin = count_by(clean_data_flights_df, ["origin"], logger)
        flights_by_origin.show(5)

        flights_by_dest = count_by(clean_data_flights_df, ["dest"], logger)
        flights_by_dest.show(5)

        top_cat = top_cat_by(
            clean_data_flights_df, "tailnum", 10, logger)
        top_cat.show()

        flights_by_hour = count_by(clean_data_flights_df, ["hour"], logger)
        flights_by_hour.show(5)

        avg_pos_dep_delay_by_carrier = analyze_positive_delay(
            clean_data_flights_df, "carrier", "dep_delay", logger)
        avg_pos_dep_delay_by_carrier.show(5)

        avg_dep_delay_by_carrier = analyze_average_delay(
            clean_data_flights_df, "carrier", "dep_delay", logger)
        avg_dep_delay_by_carrier.show(5)

        avg_dep_delay_by_month = analyze_average_delay(
            clean_data_flights_df, "month", "dep_delay", logger)
        avg_dep_delay_by_month.show(5)

        avg_dep_delay_by_hour = analyze_average_delay(
            clean_data_flights_df, "hour", "dep_delay", logger)
        avg_dep_delay_by_hour.show(5)

        avg_neg_dep_delay_by_carrier = analyze_negative_delay(
            clean_data_flights_df, "carrier", "dep_delay", logger)
        avg_neg_dep_delay_by_carrier.show(5)

        avg_neg_dep_delay_by_month = analyze_negative_delay(
            clean_data_flights_df, "month", "dep_delay", logger)
        avg_neg_dep_delay_by_month.show(5)

        avg_neg_dep_delay_by_hour = analyze_negative_delay(
            clean_data_flights_df, "hour", "dep_delay", logger)
        avg_neg_dep_delay_by_hour.show(5)
    except Exception as e:
        logger.error(f"An error occurred: {str(e)}")
        raise e
    finally:
        if spark is not None:
            spark.stop()


if __name__ == "__main__":
    main()
