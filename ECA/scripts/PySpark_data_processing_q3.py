import logging
import os
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, avg, min, max, when, split, lit, concat

# Constants
SCRIPTS_DIR = os.path.abspath(os.path.dirname(__file__))
DATA_DIR = os.path.join(SCRIPTS_DIR, "..", "data")
VEHICLE_MPG_DATA_FILE_PATH = os.path.join(DATA_DIR, "vehicle_mpg.tsv")
VEHICLE_MANUFACTURERS_DATA_FILE_PATH = os.path.join(
    DATA_DIR, "vehicle_manufacturers.csv")
LOGGING_LEVEL = logging.INFO
LOAD_DATA_ERROR_MESSAGE = "An error occurred while loading data: {}"
FILE_NOT_FOUND_MESSAGE = "The specified file does not exist: {}"

# Configuration
numeric_dtypes = ("int", "double", "float", "decimal")

mpg_class_config = {
    "low": {"max_value": 20},
    "mid": {"min_value": 20, "max_value": 30},
    "high": {"min_value": 30, "max_value": 40},
    "very high": {"min_value": 40}
}


def configure_logging():
    """
    Configure logging settings and return a logger object.

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
        The maximum number of rows to display. Default is 100.

    show_rows : int, optional
        The rows to display if records is above max rows. Default is 20.

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


def shape(df, logger):
    """
    Display the shape of a DataFrame.

    Parameters
    ----------
    df : DataFrame
        The DataFrame to be analyzed.
    logger : Logger
        Logger object for logging messages.

    Returns
    -------
    None

    Notes
    -----
    This function calculates the number of rows and columns in the input DataFrame.

    The 'logger' object is used for logging and displaying the total number of rows and columns.

    Note: The default logging level is set to the value of the constant LOGGING_LEVEL.
    """
    num_rows = df.count()
    num_columns = len(df.columns)
    logger.info(
        f"Number of Rows: {num_rows}, Number of Columns: {num_columns}")


def load_data(spark, logger, file_path, delimiter=","):
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
    delimiter : str, optional
        The delimiter used in the CSV file. Default is ",".

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
    The data is assumed to be in CSV format, and the default delimiter is a comma (','), which can be customized using the 'delimiter' parameter.

    The 'logger' object is used for logging messages, and any error that occurs during the data loading process is logged and raised as an exception.

    Note: The default logging level is set to the value of the constant LOGGING_LEVEL.
    """
    try:
        # Load data from csv
        df = spark.read.option("inferSchema", "true").option(
            "header", "true").option("delimiter", delimiter).csv(file_path)
        return df
    except Exception as e:
        if "Path does not exist" in str(e):
            logger.error(FILE_NOT_FOUND_MESSAGE.format(file_path))
            raise FileNotFoundError(f"File not found: {file_path}")

        logger.error(LOAD_DATA_ERROR_MESSAGE.format(str(e)))
        raise e


def process_missing_data(loaded_df, logger):
    """Process and analyze the loaded data.

    Parameters
    ----------
    loaded_df : DataFrame
        DataFrame containing the data.
    logger : object
        Logger object for logging messages.

    Returns
    -------
    DataFrame
        DataFrame containing the cleansed data.

    Raises
    ------
    Exception
        If an error occurs during data processing.

    Notes
    -----
    This function determines which rows in the supplied DataFrame have missing values,
    displays those rows, and removes them from the DataFrame. It details the total number of missing values,
    the cleaned DataFrame that is produced, and any problems that may have occurred.

    The 'logger' object is used for logging messages, and any error that occurs during the processing of missing data is logged and raised as an exception.

    Note: The default logging level is set to the value of the constant LOGGING_LEVEL.
    """
    try:
        columns_to_check = loaded_df.columns
        filter_condition = None

        # Loop through the list of columns and build a filter condition to check for null values.
        for column_name in columns_to_check:
            if filter_condition is None:
                filter_condition = col(column_name).isNull()
            else:
                filter_condition = filter_condition | col(column_name).isNull()

        missing_data_df = loaded_df.filter(filter_condition)

        logger.info("Sample rows in the DataFrame with Missing Value:")
        show_dataframe(missing_data_df)
        missing_occurrence = missing_data_df.count()
        logger.info(
            f"There are {missing_occurrence} rows with missing values in DataFrame.\n")

        clean_data_df = loaded_df.filter(~filter_condition)
        return clean_data_df
    except Exception as e:
        logger.error(f"An error occurred: {str(e)}")
        raise e


def numeric_summary(cleansed_df, numeric_dtypes, logger):
    """
    Generate summary statistics for numeric columns in a DataFrame.

    Parameters
    ----------
    cleansed_df : DataFrame
        The input DataFrame containing the cleansed data.
    numeric_dtypes : str or list
        The data types associated with numeric columns, e.g., "int," "double," "float," or "decimal."
        You can also provide a list of data types.
    logger : object
        Logger object for logging messages.

    Returns
    -------
    DataFrame
        A DataFrame containing summary statistics for the numeric columns.

    Raises
    ------
    Exception
        If an error occurs while generating the summary statistics.

    Notes
    -----
    This function computes summary statistics for columns with data types typically associated with numeric values.
    It selects the numeric columns from the input DataFrame and computes summary statistics using the `summary()` method.

    The 'numeric_dtypes' parameter specifies the data types that should be considered numeric for summary calculation.
    It can be a single data type or a list of data types.

    The 'logger' object is used for logging messages, and any error that occurs during the computation of summary is logged and raised as an exception.

    Note: The default logging level is set to the value of the constant LOGGING_LEVEL.
    """
    try:
        numeric_columns = [
            col[0] for col in cleansed_df.dtypes if col[1].startswith(numeric_dtypes)]
        return cleansed_df.select(numeric_columns).summary()
    except Exception as e:
        logger.error(f"An error occurred: {str(e)}")
        raise e


def count_by(cleansed_df, grouped_columns, logger):
    """
    Count occurrences of rows by grouping columns.

    Parameters
    ----------
    cleansed_df : DataFrame
        DataFrame containing the cleansed data.
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

    The 'logger' object is used for logging messages, and any error that occurs during the count by process is logged and raised as an exception.

    Note: The default logging level is set to the value of the constant LOGGING_LEVEL.
    """
    try:
        count_by_col = cleansed_df.groupby(*grouped_columns).count()
        sorted_counts_df = count_by_col.orderBy("count", ascending=False)
        return sorted_counts_df
    except Exception as e:
        logger.error(f"An error occurred: {str(e)}")
        raise e


def custom_split(cleansed_df, column_to_split, seperator, value_position, new_column_name, logger):
    """
    Split a DataFrame column based on a separator and create a new column with the selected value position.

    Parameters
    ----------
    cleansed_df : DataFrame
        The input DataFrame containing the cleansed data.
    column_to_split : str
        The name of the column to split.
    separator : str
        The separator used to split the column values.
    value_position : int
        The position of the value to select after splitting (1-based index).
    new_column_name : str
        The name of the new column to store the selected values.
    logger : object
        Logger object for logging messages.

    Returns
    -------
    DataFrame
        A new DataFrame with the added column containing the selected values.

    Raises
    ------
    Exception
        If an error occurs during the split and column creation.

    Notes
    -----
    This function takes an input DataFrame and splits a specified column using a given separator.
    It then creates a new column containing the selected value at the specified position after splitting.
    The resulting DataFrame includes the new column and the original data.

    The 'logger' object is used for logging messages, and any error that occurs during the custom split is logged and raised as an exception.

    Note: The default logging level is set to the value of the constant LOGGING_LEVEL.
    """
    try:
        clean_data_df = cleansed_df.withColumn(new_column_name, split(
            col(column_to_split), seperator)[value_position-1])
        return (clean_data_df)
    except Exception as e:
        logger.error(f"An error occurred: {str(e)}")
        raise e


def column_class(cleansed_df, numeric_column, new_column_name, config: dict, logger):
    """
    Classify numeric values in a DataFrame based on a provided configuration and add the result as a new column.

    Parameters:
    -----------
    cleansed_df : DataFrame
        The input DataFrame containing the cleansed data.
    numeric_column : str
        The name of the numeric column in the DataFrame to be classified.
    new_column_name : str
        The name of the new column to store the classification results.
    config : dict
        A dictionary that defines the classification criteria.
        It should be in the format:
        {
            "class1": {"min_value": min1, "max_value": max1},
            "class2": {"min_value": min2, "max_value": max2},
            ...
        }
        where "min_value" and "max_value" define the range for each class.
    logger : object
        Logger object for logging messages.

    Returns:
    --------
    DataFrame
        A DataFrame with the new classification column added.

    Notes:
    ------
    This function takes a DataFrame, a numeric column to classify, a new column name to store the classification results,
    a configuration dictionary specifying the classification ranges, and a logger for error logging.
    It classifies the values in the specified numeric column based on the configuration and adds the classification results
    to the DataFrame as a new column.

    The 'logger' object is used for logging messages, and any error that occurs during the classification is logged and raised as an exception.

    Note: The default logging level is set to the value of the constant LOGGING_LEVEL.
    """
    try:
        classification = when(col(numeric_column) <=
                              config["low"]["max_value"], "low")

        for class_name, class_config in config.items():
            if class_name != "unknown":
                min_value = class_config.get("min_value", float("-inf"))
                max_value = class_config.get("max_value", float("inf"))
                classification = classification.when(
                    (col(numeric_column) > min_value) & (
                        col(numeric_column) <= max_value), class_name
                )

        return cleansed_df.withColumn(new_column_name, classification)
    except Exception as e:
        logger.error(f"An error occurred: {str(e)}")
        raise e


def numeric_stats(cleansed_df, group_by_column, numeric_column, logger):
    """
    Compute statistics for a numeric column grouped by another column.

    Parameters
    ----------
    cleansed_df : DataFrame
        The input DataFrame containing the cleansed data.
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
        grouped by the specified column, sorted in descending order.

    Raises
    ------
    Exception
        If an error occurs during the computation.

    Notes
    -----
    This function calculates statistics (average, minimum, maximum) for a specified numeric column in the DataFrame.
    The statistics are computed based on groups formed by the values in the specified 'group_by_column.'
    The resulting DataFrame is ordered in descending order of the average of the numeric column.

    The 'logger' object is used for logging messages, and any error that occurs during the calculation of mathematical functions is logged and raised as an exception.

    Note: The default logging level is set to the value of the constant LOGGING_LEVEL.
    """
    try:
        col_stats = cleansed_df.groupBy(group_by_column).agg(
            avg(numeric_column).alias(f"average_{numeric_column}"),
            min(numeric_column).alias(f"min_{numeric_column}"),
            max(numeric_column).alias(f"max_{numeric_column}")
        ).orderBy(f"average_{numeric_column}", ascending=False)
        return col_stats
    except Exception as e:
        logger.error(f"An error occurred: {str(e)}")
        raise e


def numeric_stats_sql(cleansed_df, group_by_column, numeric_column, spark, logger):
    """
    Compute statistics for a numeric column grouped by another column using PySpark SQL.

    Parameters
    ----------
    cleansed_df : DataFrame
        The input DataFrame containing the cleansed data.
    group_by_column : str
        The name of the column to group by.
    numeric_column : str
        The name of the numeric column to compute statistics for.
    spark : SparkSession
        The Spark session.
    logger : object
        Logger object for logging messages.

    Returns
    -------
    DataFrame
        A DataFrame containing statistics (average, minimum, and maximum) for the numeric column
        grouped by the specified column, sorted in descending order.

    Raises
    ------
    Exception
        If an error occurs during the computation.

    Notes
    -----
    This function calculates statistics (average, minimum, maximum) for a specified numeric column in the DataFrame.
    The statistics are computed based on groups formed by the values in the specified 'group_by_column.'
    The resulting DataFrame is ordered in descending order of the average of the numeric column.

    The 'logger' object is used for logging messages, and any error that occurs during the calculation of mathematical functions is logged and raised as an exception.

    Note: The default logging level is set to the value of the constant LOGGING_LEVEL.
    """
    try:
        cleansed_df.createOrReplaceTempView("data")
        query = f"""
            SELECT {group_by_column},
                AVG({numeric_column}) AS average_{numeric_column},
                MIN({numeric_column}) AS min_{numeric_column},
                MAX({numeric_column}) AS max_{numeric_column}
            FROM data
            GROUP BY {group_by_column}
            ORDER BY average_{numeric_column} DESC
        """
        col_stats = spark.sql(query)
        return col_stats
    except Exception as e:
        logger.error(f"An error occurred: {str(e)}")
        raise e


def average_by(cleansed_df, grouped_columns, measurement, logger):
    """
    Average occurrences of rows by grouping columns.

    Parameters
    ----------
    cleansed_df : DataFrame
        DataFrame containing the cleansed data.
    grouped_columns : list
        List of columns to group by.
    aggregate_column : str
        Name of the column to aggregate.
    logger : object
        Logger object for logging messages.

    Returns
    -------
    DataFrame
        DataFrame with averages, sorted in descending order.

    Raises
    ------
    Exception
        If an error occurs during averaging.

    Notes
    -----
    This function divides the data in the input DataFrame into groups according to the chosen columns,
    then averages the specified column in each group. The outcome is a DataFrame with averages that is
    sorted in decreasing order according to the average values.

    The 'logger' object is used for logging messages, and any error that occurs during the calculation of mathematical function is logged and raised as an exception.

    Note: The default logging level is set to the value of the constant LOGGING_LEVEL.
    """
    try:
        avg_column = avg(col(measurement)).alias(f"avg_{measurement}")
        avg_by_col = cleansed_df.groupBy(grouped_columns).agg(avg_column)

        sorted_average_df = avg_by_col.orderBy(
            f"avg_{measurement}", ascending=False)
        return sorted_average_df
    except Exception as e:
        logger.error(f"An error occurred: {str(e)}")
        raise e


def average_by_sql(cleansed_df, grouped_columns, measurement, spark, logger):
    """
    Average occurrences of rows by grouping columns using PySpark SQL.

    Parameters
    ----------
    cleansed_df : DataFrame
        DataFrame containing the cleansed data.
    grouped_columns : list
        List of columns to group by.
    measurement : str
        Name of the column to aggregate.
    spark : SparkSession
        The Spark session.
    logger : object
        Logger object for logging messages.

    Returns
    -------
    DataFrame
        DataFrame with averages, sorted in descending order.

    Raises
    ------
    Exception
        If an error occurs during averaging.

    Notes
    -----
    This function divides the data in the input DataFrame into groups according to the chosen columns,
    then averages the specified column in each group. The outcome is a DataFrame with averages that is
    sorted in decreasing order according to the average values.

    The 'logger' object is used for logging messages, and any error that occurs during the calculation of mathematical function is logged and raised as an exception.

    Note: The default logging level is set to the value of the constant LOGGING_LEVEL.
    """
    try:
        cleansed_df.createOrReplaceTempView("data")
        query = f"""
            SELECT {",".join(grouped_columns)},
                AVG({measurement}) AS avg_{measurement}
            FROM data
            GROUP BY {",".join(grouped_columns)}
            ORDER BY avg_{measurement} DESC
        """
        avg_by_col = spark.sql(query)
        return avg_by_col
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
        # Load tab delimited file
        vehicle_mpg_data_frame = load_data(
            spark, logger, VEHICLE_MPG_DATA_FILE_PATH, "\t")
        logger.info("Sample rows in the vehicle DataFrame:")
        show_dataframe(vehicle_mpg_data_frame)
        occurrence = vehicle_mpg_data_frame.count()
        logger.info(f"There are {occurrence} occurrences in the DataFrame.\n")
        logger.info(vehicle_mpg_data_frame.schema)
        shape(vehicle_mpg_data_frame, logger)

        clean_vehicle_mpg_df_01 = process_missing_data(
            vehicle_mpg_data_frame, logger)
        logger.info("Sample rows in the cleaned vehicle DataFrame:")
        show_dataframe(clean_vehicle_mpg_df_01)
        clean_occurrence = clean_vehicle_mpg_df_01.count()
        logger.info(
            f"{clean_occurrence} rows remained after removing the rows with missing values.\n")

        stats_num_columns = numeric_summary(
            clean_vehicle_mpg_df_01, numeric_dtypes, logger)
        show_dataframe(stats_num_columns)

        clean_vehicle_mpg_df_02 = custom_split(
            clean_vehicle_mpg_df_01, "carname", " ", 1, "manufacturer", logger)
        show_dataframe(clean_vehicle_mpg_df_02)

        manufacturer_occurrence = count_by(
            clean_vehicle_mpg_df_02, ["manufacturer"], logger)
        show_dataframe(manufacturer_occurrence)

        clean_vehicle_mpg_df_03 = clean_vehicle_mpg_df_02.withColumn(
            "modelyear", concat(lit("19"), col("modelyear")))
        show_dataframe(clean_vehicle_mpg_df_03)

        model_year_occurrence = count_by(
            clean_vehicle_mpg_df_03, ["modelyear"], logger)
        show_dataframe(model_year_occurrence)

        clean_vehicle_mpg_df_mpg_class = column_class(
            clean_vehicle_mpg_df_03, "mpg", "mpg_class", mpg_class_config, logger)
        show_dataframe(clean_vehicle_mpg_df_mpg_class)

        mpg_class_occurrence = count_by(
            clean_vehicle_mpg_df_mpg_class, ["mpg_class"], logger)
        show_dataframe(mpg_class_occurrence)

        vehicle_manu_data_frame = load_data(
            spark, logger, VEHICLE_MANUFACTURERS_DATA_FILE_PATH)
        logger.info("Sample rows in the vehicle manufacturers DataFrame:")
        show_dataframe(vehicle_manu_data_frame)
        occurrence = vehicle_manu_data_frame.count()
        logger.info(f"There are {occurrence} occurrences in the DataFrame.\n")

        vehicle_full_df = clean_vehicle_mpg_df_mpg_class.join(
            vehicle_manu_data_frame, on=["manufacturer"], how="inner")
        show_dataframe(vehicle_full_df)
        occurrence = vehicle_full_df.count()
        logger.info(
            f"There are {occurrence} occurrences in the full vehicle DataFrame.\n")

        mpg_stats_by_country = numeric_stats(
            vehicle_full_df, "country", "mpg", logger)
        show_dataframe(mpg_stats_by_country)

        mpg_stats_by_cylinders = numeric_stats(
            vehicle_full_df, "cylinders", "mpg", logger)
        show_dataframe(mpg_stats_by_cylinders)

        mpg_stats_by_modelyear = numeric_stats(
            vehicle_full_df, "modelyear", "mpg", logger)
        show_dataframe(mpg_stats_by_modelyear)

        mpg_stats_by_manufacturer = numeric_stats(
            vehicle_full_df, "manufacturer", "mpg", logger)
        show_dataframe(mpg_stats_by_manufacturer)

        average_mpg_by_carname_manufacturer = average_by(
            vehicle_full_df, ["carname", "manufacturer"], "mpg", logger)
        show_dataframe(average_mpg_by_carname_manufacturer)

        mpg_stats_by_country_sql = numeric_stats_sql(
            vehicle_full_df, "country", "mpg", spark, logger)
        show_dataframe(mpg_stats_by_country_sql)

        mpg_stats_by_cylinders_sql = numeric_stats_sql(
            vehicle_full_df, "cylinders", "mpg", spark, logger)
        show_dataframe(mpg_stats_by_cylinders_sql)

        mpg_stats_by_modelyear_sql = numeric_stats_sql(
            vehicle_full_df, "modelyear", "mpg", spark, logger)
        show_dataframe(mpg_stats_by_modelyear_sql)

        mpg_stats_by_manufacturer_sql = numeric_stats_sql(
            vehicle_full_df, "manufacturer", "mpg", spark, logger)
        show_dataframe(mpg_stats_by_manufacturer_sql)

        average_mpg_by_carname_manufacturer_sql = average_by_sql(
            vehicle_full_df, ["carname", "manufacturer"], "mpg", spark, logger)
        show_dataframe(average_mpg_by_carname_manufacturer_sql)
    except Exception as e:
        logger.error(f"An error occurred: {str(e)}")
        raise e
    finally:
        if spark is not None:
            spark.stop()


if __name__ == "__main__":
    main()
