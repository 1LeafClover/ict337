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


def shape(df):
    num_rows = df.count()
    num_columns = len(df.columns)
    print(f"Number of Rows: {num_rows}, Number of Columns: {num_columns}")


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
            "header", "true").option("delimiter", delimiter).csv(file_path)
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


def numeric_stats_sql(df, group_by_column, numeric_column, spark, logger):
    """
    Compute statistics for a numeric column grouped by another column using PySpark SQL.

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
        df.createOrReplaceTempView("data")
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


def average_by(df, grouped_columns, measurement, logger):
    """
    Average occurrences of rows by grouping columns.

    Parameters
    ----------
    df : DataFrame
        DataFrame containing the data.
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
    """
    try:
        avg_column = avg(col(measurement)).alias(f"avg_{measurement}")
        avg_by_col = df.groupBy(grouped_columns).agg(avg_column)

        sorted_average_df = avg_by_col.orderBy(
            f"avg_{measurement}", ascending=False)
        return sorted_average_df
    except Exception as e:
        logger.error(f"An error occurred: {str(e)}")
        raise e


def average_by_sql(df, grouped_columns, measurement, spark, logger):
    """
    Average occurrences of rows by grouping columns using PySpark SQL.

    Parameters
    ----------
    df : DataFrame
        DataFrame containing the data.
    grouped_columns : list
        List of columns to group by.
    measurement : str
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
    """
    try:
        df.createOrReplaceTempView("data")
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
        logger.info(f"There are {occurrence} occurences in the DataFrame.\n")
        logger.info(vehicle_mpg_data_frame.schema)
        shape(vehicle_mpg_data_frame)

        clean_vehicle_mpg_df_01 = process_missing_data(
            vehicle_mpg_data_frame, logger)
        logger.info("Sample rows in the cleaned vehicle DataFrame:")
        show_dataframe(clean_vehicle_mpg_df_01)
        clean_occurrence = clean_vehicle_mpg_df_01.count()
        logger.info(
            f"{clean_occurrence} rows remained after removing the rows with missing values.\n")
        numeric_columns = [col[0] for col in clean_vehicle_mpg_df_01.dtypes if col[1].startswith(
            ("int", "double", "float", "decimal"))]
        clean_vehicle_mpg_df_01.select(numeric_columns).summary().show()

        clean_vehicle_mpg_df_02 = clean_vehicle_mpg_df_01.withColumn(
            "manufacturer", split(col("carname"), " ")[0])
        show_dataframe(clean_vehicle_mpg_df_02)

        clean_vehicle_mpg_df_03 = clean_vehicle_mpg_df_02.withColumn(
            "modelyear", concat(lit("19"), col("modelyear")))
        show_dataframe(clean_vehicle_mpg_df_03)

        clean_vehicle_mpg_df_mpg_class = clean_vehicle_mpg_df_03.withColumn("mpg_class", when(col("mpg") <= 20, "low")
                                                                            .when((col("mpg") > 20) & (col("mpg") <= 30), "mid")
                                                                            .when((col("mpg") > 30) & (col("mpg") <= 40), "high")
                                                                            .when(col("mpg") > 40, "very high")
                                                                            .otherwise("unknown"))
        show_dataframe(clean_vehicle_mpg_df_mpg_class)

        vehicle_manu_data_frame = load_data(
            spark, logger, VEHICLE_MANUFACTURERS_DATA_FILE_PATH)
        logger.info("Sample rows in the vehicle manufacturers DataFrame:")
        show_dataframe(vehicle_manu_data_frame)
        occurrence = vehicle_manu_data_frame.count()
        logger.info(f"There are {occurrence} occurences in the DataFrame.\n")

        vehicle_full_df = clean_vehicle_mpg_df_mpg_class.join(
            vehicle_manu_data_frame, on=["manufacturer"], how="inner")
        show_dataframe(vehicle_full_df)
        occurrence = vehicle_full_df.count()
        logger.info(
            f"There are {occurrence} occurences in the full vehicle DataFrame.\n")

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
