import logging
from pyspark.sql import SparkSession
from pyspark.sql.functions import col

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
    """
    logging.basicConfig(level=LOGGING_LEVEL)
    return logging.getLogger(__name__)


def create_spark_session(app_name="TMA_DataProcessing"):
    """
    Create and return a Spark session.

    Parameters
    ----------
    app_name : str, optional
        The name of the Spark application, by default "TMA_DataProcessing".

    Returns
    -------
    SparkSession
        The Spark session object.
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
    This function configures the logging settings, including the logging level, and returns a logger
    object that can be used for logging messages within the application.

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


def process_missing_data(logger, flights_df):
    """Process and analyze the loaded data.

    Parameters
    ----------
    logger : object
        Logger object for logging messages.
    spark : object
        Spark session.
    flights_df : DataFrame
        DataFrame containing the data.

    Raises
    ------
    Exception
        If an error occurs during data processing.
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
    This function initializes the Spark session, loads data, processes and analyzes it,
    and then stops the Spark session. Any exceptions raised during execution are logged.

    Usage
    -----
    This script is intended to be executed as the main entry point.
    """
    logger = configure_logging()
    spark = create_spark_session()
    try:
        flights_data_frame = load_data(spark, logger)
        process_missing_data(logger, flights_data_frame)
    except Exception as e:
        logger.error(f"An error occurred: {str(e)}")
        raise e
    finally:
        if spark is not None:
            spark.stop()


if __name__ == "__main__":
    main()
