import logging
import os
from pyspark import SparkContext

# Constants
SCRIPTS_DIR = os.path.abspath(os.path.dirname(__file__))
DATA_DIR = os.path.join(SCRIPTS_DIR, "..", "data")
GROCERY_DATA_FILE_PATH = os.path.join(DATA_DIR, "grocery_data.csv")
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


def create_spark_context(app_name="TMA_Market_Basket_Analysis"):
    """
    Create and return a SparkContext.

    Parameters
    ----------
    app_name : str, optional
        The name of the Spark application, by default "TMA_Market_Basket_Analysis".

    Returns
    -------
    SparkSession
        The Spark session object.

    Notes
    -----
    This function initializes a Spark session, which is the entry point for working with Spark functionality.
    """
    sc = SparkContext("local", app_name)
    return sc


def show_rdd(rdd, logger, max_rows=100, show_rows=20):
    """
    Show rows of a rdd with the option to limit the number of rows displayed.

    Parameters
    ----------
    df : rdd
        The rdd to be displayed.

    max_rows : int, optional
        The maximum number of rows to display. Default is 20.

    Returns
    -------
    None

    Notes
    -----
    This function displays rows of the input rdd. If the rdd contains more
    rows than the specified `max_rows`, it will limit the display to the first `show_rows`
    rows. If the rdd has fewer rows than `max_rows`, it will display all available
    rows without truncation.
    """
    if rdd.count() > max_rows:
        logger.info(rdd.take(show_rows))
    else:
        logger.info(rdd.collect())


def load_data(sc, logger, file_path):
    """
    Load data from CSV file into a Spark RDD.

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
    RDD
        RDD containing the data.

    Raises
    ------
    FileNotFoundError
        If the specified file does not exist.

    Notes
    -----
    This function reads data from a CSV file and loads it into a Spark RDD.

    Note: The default logging level is set to the value of the constant LOGGING_LEVEL.
    """
    try:
        rdd = sc.textFile(file_path)
        return rdd
    except Exception as e:
        if "Path does not exist" in str(e):
            logger.error(FILE_NOT_FOUND_MESSAGE.format(file_path))
            raise FileNotFoundError(f"File not found: {file_path}")

        logger.error(LOAD_DATA_ERROR_MESSAGE.format(str(e)))
        raise e


def cleanse(rdd, logger):
    try:
        cleansed_rdd = rdd.map(
            lambda x: [item.strip().lower() for item in x.split(',')])
        return cleansed_rdd
    except Exception as e:
        logger.error(f"An error occurred: {str(e)}")
        raise e


def find_transaction_with_most_items(rdd, logger):
    try:
        transaction_counts = rdd.map(lambda items: (items, len(items)))
        max_transaction = transaction_counts.max(key=lambda x: x[1])

        max_transaction_content, max_transaction_items = max_transaction
        return max_transaction_content, max_transaction_items
    except Exception as e:
        logger.error(f"An error occurred: {str(e)}")
        raise e


def count_unique_items(rdd, logger):
    try:
        unique_items = rdd.flatMap(lambda items: items).distinct()
        count = unique_items.count()
        return count
    except Exception as e:
        logger.error(f"An error occurred: {str(e)}")
        raise e


def top_n_item(rdd, n, logger):
    try:
        item_count = rdd.flatMap(lambda items: items).countByValue().items()
        sorted_item_count = sorted(
            item_count, key=(lambda x: x[1]), reverse=True)

        top_item = sorted_item_count[:n]
        total_transaction = rdd.count()
        top_items_with_percentage = [
            (item, count, (count / total_transaction)) for item, count in top_item]

        return top_items_with_percentage
    except Exception as e:
        logger.error(f"An error occurred: {str(e)}")
        raise e


def bottom_n_item(rdd, n, logger):
    try:
        item_count = rdd.flatMap(lambda items: items).countByValue().items()
        sorted_item_count = sorted(
            item_count, key=(lambda x: x[1]), reverse=True)

        bottom_item = sorted_item_count[-n:]
        total_transaction = rdd.count()
        bottom_items_with_percentage = [
            (item, count, (count / total_transaction)) for item, count in bottom_item]

        return bottom_items_with_percentage
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
    2. Creates a SparkContext for data processing.
    3. Loads grocery data from a CSV file and cleanse it from trailing spaces.
    ...
    5. Displays and logs the analysis results.
    6. Stops the SparkContext when processing is complete.
    """
    logger = configure_logging()
    sc = create_spark_context()

    try:
        grocery_rdd = load_data(sc, logger, GROCERY_DATA_FILE_PATH)

        cleansed_grocery_rdd = cleanse(grocery_rdd, logger)
        show_rdd(cleansed_grocery_rdd, logger)
        occurrence = cleansed_grocery_rdd.count()
        logger.info(f"There are {occurrence} transactions in the rdd.\n")

        most_groceries = find_transaction_with_most_items(
            cleansed_grocery_rdd, logger)
        logger.info(most_groceries)

        unique_groceries_count = count_unique_items(
            cleansed_grocery_rdd, logger)
        logger.info(
            f"There are {unique_groceries_count} unique items in the rdd.\n")

        top_20_items = top_n_item(cleansed_grocery_rdd, 20, logger)
        logger.info(top_20_items)

        bottom_20_items = bottom_n_item(cleansed_grocery_rdd, 20, logger)
        logger.info(bottom_20_items)
    except Exception as e:
        logger.error(f"An error occurred: {str(e)}")
        raise e
    finally:
        if sc is not None:
            sc.stop()


if __name__ == "__main__":
    main()
