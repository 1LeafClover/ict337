import logging
import os
from pyspark import SparkContext

# Constants
SCRIPTS_DIR = os.path.abspath(os.path.dirname(__file__))
DATA_DIR = os.path.join(SCRIPTS_DIR, "..", "data")
MOVIE_RATINGS_DATA_FILE_PATH = os.path.join(DATA_DIR, "mov_rating.dat")
MOVIE_ITEMS_DATA_FILE_PATH = os.path.join(DATA_DIR, "mov_item.dat")
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


def create_spark_context(app_name="TMA_Market_Basket_Analysis"):
    """
    Create and return a SparkContext.

    Parameters
    ----------
    app_name : str, optional
        The name of the Spark application. Default is "TMA_Market_Basket_Analysis".

    Returns
    -------
    SparkContext
        An instance of SparkContext.

    Notes
    -----
    This function initializes a SparkContext, which is the entry point for Spark operations.
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
        The maximum number of rows to display. Default is 100.

    show_rows : int, optional
        The rows to display if records is above max rows. Default is 20.

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


def load_data(sc, logger, file_path, delimiter=","):
    """
    Load data from CSV file into a Spark RDD.

    Parameters
    ----------
    spark : SparkContext
        The Spark context.
    logger : object
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
        rdd = rdd.map(lambda line: line.split(delimiter))
        return rdd
    except Exception as e:
        if "Path does not exist" in str(e):
            logger.error(FILE_NOT_FOUND_MESSAGE.format(file_path))
            raise FileNotFoundError(f"File not found: {file_path}")
        logger.error(LOAD_DATA_ERROR_MESSAGE.format(str(e)))
        raise e


def count_unique_by(rdd, column_index, logger):
    try:
        unique_reviewers_count = rdd.map(
            lambda line: line[column_index-1]).distinct().count()
        return unique_reviewers_count
    except Exception as e:
        logger.error(f"An error occurred: {str(e)}")
        raise e


def top_n_reviewers(rdd, column_index, n, logger):
    try:
        review_count = rdd.map(
            lambda line: line[column_index-1]).countByValue()

        sorted_review_count = sorted(
            review_count.items(), key=lambda item: item[1], reverse=True)
        return sorted_review_count[:n]
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
    3. Loads grocery data from a CSV file and cleanses it from trailing spaces.
    4. Displays and logs various statistics and analysis results, such as transaction counts, unique item counts, top and bottom items, support, etc.
    5. Computes confidence values for item pairs and displays the top and bottom item pairs with confidence.
    6. Stops the SparkContext when processing is complete.
    """
    logger = configure_logging()
    sc = create_spark_context()

    try:
        mov_review_rdd = load_data(
            sc, logger, MOVIE_RATINGS_DATA_FILE_PATH, "\t")
        show_rdd(mov_review_rdd, logger)
        rating_count = mov_review_rdd.count()
        logger.info(f"There are {rating_count} rating records in the rdd.\n")

        count_unique_reviewers = count_unique_by(mov_review_rdd, 1, logger)
        logger.info(count_unique_reviewers)

        count_unique_movies = count_unique_by(mov_review_rdd, 2, logger)
        logger.info(count_unique_movies)

        top_10_reviewers = top_n_reviewers(mov_review_rdd, 1, 10, logger)
        logger.info(top_10_reviewers)

        top_10_movie_id = top_n_reviewers(mov_review_rdd, 2, 10, logger)
        top_10_movie_id_rdd = sc.parallelize(
            top_10_movie_id).map(lambda x: (x[0], x[1]))

        mov_item_rdd = load_data(
            sc, logger, MOVIE_ITEMS_DATA_FILE_PATH, "|")

        top_10_movie_rdd = top_10_movie_id_rdd.join(mov_item_rdd)
        top_10_movie_rdd = top_10_movie_rdd.map(
            lambda item: (item[0], item[1][1], item[1][0]))
        top_10_movie_rdd = top_10_movie_rdd.sortBy(
            lambda item: item[2], ascending=False)
        show_rdd(top_10_movie_rdd, logger)
    except Exception as e:
        logger.error(f"An error occurred: {str(e)}")
        raise e
    finally:
        if sc is not None:
            sc.stop()


if __name__ == "__main__":
    main()
