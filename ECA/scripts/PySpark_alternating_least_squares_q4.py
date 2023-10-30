import logging
import os
from pyspark import SparkContext

# Constants
SCRIPTS_DIR = os.path.abspath(os.path.dirname(__file__))
DATA_DIR = os.path.join(SCRIPTS_DIR, "..", "data")
MOVIE_RATINGS_DATA_FILE_PATH = os.path.join(DATA_DIR, "mov_rating.dat")
MOVIE_ITEMS_DATA_FILE_PATH = os.path.join(DATA_DIR, "mov_item.dat")
MOVIE_GENRE_DATA_FILE_PATH = os.path.join(DATA_DIR, "mov_genre.dat")
MOVIE_USER_DATA_FILE_PATH = os.path.join(DATA_DIR, "mov_user.dat")
TOP3_MOVIE_BY_GENRE_OUTPUT_PATH = os.path.join(DATA_DIR, "top3_mov_by_genre")
TOP30_MOVIE_BY_AGE_GROUP_OUTPUT_PATH = os.path.join(
    DATA_DIR, "top30_mov_by_age_group")
TOP3_SUMMER_MOVIE_BY_GENRE_OUTPUT_PATH = os.path.join(
    DATA_DIR, "top3_summer_mov_by_age_group")
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


def create_spark_context(app_name="ECA_Alternating_Least_Squares"):
    """
    Create and return a SparkContext.

    Parameters
    ----------
    app_name : str, optional
        The name of the Spark application. Default is "ECA_Alternating_Least_Squares".

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


# TODO: add part b) functions


def assign_age_group(age, logger):
    try:
        age_groups = {
            (0, 6): "[0-6]",
            (7, 12): "(6-12]",
            (13, 18): "(12-18]",
            (19, 30): "(18-30]",
            (31, 50): "(30-50]",
            (51, float("inf")): "50+"
        }

        age = int(age)
        for age_range, group in age_groups.items():
            if age_range[0] <= age <= age_range[1]:
                return group
        return 'Unknown'
    except Exception as e:
        logger.error(f"An error occurred: {str(e)}")
        raise e


# def top_n_movies_per_genre():


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

        movie_release_years = mov_item_rdd.map(lambda line: line[2][-4:])
        release_year_counts = movie_release_years.countByValue()
        sorted_release_year_counts = dict(
            sorted(release_year_counts.items(), key=lambda x: x[1], reverse=True))
        logger.info(sorted_release_year_counts)

        movie_release_years = movie_release_years.filter(
            lambda year: year != '')
        min_year = movie_release_years.min()
        max_year = movie_release_years.max()
        print(f"Range of Movie Release Years: {min_year} to {max_year}")

        top_year = list(sorted_release_year_counts)[0]
        movies_in_max_year = mov_item_rdd.filter(
            lambda line: line[2][-4:] == top_year)

        # Split the movies by genre
        movies_with_genre = movies_in_max_year.map(lambda line: (
            line[0], (line[1], line[5:])))

        mov_genre_rdd = load_data(
            sc, logger, MOVIE_GENRE_DATA_FILE_PATH, "|")
        genre_mapping = mov_genre_rdd.map(
            lambda genre: (genre[1], genre[0])).collectAsMap()

        movies_with_genre = movies_with_genre.map(lambda record: ([genre_mapping[str(
            index)] for index, value in enumerate(record[1][1]) if value == "1"], record[0], record[1][0]))

        movies_with_genre = movies_with_genre.map(
            lambda x: (x[1], (x[0], x[2])))
        mov_ratings = mov_review_rdd.map(lambda x: (x[1], (x[2])))
        movie_genre_with_rating = mov_ratings.join(movies_with_genre)

        movie_genre_with_avg_rating = movie_genre_with_rating.groupBy(lambda x: (tuple(x[1][1][0]), x[0], x[1][1][1])).map(
            lambda x: (x[0], len(x[1]), sum(int(item[1][0]) for item in x[1]) / len(x[1])))

        sorted_movie_genre_with_avg_rating = movie_genre_with_avg_rating.sortBy(
            lambda x: (x[0][0], -x[1]), ascending=[True, False]).map(lambda x: (x[0], x[2]))
        show_rdd(sorted_movie_genre_with_avg_rating, logger)

        grouped_movie_genre_with_avg_rating = sorted_movie_genre_with_avg_rating.groupBy(
            lambda x: x[0][0])

        # Get the top three values for each groups
        top3_movie_by_genre = grouped_movie_genre_with_avg_rating.flatMap(
            lambda key_values: (list(key_values[1])[:3],))
        show_rdd(top3_movie_by_genre, logger)

        # FIXME: winutils not compatible
        # top3_movie_by_genre.saveAsTextFile(TOP3_MOVIE_BY_GENRE_OUTPUT_PATH)
        # ''.join(sorted(input(glob(TOP3_MOVIE_BY_GENRE_OUTPUT_PATH + "/part-0000*"))))

        mov_user_rdd = load_data(
            sc, logger, MOVIE_USER_DATA_FILE_PATH, "|")
        mov_user_rdd = mov_user_rdd.map(lambda x: (x[0], (x[1])))

        mov_ratings = mov_review_rdd.map(lambda x: (x[0], (x[1], x[2])))

        mov_review_with_user = mov_ratings.join(mov_user_rdd).map(
            lambda x: (x[1][0][0], (x[1][0][1], x[1][1])))

        mov_name = mov_item_rdd.map(lambda x: (x[0], (x[1])))

        mov_names_review_with_user = mov_review_with_user.join(mov_name).map(
            lambda x: (x[0], x[1][1], x[1][0][0], x[1][0][1]))

        mov_names_review_with_user = mov_names_review_with_user.map(lambda x: (
            (assign_age_group(x[3], logger), x[0], x[1]), (x[3], float(x[2]))))

        movie_names_with_avg_rating = mov_names_review_with_user.groupBy(lambda x: (x[0])).map(
            lambda x: (len(x[1]), list(x[1])[0][0], [list(x[1])[0][1][0] for item in x[1]], sum(int(item[1][1]) for item in x[1]) / len(x[1])))

        movie_names_with_avg_rating = movie_names_with_avg_rating.map(
            lambda x: (x[1][0], (x[0], x[1][1], x[1][2], x[2], x[3])))

        # Calculate the total count of movies per age group
        total_movie_count_by_age_group = movie_names_with_avg_rating.map(
            lambda x: (x[0], 1)).reduceByKey(lambda a, b: a + b)

        # Join the original RDD with the calculated total counts
        total_movie_count_with_avg_rating = movie_names_with_avg_rating.join(total_movie_count_by_age_group).map(
            lambda x: (x[1][1], x[0], x[1][0][1], x[1][0][2], x[1][0][3], x[1][0][4], x[1][0][0]))

        sorted_total_movie_count_with_avg_rating = total_movie_count_with_avg_rating.sortBy(lambda x: (
            -int(x[6]), -float(x[5])), ascending=[False, False])

        grouped_total_movie_count_with_avg_rating = sorted_total_movie_count_with_avg_rating.groupBy(
            lambda x: x[1])

        # Take the top 30 values for each age group based on the number of reviews.
        top30_movie_by_age_group = grouped_total_movie_count_with_avg_rating.flatMap(
            lambda x: list(x[1])[:30])

        top30_movie_by_age_group = top30_movie_by_age_group.map(
            lambda x: (x[0], x[1], x[4], x[2], x[3], x[5]))
        show_rdd(top30_movie_by_age_group, logger)

        # FIXME: winutils not compatible
        # top30_movie_by_age_group.saveAsTextFile(TOP30_MOVIE_BY_AGE_GROUP_OUTPUT_PATH)
        # ''.join(sorted(input(glob(TOP30_MOVIE_BY_AGE_GROUP_OUTPUT_PATH + "/part-0000*"))))

        summer = ["may", "jun", "jul"]
        movies_in_summer = mov_item_rdd.filter(
            lambda line: line[2][3:-5].lower() in summer)

        # Split the movies by genre
        summer_movies_with_genre = movies_in_summer.map(lambda line: (
            line[0], (line[1], line[5:])))

        mov_genre_rdd = load_data(
            sc, logger, MOVIE_GENRE_DATA_FILE_PATH, "|")
        genre_mapping = mov_genre_rdd.map(
            lambda genre: (genre[1], genre[0])).collectAsMap()

        summer_movies_with_genre = summer_movies_with_genre.map(lambda record: ([genre_mapping[str(
            index)] for index, value in enumerate(record[1][1]) if value == "1"], record[0], record[1][0]))

        summer_movies_with_genre = summer_movies_with_genre.map(
            lambda x: (x[1], (x[0], x[2])))
        mov_ratings = mov_review_rdd.map(lambda x: (x[1], (x[2])))
        summer_movie_genre_with_rating = mov_ratings.join(
            summer_movies_with_genre)

        summer_movie_genre_with_avg_rating = summer_movie_genre_with_rating.groupBy(lambda x: (tuple(x[1][1][0]), x[0], x[1][1][1])).map(
            lambda x: (x[0], len(x[1]), sum(int(item[1][0]) for item in x[1]) / len(x[1])))

        sorted_summer_movie_genre_with_avg_rating = summer_movie_genre_with_avg_rating.sortBy(
            lambda x: (x[0][0], -x[1]), ascending=[True, False]).map(lambda x: (x[0], x[2]))

        grouped_summer_movie_genre_with_avg_rating = sorted_summer_movie_genre_with_avg_rating.groupBy(
            lambda x: x[0][0])

        # Get the top three values for each groups
        top3_summer_movie_by_genre = grouped_summer_movie_genre_with_avg_rating.flatMap(
            lambda key_values: (list(key_values[1])[:3],))
        show_rdd(top3_summer_movie_by_genre, logger)

        # FIXME: winutils not compatible
        # top3_summer_movie_by_genre.saveAsTextFile(TOP3_SUMMER_MOVIE_BY_GENRE_OUTPUT_PATH)
        # ''.join(sorted(input(glob(TOP3_SUMMER_MOVIE_BY_GENRE_OUTPUT_PATH + "/part-0000*"))))

    except Exception as e:
        logger.error(f"An error occurred: {str(e)}")
        raise e
    finally:
        if sc is not None:
            sc.stop()


if __name__ == "__main__":
    main()
