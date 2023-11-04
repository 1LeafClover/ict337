import logging
import os
from pyspark import SparkContext
from pyspark.mllib.recommendation import ALS

# Constants
SCRIPTS_DIR = os.path.abspath(os.path.dirname(__file__))
DATA_DIR = os.path.join(SCRIPTS_DIR, "..", "data")
MOVIE_RATINGS_DATA_FILE_PATH = os.path.join(DATA_DIR, "mov_rating.dat")
MOVIE_ITEMS_DATA_FILE_PATH = os.path.join(DATA_DIR, "mov_item.dat")
MOVIE_GENRE_DATA_FILE_PATH = os.path.join(DATA_DIR, "mov_genre.dat")
MOVIE_USER_DATA_FILE_PATH = os.path.join(DATA_DIR, "mov_user.dat")
TOP_3_MOVIE_BY_GENRE_OUTPUT_PATH = os.path.join(DATA_DIR, "top_3_mov_by_genre")
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

    Note: The default logging level is set to the value of the constant LOGGING_LEVEL.
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
    Load data from a text file into an RDD using SparkContext.

    Parameters
    ----------
    sc : SparkContext
        The SparkContext for creating RDDs.
    logger : object
        Logger object for logging messages.
    file_path : str
        The path to the text file to load.
    delimiter : str, optional
        The delimiter used to split data in the text file. Default is a comma (',').

    Returns
    -------
    RDD
        An RDD containing the loaded data.

    Raises
    ------
    FileNotFoundError
        If the specified file does not exist.
    Exception
        If an error occurs while loading the data.

    Notes
    -----
    This function uses the SparkContext 'sc' to load data from a text file into an RDD.
    The 'delimiter' parameter specifies how the data in the text file is split into columns.

    The 'logger' object is used for logging messages, and any error that occurs during the loading process is logged and raised as an exception.

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


def count_unique_by(loaded_rdd, column_index, logger):
    """
    Count the number of unique values in an RDD based on a specified column index.

    Parameters
    ----------
    loaded_rdd : RDD
        The input RDD containing data.
    column_index : int
        The index of the column to consider for counting unique values.
    logger : object
        Logger object for logging messages.

    Returns
    -------
    int
        The count of unique values in the specified column.

    Raises
    ------
    Exception
        If an error occurs while counting unique values.

    Notes
    -----
    This function processes an RDD to count the number of unique values in a specified column.
    The 'column_index' parameter determines which column to consider, and uniqueness is determined based on distinct values.

    The 'logger' object is used for logging messages, and any error that occurs during the counting process is logged and raised as an exception.

    Note: The default logging level is set to the value of the constant LOGGING_LEVEL.
    """
    try:
        unique_reviewers_count = loaded_rdd.map(
            lambda line: line[column_index-1]).distinct().count()
        return unique_reviewers_count
    except Exception as e:
        logger.error(f"An error occurred: {str(e)}")
        raise e


def top_n_counts_by(loaded_rdd, column_index, n, logger):
    """
    Get the top 'n' counts of values in an RDD based on a specified column index.

    Parameters
    ----------
    loaded_rdd : RDD
        The input RDD containing data.
    column_index : int
        The index of the column to consider for counting values.
    n : int
        The number of top counts to retrieve.
    logger : object
        Logger object for logging messages.

    Returns
    -------
    list of (value, count) tuples
        A list of the top 'n' counts and their corresponding values, sorted in descending order.

    Raises
    ------
    Exception
        If an error occurs while retrieving the top 'n' counts.

    Notes
    -----
    This function processes an RDD to retrieve the top 'n' counts and their corresponding values in a specified column.
    The 'column_index' parameter determines which column to consider, and the values are counted using the `countByValue()` method.

    The 'logger' object is used for logging messages, and any error that occurs during the retrieval process is logged and raised as an exception.

    Note: The default logging level is set to the value of the constant LOGGING_LEVEL.
    """
    try:
        review_count = loaded_rdd.map(
            lambda line: line[column_index-1]).countByValue()

        sorted_review_count = sorted(
            review_count.items(), key=lambda item: item[1], reverse=True)
        return sorted_review_count[:n]
    except Exception as e:
        logger.error(f"An error occurred: {str(e)}")
        raise e


def date_parse(movie_item_rdd, column_index, logger):
    """
    Extracts the release years from a specified column of an RDD.

    Parameters
    ----------
    movie_item_rdd : RDD
        The input RDD containing movie data.
    column_index : int
        The index of the column to extract release years from.
    logger : object
        Logger object for logging messages.

    Returns
    -------
    RDD
        An RDD containing the extracted movie release years.

    Raises
    ------
    Exception
        If an error occurs during processing, it is logged and raised.

    Notes
    -----
    This function processes the input movie RDD to extract the release years from a specified column.
    It extracts the last 4 digits from strings that end with a 4-digit year and returns an RDD of release years.

    The 'logger' object is used for logging messages, and any error that occurs during the date parse process is logged and raised as an exception.

    Note: The default logging level is set to the value of the constant LOGGING_LEVEL.
    """
    try:
        # Extract last 4 string from the date column that ends with a 4 digit year
        movie_release_years = movie_item_rdd.map(
            lambda line: line[column_index][-4:])
        return movie_release_years
    except Exception as e:
        logger.error(f"An error occurred: {str(e)}")
        raise e


def count_movie_release_years(movie_release_years, logger):
    """
    Counts and sorts movie release years from an RDD.

    Parameters
    ----------
    movie_release_years : RDD
        An RDD containing movie release years.
    logger : object
        Logger object for logging messages.

    Returns
    -------
    dict
        A dictionary with movie release years as keys and their counts as values,
        sorted in descending order of counts.

    Raises
    ------
    Exception
        If an error occurs during counting and sorting, it is logged and raised.

    Notes
    -----
    This function takes an RDD containing movie release years, counts the occurrences of each release year,
    and returns a dictionary with release years as keys and their counts as values.
    The dictionary is sorted in descending order based on counts.

    The 'logger' object is used for logging messages, and any error that occurs during the counting and sorting process is logged and raised as an exception.

    Note: The default logging level is set to the value of the constant LOGGING_LEVEL.
    """
    try:
        release_year_counts = movie_release_years.countByValue()
        sorted_release_year_counts = dict(
            sorted(release_year_counts.items(), key=lambda x: x[1], reverse=True))
        return sorted_release_year_counts
    except Exception as e:
        logger.error(f"An error occurred: {str(e)}")
        raise e


def movie_release_year_range(movie_release_years, logger):
    """
    Calculate the range of movie release years from an RDD.

    Parameters
    ----------
    movie_release_years : RDD
        An RDD containing movie release years.
    logger : object
        Logger object for logging messages.

    Returns
    -------
    None
        This function does not return a value, but it logs the range of movie release years.

    Raises
    ------
    Exception
        If an error occurs during the calculating, it is logged and raised.

    Notes
    -----
    This function takes an RDD containing movie release years, filters out any empty values, and calculates the
    range of release years (from the minimum to the maximum). It logs the calculated range using the provided 'logger' object.

    The 'logger' object is used for logging messages, and any error that occurs during the calculation process is logged and raised as an exception.

    Note: The default logging level is set to the value of the constant LOGGING_LEVEL.
    """
    try:
        movie_release_years = movie_release_years.filter(
            lambda year: year != '')
        min_year = movie_release_years.min()
        max_year = movie_release_years.max()
        logger.info(f"Range of Movie Release Years: {min_year} to {max_year}")
    except Exception as e:
        logger.error(f"An error occurred: {str(e)}")
        raise e


def extract_genres(movies_with_genre, mov_genre_rdd, logger):
    """
    Extract and map movie genres in the RDD 'movies_with_genre'.

    Parameters
    ----------
    movies_with_genre : RDD
        An RDD containing movie records with genre indicators.
    mov_genre_rdd : RDD
        An RDD containing genre mapping data.
    logger : object
        Logger object for logging messages.

    Returns
    -------
    RDD
        An RDD with movie records, where genres are mapped to their names.

    Raises
    ------
    Exception
        If an error occurs during extracting, it is logged and raised.

    Notes
    -----
    This function performs the following steps:
    1. Extracts genre mapping information from 'mov_genre_rdd' and creates a dictionary for mapping genre indicators to genre names.
    2. Extracts and maps movie genres in 'movies_with_genre' RDD. For each movie record in 'movies_with_genre', this step involves:
        a. Creating a list of genres by identifying '1' values in the record's genre indicator.
        b. Mapping these genre indicators to their corresponding genre names using the genre_mapping dictionary.
        c. Associating the movie ID and its title with the list of genres.
    3. Returns the transformed RDD with genres mapped to their names.

    The 'logger' object is used for logging messages, and any error that occurs during the extract process is logged and raised as an exception.

    Note: The default logging level is set to the value of the constant LOGGING_LEVEL.
    """
    try:
        genre_mapping = mov_genre_rdd.map(
            lambda genre: (genre[1], genre[0])).collectAsMap()

        # Extract and map movie genres in movies_with_genre RDD
        movies_with_genre = movies_with_genre.map(lambda record: ([genre_mapping[str(
            index)] for index, value in enumerate(record[1][1]) if value == "1"], record[0], record[1][0]))

        transformed_movies_with_genre = movies_with_genre.map(
            lambda x: (x[1], (x[0], x[2])))
        return transformed_movies_with_genre
    except Exception as e:
        logger.error(f"An error occurred: {str(e)}")
        raise e


def write_top_n_reviewed_movies_by_genre(movie_genre_with_avg_rating, genre_rdd, n, genre_output_path, logger):
    """
    Write the top 'n' reviewed movies for each genre to separate text files.

    Parameters
    ----------
    movie_genre_with_avg_rating : RDD
        An RDD containing movie records with genres and average ratings.
    genre_rdd : RDD
        An RDD containing genre information.
    n : int
        The number of top reviewed movies to retrieve for each genre.
    genre_output_path : str
        The base path for saving genre-specific text files.
    logger : object
        Logger object for logging messages.

    Returns
    -------
    None

    Raises
    ------
    Exception
        If an error occurs during writing, it is logged and raised.

    Notes
    -----
    This function processes movie data and genre information to write the top 'n' reviewed movies for each genre to separate text files.
    It follows these steps:
    1. Loop through each genre in the 'genre_rdd'.
    2. Filter movie records by the current genre and create an RDD of movies specific to that genre.
    3. Sort the movies by the number of reviews in descending order.
    4. Select the top 'n' reviewed movies for the current genre.
    5. Sort the selected movies by their average rating in descending order.
    6. Generate a file path specific to the current genre using 'genre_output_path'.
    7. Save the sorted movies to a text file with the genre-specific file path.

    The 'logger' object is used for logging messages, and any error that occurs during the process is logged and raised as an exception.

    Note: The default logging level is set to the value of the constant LOGGING_LEVEL.
    """
    try:
        # Loop through each genre
        for genre_name, genre_id in genre_rdd.collect():
            # Filter movies by the current genre
            filtered_movie_genre_with_avg_rating = movie_genre_with_avg_rating.filter(
                lambda x: genre_name in x[0][0]).map(
                lambda x: (x[0][0], x[0][1], x[0][2], x[1], x[2]))

            # Sort movies by number of reviews in descending order
            sorted_movie_genre_with_avg_rating = filtered_movie_genre_with_avg_rating.sortBy(
                lambda x: x[3], ascending=False).map(lambda x: (x[0], x[1], x[2], x[4]))

            top_n_movie_by_genre = sorted_movie_genre_with_avg_rating.zipWithIndex().filter(
                lambda x: x[1] < n).keys()

            # Sort movies by average rating in descending order
            sorted_top_n_movie_by_genre = top_n_movie_by_genre.sortBy(
                lambda x: x[3], ascending=False)

            genre_output_path_for_genre = f"{genre_output_path}\{genre_name}.txt"

            sorted_top_n_movie_by_genre.coalesce(
                1).saveAsTextFile(genre_output_path_for_genre)
    except Exception as e:
        logger.error(f"An error occurred: {str(e)}")
        raise e


# TODO: add part c) functions
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


# TODO: add part d) functions


# TODO: add part e) functions


def add_new_user_profiles(existing_reviews_rdd, new_user_reviews, sc, logger):
    try:
        new_user_reviews_rdd = sc.parallelize(new_user_reviews)
        updated_reviews = new_user_reviews_rdd.union(existing_reviews_rdd)
        return updated_reviews
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
        # Load tab delimited file
        mov_review_rdd = load_data(
            sc, logger, MOVIE_RATINGS_DATA_FILE_PATH, "\t")
        show_rdd(mov_review_rdd, logger)
        rating_count = mov_review_rdd.count()
        logger.info(f"There are {rating_count} rating records in the rdd.\n")

        count_unique_reviewers = count_unique_by(mov_review_rdd, 1, logger)
        logger.info(
            f"There are {count_unique_reviewers} unique number of reviewers.")

        count_unique_mov = count_unique_by(mov_review_rdd, 2, logger)
        logger.info(
            f"There are {count_unique_mov} unique number of movies reviewed.")

        top_10_reviewers = top_n_counts_by(mov_review_rdd, 1, 10, logger)
        logger.info(top_10_reviewers)

        top_10_mov_reviewed = top_n_counts_by(mov_review_rdd, 2, 10, logger)
        top_10_mov_reviewed_rdd = sc.parallelize(
            top_10_mov_reviewed).map(lambda x: (x[0], x[1]))

        mov_item_rdd = load_data(
            sc, logger, MOVIE_ITEMS_DATA_FILE_PATH, "|")

        top_10_mov_reviewed_rdd = top_10_mov_reviewed_rdd.join(
            mov_item_rdd)
        top_10_mov_reviewed_rdd = top_10_mov_reviewed_rdd.map(
            lambda item: (item[0], item[1][1], item[1][0]))
        top_10_mov_reviewed_rdd = top_10_mov_reviewed_rdd.sortBy(
            lambda item: item[2], ascending=False)
        show_rdd(top_10_mov_reviewed_rdd, logger)

        mov_release_years = date_parse(mov_item_rdd, 2, logger)

        sorted_mov_release_years_count = count_movie_release_years(
            mov_release_years, logger)
        logger.info(sorted_mov_release_years_count)

        movie_release_year_range(mov_release_years, logger)

        top_year = list(sorted_mov_release_years_count)[0]
        mov_in_max_year = mov_item_rdd.filter(
            lambda line: line[2][-4:] == top_year)

        # Select movie id, movie name and list of genres
        mov_with_genre = mov_in_max_year.map(lambda line: (
            line[0], (line[1], line[5:])))

        mov_genre_rdd = load_data(
            sc, logger, MOVIE_GENRE_DATA_FILE_PATH, "|")

        mov_with_genre = extract_genres(mov_with_genre, mov_genre_rdd, logger)

        mov_ratings = mov_review_rdd.map(lambda x: (x[1], (x[2])))
        mov_genre_with_rating = mov_ratings.join(mov_with_genre)

        # Compute for average rating based on genre, movie id and movie name
        mov_genre_with_avg_rating = mov_genre_with_rating.groupBy(lambda x: (tuple(x[1][1][0]), x[0], x[1][1][1])).map(
            lambda x: (x[0], len(x[1]), sum(int(item[1][0]) for item in x[1]) / len(x[1])))

        # TODO: Uncomment before prod
        # write_top_n_reviewed_movies_by_genre(
        #     mov_genre_with_avg_rating, mov_genre_rdd, 3, TOP_3_MOVIE_BY_GENRE_OUTPUT_PATH, logger)

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

        # # FIXME: winutils not compatible
        # # top30_movie_by_age_group.saveAsTextFile(TOP30_MOVIE_BY_AGE_GROUP_OUTPUT_PATH)
        # # ''.join(sorted(input(glob(TOP30_MOVIE_BY_AGE_GROUP_OUTPUT_PATH + "/part-0000*"))))

        # summer = ["may", "jun", "jul"]
        # movies_in_summer = mov_item_rdd.filter(
        #     lambda line: line[2][3:-5].lower() in summer)

        # # Split the movies by genre
        # summer_movies_with_genre = movies_in_summer.map(lambda line: (
        #     line[0], (line[1], line[5:])))

        # summer_movies_with_genre = summer_movies_with_genre.map(lambda record: ([genre_mapping[str(
        #     index)] for index, value in enumerate(record[1][1]) if value == "1"], record[0], record[1][0]))

        # summer_movies_with_genre = summer_movies_with_genre.map(
        #     lambda x: (x[1], (x[0], x[2])))
        # mov_ratings = mov_review_rdd.map(lambda x: (x[1], (x[2])))
        # summer_movie_genre_with_rating = mov_ratings.join(
        #     summer_movies_with_genre)

        # summer_movie_genre_with_avg_rating = summer_movie_genre_with_rating.groupBy(lambda x: (tuple(x[1][1][0]), x[0], x[1][1][1])).map(
        #     lambda x: (x[0], len(x[1]), sum(int(item[1][0]) for item in x[1]) / len(x[1])))

        # sorted_summer_movie_genre_with_avg_rating = summer_movie_genre_with_avg_rating.sortBy(
        #     lambda x: (x[0][0], -x[1]), ascending=[True, False]).map(lambda x: (x[0], x[2]))

        # grouped_summer_movie_genre_with_avg_rating = sorted_summer_movie_genre_with_avg_rating.groupBy(
        #     lambda x: x[0][0])

        # # Get the top three values for each groups
        # top3_summer_movie_by_genre = grouped_summer_movie_genre_with_avg_rating.flatMap(
        #     lambda key_values: (list(key_values[1])[:3],))

        # top3_summer_movie_by_genre = top3_summer_movie_by_genre.flatMap(
        #     lambda row: row)
        # show_rdd(top3_summer_movie_by_genre, logger)

        # # FIXME: winutils not compatible
        # # top3_summer_movie_by_genre.saveAsTextFile(TOP3_SUMMER_MOVIE_BY_GENRE_OUTPUT_PATH)
        # # ''.join(sorted(input(glob(TOP3_SUMMER_MOVIE_BY_GENRE_OUTPUT_PATH + "/part-0000*"))))

        # mov_ratings = mov_review_rdd.map(lambda x: (x[0], (x[1], x[2])))

        # mov_user_rdd = load_data(
        #     sc, logger, MOVIE_USER_DATA_FILE_PATH, "|")

        # mov_user_rdd = mov_user_rdd.map(lambda x: (x[0], x[3]))

        # mov_review_with_user = mov_ratings.join(mov_user_rdd).map(
        #     lambda x: (x[1][0][0], (x[1][0][1], x[1][1])))

        # mov_item_rdd = mov_item_rdd.map(lambda line: (
        #     line[0], (line[1], line[5:])))

        # movies_with_genre = mov_item_rdd.map(lambda record: ([genre_mapping[str(
        #     index)] for index, value in enumerate(record[1][1]) if value == "1"], record[0], record[1][0]))

        # movies_with_genre = movies_with_genre.map(
        #     lambda x: (x[1], (x[0], x[2])))

        # movie_genre_with_rating = mov_review_with_user.join(movies_with_genre)

        # movie_genre_with_avg_rating = movie_genre_with_rating.groupBy(lambda x: (tuple(x[1][1][0]), x[1][0][1], x[0], x[1][1][1])).map(
        #     lambda x: (x[0], len(x[1]), sum(int(item[1][0][0]) for item in x[1]) / len(x[1])))

        # sorted_movie_genre_with_avg_rating = movie_genre_with_avg_rating.sortBy(
        #     lambda x: (x[0][0], -x[1]), ascending=[True, False]).map(lambda x: (x[0][1], x[0][0], x[0][2], x[0][3], x[2]))

        # grouped_movie_genre_with_avg_rating = sorted_movie_genre_with_avg_rating.groupBy(
        #     lambda x: x[1])

        # # Get the top three values for each groups
        # top3_movie_by_genre = grouped_movie_genre_with_avg_rating.flatMap(
        #     lambda key_values: (list(key_values[1])[:3],))

        # top3_movie_by_genre = top3_movie_by_genre.flatMap(lambda row: row)
        # show_rdd(top3_movie_by_genre, logger)

        # filtered_data = top3_movie_by_genre.filter(
        #     lambda row: row[0].lower() == "administrator" and "action" in row[1][0].lower())

        # top3_admin_action_mov = filtered_data.take(3)
        # logger.info(top3_admin_action_mov)

        # new_user_profiles = [
        #     [0, 50, 5, 881250949],
        #     [0, 172, 5, 881250949],
        #     [0, 181, 5, 881250949]
        # ]

        # updated_reviews = add_new_user_profiles(
        #     mov_review_rdd, new_user_profiles, sc, logger)
        # show_rdd(updated_reviews, logger)
        # logger.info(updated_reviews.count())

        # # Selecting user/reviewer identifier, movie identifier, rating
        # ratings = updated_reviews.filter(lambda row: len(row) == 4).map(
        #     lambda x: (int(x[0]), int(x[1]), float(x[2])))

        # # Define ALS model parameters
        # rank = 20
        # num_iterations = 15

        # mov_ratings_model = ALS.train(ratings, rank, num_iterations)
        # show_rdd(mov_ratings_model.userFeatures(), logger)
        # logger.info(mov_ratings_model.userFeatures().count())

        # user_id = 0
        # num_recommendations = 10

        # # Use the model to recommend movies for the user
        # mov_recommendations = mov_ratings_model.recommendProducts(
        #     user_id, num_recommendations)
        # print(mov_recommendations)
        # logger.info(len(mov_recommendations))
    except Exception as e:
        logger.error(f"An error occurred: {str(e)}")
    finally:
        if sc is not None:
            sc.stop()


if __name__ == "__main__":
    main()
