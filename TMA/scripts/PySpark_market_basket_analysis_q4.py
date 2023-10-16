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


def load_data(sc, logger, file_path):
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
        return rdd
    except Exception as e:
        if "Path does not exist" in str(e):
            logger.error(FILE_NOT_FOUND_MESSAGE.format(file_path))
            raise FileNotFoundError(f"File not found: {file_path}")
        logger.error(LOAD_DATA_ERROR_MESSAGE.format(str(e)))
        raise e


def cleanse(rdd, logger):
    """
    Cleanse data in an RDD by stripping whitespace and converting to lowercase.

    Parameters
    ----------
    rdd : RDD
        The input RDD containing data to be cleansed.

    logger : object
        Logger object for logging messages.

    Returns
    -------
    RDD
        RDD containing the cleansed data.

    Raises
    ------
    Exception
        If an error occurs during the cleansing process, an exception is raised.

    Notes
    -----
    This function takes an RDD as input and performs the following cleansing operations on each row:

    1. Remove leading and trailing whitespace.
    2. Convert all text to lowercase.

    The cleansed data is returned as an RDD.
    """
    try:
        cleansed_rdd = rdd.map(
            lambda x: [item.strip().lower() for item in x.split(',')])
        return cleansed_rdd
    except Exception as e:
        logger.error(f"An error occurred: {str(e)}")
        raise e


def transaction_with_most_items(cleanse_rdd, logger):
    """
    Find the transaction with the most items in an RDD.

    Parameters
    ----------
    cleanse_rdd : RDD
        The RDD containing cleansed transaction data.

    logger : object
        Logger object for logging messages.

    Returns
    -------
    tuple
        A tuple containing the content of the transaction with the most items and the number of items in that transaction.

    Raises
    ------
    Exception
        If an error occurs during the process, an exception is raised.

    Notes
    -----
    This function takes an RDD of cleansed transaction data and finds the transaction with the highest number of items.
    It returns a tuple containing the content of the transaction and the count of items in that transaction.
    """
    try:
        transaction_item_count = cleanse_rdd.map(
            lambda items: (items, len(items)))
        max_transaction = transaction_item_count.max(key=lambda x: x[1])

        max_transaction_content, max_transaction_items = max_transaction
        return max_transaction_content, max_transaction_items
    except Exception as e:
        logger.error(f"An error occurred: {str(e)}")
        raise e


def count_unique_items(cleanse_rdd, logger):
    """
    Count the number of unique items in an RDD of transaction data.

    Parameters
    ----------
    cleanse_rdd : RDD
        The RDD containing cleansed transaction data.

    logger : object
        Logger object for logging messages.

    Returns
    -------
    int
        The count of unique items in the RDD.

    Raises
    ------
    Exception
        If an error occurs during the process, an exception is raised.

    Notes
    -----
    This function takes an RDD of cleansed transaction data, flattens it to extract individual items, and then counts the number
    of unique items in the RDD.
    """
    try:
        unique_items = cleanse_rdd.flatMap(lambda items: items).distinct()
        count = unique_items.count()
        return count
    except Exception as e:
        logger.error(f"An error occurred: {str(e)}")
        raise e


def top_n_item_with_percentage(cleanse_rdd, n, logger):
    """
    Calculate the top N most frequent items along with their occurrence percentages.

    Parameters
    ----------
    cleanse_rdd : RDD
        The RDD containing cleansed transaction data.

    n : int
        The number of top items to retrieve.

    logger : object
        Logger object for logging messages.

    Returns
    -------
    list
        A list of tuples containing the top N items, their occurrence counts, and occurrence percentages.

    Raises
    ------
    Exception
        If an error occurs during the process, an exception is raised.

    Notes
    -----
    This function takes an RDD of cleansed transaction data, calculates the most frequent N items, and computes their
    occurrence percentages relative to the total number of transactions.
    """
    try:
        item = cleanse_rdd.flatMap(lambda items: items)
        item_count = item.countByValue().items()
        sorted_item_count = sorted(
            item_count, key=(lambda x: x[1]), reverse=True)

        top_item = sorted_item_count[:n]
        total_transaction = item.count()
        top_items_with_percentage = [
            (item, count, ((count / total_transaction) * 100)) for item, count in top_item]
        return top_items_with_percentage
    except Exception as e:
        logger.error(f"An error occurred: {str(e)}")
        raise e


def bottom_n_item_with_percentage(cleanse_rdd, n, logger):
    """
    Calculate the bottom N least frequent items along with their occurrence percentages.

    Parameters
    ----------
    cleanse_rdd : RDD
        The RDD containing cleansed transaction data.

    n : int
        The number of bottom items to retrieve.

    logger : object
        Logger object for logging messages.

    Returns
    -------
    list
        A list of tuples containing the bottom N items, their occurrence counts, and occurrence percentages.

    Raises
    ------
    Exception
        If an error occurs during the process, an exception is raised.

    Notes
    -----
    This function takes an RDD of cleansed transaction data, calculates the least frequent N items, and computes their
    occurrence percentages relative to the total number of transactions.
    """
    try:
        item = cleanse_rdd.flatMap(lambda items: items)
        item_count = item.countByValue().items()
        sorted_item_count = sorted(
            item_count, key=(lambda x: x[1]), reverse=False)

        bottom_item = sorted_item_count[:n]
        total_transaction = item.count()
        bottom_items_with_percentage = [
            (item, count, ((count / total_transaction) * 100)) for item, count in bottom_item]
        return bottom_items_with_percentage
    except Exception as e:
        logger.error(f"An error occurred: {str(e)}")
        raise e


def add_index(cleanse_rdd, logger):
    """
    Add an index to each transaction of the RDD.

    Parameters
    ----------
    cleanse_rdd : RDD
        The RDD containing cleansed data.

    logger : object
        Logger object for logging messages.

    Returns
    -------
    RDD
        RDD containing the data with added indices.

    Raises
    ------
    Exception
        If an error occurs during the process, an exception is raised.

    Notes
    -----
    This function takes an RDD and adds a unique index to each transaction. The resulting RDD contains the original data
    along with the corresponding index.
    """
    try:
        indexed_rdd = cleanse_rdd.zipWithIndex()
        return indexed_rdd
    except Exception as e:
        logger.error(f"An error occurred: {str(e)}")
        raise e


def generate_combinations(indexed_rdd, logger):
    """
    Generate combinations of two grocery items within each transaction.

    Parameters
    ----------
    indexed_rdd : RDD
        The RDD containing data with added indices.

    logger : object
        Logger object for logging messages.

    Returns
    -------
    RDD
        RDD containing combinations of two items and their corresponding transaction indices.

    Raises
    ------
    Exception
        If an error occurs during the process, an exception is raised.

    Notes
    -----
    This function takes an RDD with added indices and generates all possible combinations of two grocery items within each transaction.
    The resulting RDD contains the item pairs and their corresponding transaction indices.
    """
    try:
        item_combinations_rdd = indexed_rdd.flatMap(lambda transaction: [(
            (item1, item2), transaction[1])
            for item1 in transaction[0] for item2 in transaction[0]
            # Ensure item1 is less than item2 (in alphabetical order) to avoid duplicate pairs
            if item1 < item2
        ])
        return item_combinations_rdd
    except Exception as e:
        logger.error(f"An error occurred: {str(e)}")
        raise e


def association(combinations_rdd, logger):
    """
    Generate associations between grocery items and their transaction indices.

    Parameters
    ----------
    combinations_rdd : RDD
        RDD containing combinations of grocery items and their corresponding transaction indices.

    logger : object
        Logger object for logging messages.

    Returns
    -------
    RDD
        RDD containing associations between item combinations and their transaction indices.

    Raises
    ------
    Exception
        If an error occurs during the process, an exception is raised.

    Notes
    -----
    This function takes an RDD with item combinations and their transaction indices and generates associations.
    The resulting RDD contains the associations between grocery item combinations and their transaction indices.
    """
    try:
        transaction_indices = combinations_rdd.groupByKey().map(
            lambda x: (x[0], list(x[-1])))
        return transaction_indices
    except Exception as e:
        logger.error(f"An error occurred: {str(e)}")
        raise e


def item_pair_counts(association_rdd, logger):
    """
    Calculate the counts of item pairs in association data.

    Parameters
    ----------
    association_rdd : RDD
        RDD containing associations between item combinations and their transaction indices.

    logger : object
        Logger object for logging messages.

    Returns
    -------
    RDD
        RDD containing the counts of item pairs sorted by count in descending order.

    Raises
    ------
    Exception
        If an error occurs during the process, an exception is raised.

    Notes
    -----
    This function takes an RDD with item associations and calculates the counts of item pairs.
    The resulting RDD contains the counts of item pairs, sorted in descending order by count.
    """
    try:
        item_count_rdd = association_rdd.map(lambda x: (x[0], len(x[1])))
        sorted_item_count_rdd = item_count_rdd.sortBy(
            lambda x: x[1], ascending=False)
        return sorted_item_count_rdd
    except Exception as e:
        logger.error(f"An error occurred: {str(e)}")
        raise e


def support(item_pair_counts_rdd, logger):
    """
    Calculate the support of item pairs in association data.

    Parameters
    ----------
    item_pair_counts_rdd : RDD
        RDD containing the counts of item pairs sorted by count.

    logger : object
        Logger object for logging messages.

    Returns
    -------
    RDD
        RDD containing the support of item pairs.

    Raises
    ------
    Exception
        If an error occurs during the process, an exception is raised.

    Notes
    -----
    This function calculates the support of item pairs based on their counts.
    """
    try:
        total_records = item_pair_counts_rdd.map(lambda x: x[1]).sum()
        support_rdd = item_pair_counts_rdd.map(lambda x: (
            (x[0], (x[1], x[1] / total_records))))
        return support_rdd
    except Exception as e:
        logger.error(f"An error occurred: {str(e)}")
        raise e


def top_n_item_pair_with_support(support_rdd, n, logger):
    """
    Get the top N item pairs with the highest occurrence count.

    Parameters
    ----------
    support_rdd : RDD
        RDD containing item pairs and their support as a percentage of total records.

    n : int
        The number of top item pairs to retrieve.

    logger : object
        Logger object for logging messages.

    Returns
    -------
    list
        A list of the top N item pairs with the occurrence count and support value, sorted by occurence count.

    Raises
    ------
    Exception
        If an error occurs during the process, an exception is raised.

    Notes
    -----
    This function retrieves the top N item pairs with the highest occurrence count.
    """
    try:
        sorted_items = support_rdd.sortBy(
            lambda x: x[1][0], ascending=False).take(n)
        return sorted_items
    except Exception as e:
        logger.error(f"An error occurred: {str(e)}")
        raise e


def bottom_n_item_pair_with_support(support_rdd, n, logger):
    """
    Get the bottom N item pairs with the lowest occurrence count.

    Parameters
    ----------
    support_rdd : RDD
        RDD containing item pairs and their support as a percentage of total records.

    n : int
        The number of bottom item pairs to retrieve.

    logger : object
        Logger object for logging messages.

    Returns
    -------
    list
        A list of the bottom N item pairs with the occurrence count and support value, sorted by occurence count.

    Raises
    ------
    Exception
        If an error occurs during the process, an exception is raised.

    Notes
    -----
    This function retrieves the bottom N item pairs with the lowest occurrence count.
    """
    try:
        sorted_items = support_rdd.sortBy(
            lambda x: x[1][0], ascending=True).take(n)
        return sorted_items
    except Exception as e:
        logger.error(f"An error occurred: {str(e)}")
        raise e


def term_frequency(cleansed_rdd, logger):
    """
    Calculate the term frequency of items in transactions.

    Parameters
    ----------
    cleansed_rdd : RDD
        RDD containing cleansed transaction data.

    logger : object
        Logger object for logging messages.

    Returns
    -------
    RDD
        RDD containing term frequency information for each item in the transactions.

    Raises
    ------
    Exception
        If an error occurs during the process, an exception is raised.

    Notes
    -----
    This function calculates the term frequency of items in each transaction within the RDD.
    Term frequency is the number of times each item appears in each transaction.
    The result is an RDD containing (item, transaction index, term frequency) tuples.
    """
    try:
        indexed_transactions_rdd = cleansed_rdd.zipWithIndex()
        term_frequencies = indexed_transactions_rdd.flatMap(lambda x: [
            (item, x[1], x[0].count(item)) for item in x[0]
        ])
        return term_frequencies
    except Exception as e:
        logger.error(f"An error occurred: {str(e)}")
        raise e


def generate_permutations(term_frequency_rdd, logger):
    """
    Generate all permutations of item pairs within each transaction.

    Parameters
    ----------
    term_frequency_rdd : RDD
        RDD containing term frequency information for items in transactions.

    logger : object
        Logger object for logging messages.

    Returns
    -------
    RDD
        RDD containing sorted item permutations within each transaction.

    Raises
    ------
    Exception
        If an error occurs during the process, an exception is raised.

    Notes
    -----
    This function generates all possible permutations of item pairs within each transaction in the RDD.
    It groups the transactions and calculates permutations for each transaction separately.
    The output RDD contains sorted item permutations for each transaction.
    """
    try:
        def generate_pairs(transaction):
            item_list = list(transaction)
            item_pairs = []
            # Iterate through items in the transaction
            for i in range(len(item_list)):
                for j in range(i + 1, len(item_list)):
                    item1 = item_list[i][0]
                    item2 = item_list[j][0]
                    transaction_index = item_list[i][1]
                    # Create pairs for item1 and item2, as well as their reverse order
                    item_pairs.append(((item1, item2), transaction_index))
                    item_pairs.append(((item2, item1), transaction_index))
            return item_pairs

        grouped_items = term_frequency_rdd.groupBy(lambda x: x[1])
        # Generate permutations within each group
        item_permutations = grouped_items.flatMap(
            lambda x: generate_pairs(list(x[1])))
        sorted_item_permutations = item_permutations.sortBy(lambda x: x[0])
        return sorted_item_permutations
    except Exception as e:
        logger.error(f"An error occurred: {str(e)}")
        raise e


def item_count(cleansed_rdd, logger):
    """
    Count the occurrences of each item in the RDD.

    Parameters
    ----------
    cleansed_rdd : RDD
        RDD containing cleansed transaction data.
    logger : object
        Logger object for logging messages.

    Returns
    -------
    dict
        A dictionary where keys are items, and values are their counts.

    Raises
    ------
    Exception
        If any error occurs during the counting process.

    Notes
    -----
    This function counts the occurrences of each unique item in the provided RDD.
    The result is returned as a dictionary where item names are keys and their counts are values.
    """
    try:
        # Transform each transaction into a list of (item, 1) pairs, then reduce to count occurrences.
        item_frequencies_rdd = cleansed_rdd.flatMap(lambda transaction: [(
            item, 1) for item in transaction]).reduceByKey(lambda a, b: a + b)
        sorted_item_counts_rdd = item_frequencies_rdd.sortBy(
            lambda x: x[1], ascending=False)

        sorted_item_counts_dict = dict(sorted_item_counts_rdd.collect())
        return sorted_item_counts_dict
    except Exception as e:
        logger.error(f"An error occurred: {str(e)}")
        raise e


def confidence(frequency_xy_rdd, frequency_x_dict, logger):
    """
    Calculate the confidence of item pairs based on their frequencies.

    Parameters
    ----------
    frequency_xy_rdd : RDD
        RDD containing item pair frequencies.
    frequency_x_dict : dict
        A dictionary of item frequencies.
    logger : object
        Logger object for logging messages.

    Returns
    -------
    RDD
        RDD containing item pairs with their corresponding confidence values.

    Raises
    ------
    Exception
        If any error occurs during the confidence calculation.

    Notes
    -----
    This function calculates the confidence of item pairs based on their antecedents.
    It uses a dictionary of item frequencies (frequency_x_dict) to compute the confidence
    as the ratio of the frequency of the item pair (x, y) to the frequency of item x.
    """
    try:
        confidence_rdd = frequency_xy_rdd.map(lambda x: (
            x[0], (x[1], frequency_x_dict.get(x[0][0]), x[1] / frequency_x_dict.get(x[0][0]))))
        return confidence_rdd
    except Exception as e:
        logger.error(f"An error occurred: {str(e)}")
        raise e


def top_n_item_pair_with_confidence(confidence_rdd, n, logger):
    """
    Get the top N item pairs with the highest occurrence count of item X, and occurrence count of item pair X and Y.

    Parameters
    ----------
    confidence_rdd : RDD
        RDD containing item pairs with their confidence values.
    n : int
        The number of top item pairs to retrieve.
    logger : object
        Logger object for logging messages.

    Returns
    -------
    list
        A list of the top N item pairs with the highest occurrence count of item X, and occurrence count of item pair X and Y.

    Raises
    ------
    Exception
        If any error occurs during the retrieval of top item pairs.

    Notes
    -----
    This function sorts the item pairs in confidence_rdd by occurrence count of item X,
    and occurrence count of item pair X and Y in descending order.
    """
    try:
        sorted_items = confidence_rdd.sortBy(
            lambda x: (x[1][1], x[1][0]), ascending=False).take(n)
        return sorted_items
    except Exception as e:
        logger.error(f"An error occurred: {str(e)}")
        raise e


def bottom_n_item_pair_with_confidence(confidence_rdd, n, logger):
    """
    Get the top N item pairs with the lowest occurrence count of item X, and occurrence count of item pair X and Y.

    Parameters
    ----------
    confidence_rdd : RDD
        RDD containing item pairs with their confidence values.
    n : int
        The number of top item pairs to retrieve.
    logger : object
        Logger object for logging messages.

    Returns
    -------
    list
        A list of the top N item pairs with the lowest occurrence count of item X, and occurrence count of item pair X and Y.

    Raises
    ------
    Exception
        If any error occurs during the retrieval of top item pairs.

    Notes
    -----
    This function sorts the item pairs in confidence_rdd by occurrence count of item X,
    and occurrence count of item pair X and Y in ascending order.
    """
    try:
        sorted_items = confidence_rdd.sortBy(
            lambda x: (x[1][1], x[1][0]), ascending=True).take(n)
        return sorted_items
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
        grocery_rdd = load_data(sc, logger, GROCERY_DATA_FILE_PATH)
        show_rdd(grocery_rdd, logger)
        occurrence = grocery_rdd.count()
        logger.info(f"There are {occurrence} transactions in the rdd.\n")

        cleansed_grocery_rdd = cleanse(grocery_rdd, logger)
        show_rdd(cleansed_grocery_rdd, logger)
        occurrence = cleansed_grocery_rdd.count()
        logger.info(f"There are {occurrence} transactions in the rdd.\n")

        most_groceries = transaction_with_most_items(
            cleansed_grocery_rdd, logger)
        logger.info(most_groceries)

        unique_groceries_count = count_unique_items(
            cleansed_grocery_rdd, logger)
        logger.info(
            f"There are {unique_groceries_count} unique items in the rdd.\n")

        top_20_items = top_n_item_with_percentage(
            cleansed_grocery_rdd, 20, logger)
        logger.info(top_20_items)

        bottom_20_items = bottom_n_item_with_percentage(
            cleansed_grocery_rdd, 20, logger)
        logger.info(bottom_20_items)

        indexed_grocery_rdd = add_index(cleansed_grocery_rdd, logger)
        show_rdd(indexed_grocery_rdd, logger)
        occurrence = indexed_grocery_rdd.count()
        logger.info(f"There are {occurrence} number of records.\n")

        combination_2item = generate_combinations(indexed_grocery_rdd, logger)
        show_rdd(combination_2item, logger)
        occurrence = combination_2item.count()
        logger.info(f"There are {occurrence} number of records.\n")

        associated_transaction = association(combination_2item, logger)
        show_rdd(associated_transaction, logger)
        occurrence = associated_transaction.count()
        logger.info(f"There are {occurrence} number of records.\n")

        sorted_associated_count = item_pair_counts(
            associated_transaction, logger)
        show_rdd(sorted_associated_count, logger)
        occurrence = sorted_associated_count.count()
        logger.info(f"There are {occurrence} number of records.\n")

        item_pair_support = support(sorted_associated_count, logger)
        show_rdd(item_pair_support, logger)
        total_records = sorted_associated_count.map(lambda x: x[1]).sum()
        logger.info(f"There are {total_records} total number of records.\n")

        top_20_item_pairs = top_n_item_pair_with_support(
            item_pair_support, 20, logger)
        logger.info(top_20_item_pairs)

        bottom_20_item_pairs = bottom_n_item_pair_with_support(
            item_pair_support, 20, logger)
        logger.info(bottom_20_item_pairs)

        term_frequency_list = term_frequency(cleansed_grocery_rdd, logger)
        show_rdd(term_frequency_list, logger)
        occurrence = term_frequency_list.count()
        logger.info(f"There are {occurrence} number of records.\n")

        permutation_2item = generate_permutations(term_frequency_list, logger)
        show_rdd(permutation_2item, logger)
        occurrence = permutation_2item.count()
        logger.info(f"There are {occurrence} number of records.\n")

        associated_transaction = association(permutation_2item, logger)
        show_rdd(associated_transaction, logger)
        occurrence = associated_transaction.count()
        logger.info(f"There are {occurrence} number of records.\n")

        frequency_xy = item_pair_counts(
            associated_transaction, logger)
        show_rdd(frequency_xy, logger)
        occurrence = frequency_xy.count()
        logger.info(f"There are {occurrence} number of records.\n")

        frequency_x = item_count(cleansed_grocery_rdd, logger)
        logger.info(frequency_x)
        occurrence = len(frequency_x)
        logger.info(f"There are {occurrence} number of records.\n")

        confidence_rdd = confidence(frequency_xy, frequency_x, logger)
        show_rdd(confidence_rdd, logger)
        occurrence = confidence_rdd.count()
        logger.info(f"There are {occurrence} number of records.\n")

        top_20_item_pairs_confidence = top_n_item_pair_with_confidence(
            confidence_rdd, 20, logger)
        logger.info(top_20_item_pairs_confidence)

        bottom_20_item_pairs_confidence = bottom_n_item_pair_with_confidence(
            confidence_rdd, 20, logger)
        logger.info(bottom_20_item_pairs_confidence)
    except Exception as e:
        logger.error(f"An error occurred: {str(e)}")
        raise e
    finally:
        if sc is not None:
            sc.stop()


if __name__ == "__main__":
    main()
