import unittest
import logging
# ! FIXME: call from parent dir
from PySpark_program import load_data, create_spark_session


class TestDataLoading(unittest.TestCase):
    def test_file_not_found(self):
        logger = logging.getLogger(__name__)
        spark = create_spark_session()
        try:
            load_data(
                spark=spark, logger=logger, file_path=r"C:\Everything\SUSS\DE\ICT337\TMA\data\nonexistent_file.csv"
            )
        except Exception as e:
            self.assertIsInstance(e, FileNotFoundError)

    def test_successful_load(self):
        logger = logging.getLogger(__name__)
        spark = create_spark_session()
        try:
            flights = load_data(spark=spark, logger=logger)
            self.assertIsNotNone(spark)
            self.assertIsNotNone(flights)
        except Exception as e:
            self.fail(f"An error occurred: {str(e)}")


if __name__ == '__main__':
    unittest.main()
