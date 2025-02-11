import unittest
import csv
import numpy

class MyTestCase(unittest.TestCase):
    def test_can_read_command_index_table(self):
        index_file = "command_index.csv"
        data = csv.reader(open(index_file))
        print(f"\n\n{list(data)}")

    def test_can_read_command_test_examples(self):
        examples_file = "command_test_inputs.csv"
        data = csv.reader(open(examples_file))



if __name__ == '__main__':
    unittest.main()
