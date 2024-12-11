import unittest
import pandas as pd
from unittest.mock import MagicMock

from preprocessing.preprocessing import utils


class TestBaseTextCategorizationDataset(unittest.TestCase):
    def test__get_num_train_samples(self):
        """
        we want to test the class BaseTextCategorizationDataset
        we use a mock which will return a value for the not implemented methods
        then with this mocked value, we can test other methods
        """
        # we instantiate a BaseTextCategorizationDataset object with batch_size = 20 and train_ratio = 0.8
        base = utils.BaseTextCategorizationDataset(20, 0.8)
        # we mock _get_num_samples to return the value 100
        base._get_num_samples = MagicMock(return_value=100)
        # we assert that _get_num_train_samples will return 100 * train_ratio = 80
        self.assertEqual(base._get_num_train_samples(), 80)

    def test__get_num_train_batches(self):
        """
        same idea as what we did to test _get_num_train_samples
        """
        # TODO: CODE HERE
        base = utils.BaseTextCategorizationDataset(20, 0.8)
        base._get_num_samples = MagicMock(return_value=100)
        self.assertEqual(base._get_num_train_batches(), 4)

    def test__get_num_test_batches(self):
        # TODO: CODE HERE
        base = utils.BaseTextCategorizationDataset(20, 0.8)
        base._get_num_samples = MagicMock(return_value=100)
        self.assertEqual(base._get_num_test_batches(), 1)

    def test_get_index_to_label_map(self):
        # TODO: CODE HERE
        base = utils.BaseTextCategorizationDataset(20, 0.8)
        base._get_label_list = MagicMock(return_value=["cat", "dog", "bird"])
        self.assertEqual(base.get_index_to_label_map(), {0: "cat", 1: "dog", 2: "bird"})

    def test_index_to_label_and_label_to_index_are_identity(self):
        # TODO: CODE HERE
        base = utils.BaseTextCategorizationDataset(20, 0.8)
        base._get_label_list = MagicMock(return_value=["cat", "dog", "bird"])

        index_to_label = base.get_index_to_label_map()
        label_to_index = base.get_label_to_index_map()

        for index, label in index_to_label.items():
            self.assertEqual(label_to_index[label], index)

        for label, index in label_to_index.items():
            self.assertEqual(index_to_label[index], label)


    def test_to_indexes(self):
        # TODO: CODE HERE
        base = utils.BaseTextCategorizationDataset(20, 0.8)
        base._get_label_list = MagicMock(return_value=["cat", "dog", "bird"])
        self.assertEqual(base.to_indexes(base._get_label_list()), [0, 1, 2])


class TestLocalTextCategorizationDataset(unittest.TestCase):
    def test_load_dataset_returns_expected_data(self):
        # we mock pandas read_csv to return a fixed dataframe
        pd.read_csv = MagicMock(return_value=pd.DataFrame({
            'post_id': ['id_1', 'id_2'],
            'tag_name': ['tag_a', 'tag_b'],
            'tag_id': [1, 2],
            'tag_position': [0, 1],
            'title': ['title_1', 'title_2']
        }))
        # we instantiate a LocalTextCategorizationDataset (it'll use the mocked read_csv), and we load dataset
        dataset = utils.LocalTextCategorizationDataset.load_dataset("fake_path", 1)
        # we expect the data after loading to be like this
        expected = pd.DataFrame({
            'post_id': ['id_1'],
            'tag_name': ['tag_a'],
            'tag_id': [1],
            'tag_position': [0],
            'title': ['title_1']
        })
        # we confirm that the dataset and what we expected to be are the same thing
        pd.testing.assert_frame_equal(dataset, expected)

    def test__get_num_samples_is_correct(self):
        # TODO: CODE HERE
        pd.read_csv = MagicMock(return_value=pd.DataFrame({
            'post_id': ['id_1', 'id_2', 'id_3', 'id_4', 'id_5'],
            'tag_name': ['tag_a', 'tag_a', 'tag_a', 'tag_a', 'tag_a'],
            'tag_id': [1, 1, 2, 2, 2],
            'tag_position': [0, 0, 0, 0, 0],
            'title': ['title_1', 'title_2', 'title_3', 'title_4', 'title_5']
        }))

        dataset = utils.LocalTextCategorizationDataset("fake_path", 1, min_samples_per_label=1)
        self.assertEqual(dataset._get_num_samples(), 5)

    def test_get_train_batch_returns_expected_shape(self):
        # TODO: CODE HERE
        pd.read_csv = MagicMock(return_value=pd.DataFrame({
            'post_id': ['id_1', 'id_2', 'id_3', 'id_4', 'id_5'],
            'tag_name': ['tag_a', 'tag_a', 'tag_a', 'tag_a', 'tag_a'],
            'tag_id': [1, 1, 2, 3, 4],
            'tag_position': [0, 0, 0, 0, 0],
            'title': ['title_1', 'title_2', 'title_3', 'title_4', 'title_5']
        }))

        dataset = utils.LocalTextCategorizationDataset("fake_path", 1, min_samples_per_label=1)

        train_sequence = dataset.get_train_sequence()

        x_batch, y_batch = train_sequence[0]

        self.assertEqual(x_batch.shape, (1,))
        self.assertEqual(y_batch.shape, (1, len(dataset._label_list)))

    def test_get_test_batch_returns_expected_shape(self):
        # TODO: CODE HERE
        pd.read_csv = MagicMock(return_value=pd.DataFrame({
            'post_id': ['id_1', 'id_2', 'id_3', 'id_4', 'id_5'],
            'tag_name': ['tag_a', 'tag_a', 'tag_a', 'tag_a', 'tag_a'],
            'tag_id': [1, 1, 2, 3, 4],
            'tag_position': [0, 0, 0, 0, 0],
            'title': ['title_1', 'title_2', 'title_3', 'title_4', 'title_5']
        }))

        dataset = utils.LocalTextCategorizationDataset("fake_path", 1, min_samples_per_label=1)

        test_sequence = dataset.get_test_sequence()

        x_batch, y_batch = test_sequence[0]

        self.assertEqual(x_batch.shape, (1,))
        self.assertEqual(y_batch.shape, (1, len(dataset._label_list)))

    def test_get_train_batch_raises_assertion_error(self):
        # TODO: CODE HERE
        pd.read_csv = MagicMock(return_value=pd.DataFrame({
            'post_id': ['id_1', 'id_2', 'id_3', 'id_4', 'id_5'],
            'tag_name': ['tag_a', 'tag_a', 'tag_a', 'tag_a', 'tag_a'],
            'tag_id': [1, 1, 1, 1, 1],
            'tag_position': [0, 0, 0, 0, 0],
            'title': ['title_1', 'title_2', 'title_3', 'title_4', 'title_5']
        }))

        with self.assertRaises(AssertionError):
            _ = utils.LocalTextCategorizationDataset("fake_path", 3)


if __name__ == "__main__":
    unittest.main()
