import unittest
import utility as u


class CreateBatchTests(unittest.TestCase):
    ''' Tests for create_batch function '''

    def test_batch_size_n_num_batch_unprovide(self):
        with self.assertRaises(TypeError):
            batchs = u.create_batch([1, 2, 3])

    ##### test when provide batch_size #####
    # 1) when len(iterable) < batch_size
    def test_batch_size_iter_inf_batch_size(self):
        mylist = list(range(20))
        batchs = u.create_batch(mylist, batch_size=32)
        self.assertEqual(len(batchs), 1)
        self.assertEqual(len(batchs[0]), 20)

    # 2) when len(iterable) = batch_size
    def test_batch_size_iter_equal_batch_size(self):
        mylist = list(range(32))
        batchs = u.create_batch(mylist, batch_size=32)
        self.assertEqual(len(batchs), 1)
        self.assertEqual(len(batchs[0]), 32)

    # 3) when len(iterable) > batch_size and len(iterable) % batch_size = 0
    def test_batch_size_iter_sup_batch_size_modulo_zero(self):
        mylist = list(range(64))
        batchs = u.create_batch(mylist, batch_size=32)
        self.assertEqual(len(batchs), 2)
        self.assertEqual(len(batchs[0]), 32)
        self.assertEqual(len(batchs[1]), 32)

    # 4) when len(iterable) > batch_size and len(iterable) % batch_size != 0
    def test_batch_size_iter_sup_batch_size_modulo_dif_zero1(self):
        mylist = list(range(38))
        batchs = u.create_batch(mylist, batch_size=32)
        self.assertEqual(len(batchs), 2)
        self.assertEqual(len(batchs[0]), 32)
        self.assertEqual(len(batchs[1]), 6)

    def test_batch_size_iter_sup_batch_size_modulo_dif_zero2(self):
        mylist = list(range(76))
        batchs = u.create_batch(mylist, batch_size=32)
        self.assertEqual(len(batchs), 3)
        self.assertEqual(len(batchs[0]), 32)
        self.assertEqual(len(batchs[1]), 32)
        self.assertEqual(len(batchs[2]), 12)

    ##### test when provide num_batch #####
    def test_num_batch_1(self):
        batchs = u.create_batch(list(range(10)), num_batch=1)
        self.assertEqual(len(batchs), 1)
        self.assertEqual(len(batchs[0]), 10)

    def test_num_batch_2(self):
        batchs = u.create_batch(list(range(10)), num_batch=2)
        self.assertEqual(len(batchs), 2)
        self.assertEqual(len(batchs[0]), 5)
        self.assertEqual(len(batchs[1]), 5)

    def test_num_batch_3(self):
        batchs = u.create_batch(list(range(10)), num_batch=3)
        self.assertEqual(len(batchs), 3)
        self.assertEqual(len(batchs[0]), 4)
        self.assertEqual(len(batchs[1]), 3)
        self.assertEqual(len(batchs[2]), 3)

    # when len(iterable) == num_batch
    def test_num_batch_equal_iter(self):
        batchs = u.create_batch(list(range(10)), num_batch=10)
        self.assertEqual(len(batchs), 10)
        self.assertEqual([len(b) for b in batchs], [1] * 10)

    # when len(iterable) < num_batch
    def test_num_batch_sup_iter(self):
        with self.assertRaises(AssertionError):
            batchs = u.create_batch(list(range(10)), num_batch=12)


class CleanSentenceTest(unittest.TestCase):
    def test_remove_double_space(self):
        clean_sentence = u.clean_sentence('hello  world good')
        self.assertEqual(clean_sentence, 'hello world good')

    def test_remove_multiple_double_space(self):
        clean_sentence = u.clean_sentence('Hello  world good  morning')
        self.assertEqual(clean_sentence, 'Hello world good morning')

    def test_remove_border_space_left(self):
        clean_sentence = u.clean_sentence(' Hello world good morning')
        self.assertEqual(clean_sentence, 'Hello world good morning')

    def test_remove_border_space_right(self):
        clean_sentence = u.clean_sentence('Hello world good morning ')
        self.assertEqual(clean_sentence, 'Hello world good morning')

    def test_remove_border_space_left_n_right(self):
        clean_sentence = u.clean_sentence(' Hello world good morning ')
        self.assertEqual(clean_sentence, 'Hello world good morning')

    def test_remove_punctuation(self):
        clean_sentence = u.clean_sentence('hello world, good morning.')
        self.assertEqual(clean_sentence, 'hello world good morning')


class TransformToEmbeddings(unittest.TestCase):
    def test_convert_sentence(self):
        fake_emb = {'hello': [1, 2, 3], 'world': [4, 5, 6]}
        se = u.convert_to_emb('hello world', fake_emb)
        self.assertEqual(se.tolist(), [[1, 2, 3], [4, 5, 6]])

    def test_convert_sentences(self):
        fake_emb = {'hello': [1, 2, 3], 'world': [4, 5, 6], 'good': [7, 8, 9], 'morning': [10, 11, 12]}
        ses = u.to_emb(['hello world', 'good morning'], fake_emb)
        self.assertEqual([se.tolist() for se in ses], [[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]])


if __name__ == '__main__':
    unittest.main()
