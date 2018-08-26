import unittest
import numpy as np
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

  def test_to_batch(self):
    a, b = [1, 2, 3, 4], [5, 6, 7, 8, 9]
    ab, bb = u.to_batch(a, b, batch_size=2)
    self.assertEqual(len(ab), 2)
    self.assertEqual(sum([len(el) for el in ab]), 4)
    self.assertEqual(ab[0], [1, 2])
    self.assertEqual(ab[1], [3, 4])
    self.assertEqual(len(bb), 3)
    self.assertEqual(sum([len(el) for el in bb]), 5)
    self.assertEqual(bb[0], [5, 6])
    self.assertEqual(bb[1], [7, 8])
    self.assertEqual(bb[2], [9])


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


class TransformToEmbeddingsTest(unittest.TestCase):
  def test_convert_sentence(self):
    fake_emb = {'hello': [1, 2, 3], 'world': [4, 5, 6]}
    se = u.convert_to_emb('hello world', fake_emb)
    self.assertEqual(se.tolist(), [[1, 2, 3], [4, 5, 6]])

  def test_convert_sentences(self):
    fake_emb = {'hello': [1, 2, 3], 'world': [4, 5, 6], 'good': [7, 8, 9], 'morning': [10, 11, 12]}
    ses = u.to_emb(['hello world', 'good morning'], fake_emb)
    self.assertEqual([se.tolist() for se in ses], [[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]])


class PadDataTest(unittest.TestCase):
  def test_pad_with_zeros(self):
    t = [np.ones((2, 300)), np.ones((5, 300))]
    pt, sl, max_sl = u.pad_data(t)
    self.assertEqual(len(pt), 2)
    self.assertEqual(sl, [1, 4])
    self.assertEqual(max_sl, 5)
    self.assertEqual(pt[0].shape, (5, 300))
    self.assertEqual(pt[1].shape, (5, 300))
    self.assertTrue(np.array_equal(pt[0][:2,:], np.ones((2, 300))))
    self.assertTrue(np.array_equal(pt[0][2:,:], np.zeros((3, 300))))
    self.assertTrue(np.array_equal(pt[1], np.ones((5, 300))))

  def test_pad_with_list(self):
    t = [np.ones((2, 300)), np.ones((5, 300))]
    pt, sl, max_sl = u.pad_data(t, pad_with=[7]*300)
    self.assertEqual(len(pt), 2)
    self.assertEqual(sl, [1, 4])
    self.assertEqual(max_sl, 5)
    self.assertEqual(pt[0].shape, (5, 300))
    self.assertEqual(pt[1].shape, (5, 300))
    self.assertTrue(np.array_equal(pt[1], np.ones((5, 300))))
    self.assertTrue(np.array_equal(pt[0][:2,:], np.ones((2, 300))))
    self.assertTrue(np.array_equal(pt[0][2:,:], np.ones((3, 300)) * 7))

  def test_pad_with_array(self):
    t = [np.ones((2, 300)), np.ones((5, 300))]
    pt, sl, max_sl = u.pad_data(t, pad_with=np.array([7]*300))
    self.assertEqual(len(pt), 2)
    self.assertEqual(sl, [1, 4])
    self.assertEqual(max_sl, 5)
    self.assertEqual(pt[0].shape, (5, 300))
    self.assertEqual(pt[1].shape, (5, 300))
    self.assertTrue(np.array_equal(pt[1], np.ones((5, 300))))
    self.assertTrue(np.array_equal(pt[0][:2,:], np.ones((2, 300))))
    self.assertTrue(np.array_equal(pt[0][2:,:], np.ones((3, 300)) * 7))


class ShuffleDataTest(unittest.TestCase):
  def test_one_list(self):
    mylist = [1, 2, 3, 4, 5]
    res = u.shuffle_data(mylist)
    self.assertEqual(len(res), 1)
    self.assertEqual(len(res[0]), len(mylist))
    self.assertEqual(len(set(mylist) & set(res[0])), len(mylist))

  def test_two_list(self):
    list1, list2 = [1, 2, 3], [4, 5, 6]
    res = u.shuffle_data(list1, list2)
    self.assertEqual(len(res), 2)
    self.assertEqual(len(res[0]), len(list1))
    self.assertEqual(len(res[1]), len(list1))
    self.assertEqual(len(set(list1) & set(res[0])), len(list1))
    self.assertEqual(len(set(list2) & set(res[1])), len(list2))

  def test_multiple_list(self):
    list1, list2, list3 = [1, 2, 3], [4, 5, 6], [7, 8, 9]
    res = u.shuffle_data(list1, list2, list3)
    self.assertEqual(len(res), 3)
    self.assertEqual(len(res[0]), len(list1))
    self.assertEqual(len(res[1]), len(list2))
    self.assertEqual(len(res[2]), len(list3))
    self.assertEqual(len(set(list1) & set(res[0])), len(list1))
    self.assertEqual(len(set(list2) & set(res[1])), len(list2))
    self.assertEqual(len(set(list3) & set(res[2])), len(list3))


class EncodingTest(unittest.TestCase):
  def test_encode_labels(self):
    labels = ['A', 'B', 'C', 'B']
    ol, num_class = u.encode_labels(labels)
    self.assertEqual(num_class, 3)
    self.assertTrue(isinstance(ol, np.ndarray))
    self.assertEqual(ol.shape, (4, 3))
    self.assertEqual(len(set(ol[0]) & set([1, 0, 0])), 2)
    self.assertEqual(len(set(ol[0]) & set([1, 2, 0])), 2)
    self.assertEqual(len(set(ol[1]) & set([1, 0, 0])), 2)
    self.assertEqual(len(set(ol[1]) & set([1, 2, 0])), 2)
    self.assertEqual(len(set(ol[2]) & set([1, 0, 0])), 2)
    self.assertEqual(len(set(ol[2]) & set([1, 2, 0])), 2)
    self.assertEqual(len(set(ol[3]) & set([1, 0, 0])), 2)
    self.assertEqual(len(set(ol[3]) & set([1, 2, 0])), 2)
    self.assertEqual(ol[1].tolist(), ol[3].tolist())

  def test_one_hot_encoding(self):
    labels = ['A', 'B', 'C', 'B']
    l2o = u.one_hot_encoding(labels)
    self.assertTrue(isinstance(l2o, dict))
    self.assertEqual(len(l2o), 3)
    self.assertEqual(len(set(l2o.keys()) & set(['A', 'B', 'C'])), 3)
    self.assertEqual(len(set(l2o['A']) & set([1, 0, 0])), 2)
    self.assertEqual(len(set(l2o['A']) & set([1, 2, 0])), 2)
    self.assertEqual(len(set(l2o['B']) & set([1, 0, 0])), 2)
    self.assertEqual(len(set(l2o['B']) & set([1, 2, 0])), 2)
    self.assertEqual(len(set(l2o['C']) & set([1, 0, 0])), 2)
    self.assertEqual(len(set(l2o['C']) & set([1, 2, 0])), 2)


class GetVocabTest(unittest.TestCase):
  def test_get_vocabulary(self):
    sources = ['my name is', 'how are you', 'you live here', 'how is he', 'écrire', 'éCrire']
    tmp = [e for el in sources for e in el.split(' ')]
    emb = {el: tmp.index(el) for el in tmp}
    v, w2i, i2w, i2e = u.get_vocabulary(sources, emb)
    self.assertTrue(isinstance(v, list))
    self.assertTrue(isinstance(w2i, dict))
    self.assertTrue(isinstance(i2w, dict))
    self.assertTrue(isinstance(i2e, dict))
    self.assertEqual(len(v), 11)
    self.assertEqual(set(v), set(tmp))
    self.assertEqual(set(w2i.keys()), set(tmp))
    self.assertEqual(set(w2i.values()), set(i2w.keys()))
    self.assertEqual(set(w2i.values()), set(i2e.keys()))
    self.assertEqual(len(set(w2i.values())), len(set(tmp)))
    self.assertEqual(set(i2w.values()), set(tmp))
    self.assertEqual(i2e[w2i['my']], emb['my'])

  def test_get_vocabulary_unicode_lower(self):
    sources = ['je veux écrire', 'ecrire et passion', 'élaguer tout']
    emb = {'je': 1, 'veux': 2, 'ecrire': 3, 'et': 4, 'passion': 5, 'elaguer': 6, 'tout': 7}
    v, w2i, i2w, i2e = u.get_vocabulary(sources, emb, unidecode_lower=True)
    self.assertEqual(len(v), 7)
    self.assertEqual(set(emb.keys()), set(w2i.keys()))


class SplitInto3Test(unittest.TestCase):
  def test_split_one_list(self):
    a = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    splits = u.split_into_3(a, first_split=0.2, second_split=0.6)[0]
    self.assertEqual(splits[0], [0, 1])
    self.assertEqual(splits[1], [2, 3, 4, 5, 6, 7])
    self.assertEqual(splits[2], [8, 9])

  def test_split_multiple_list(self):
    a = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    b = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j']
    (a1, a2, a3), (b1, b2, b3) = u.split_into_3(a, b, first_split=0.2, second_split=0.6)
    self.assertEqual(a1, [0, 1])
    self.assertEqual(a2, [2, 3, 4, 5, 6, 7])
    self.assertEqual(a3, [8, 9])
    self.assertEqual(b1, ['a', 'b'])
    self.assertEqual(b2, ['c', 'd', 'e', 'f', 'g', 'h'])
    self.assertEqual(b3, ['i', 'j'])


if __name__ == '__main__':
  unittest.main()
