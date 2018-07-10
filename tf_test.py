import numpy as np
import tensorflow as tf
import tensorflow.contrib.eager as tfe


tfe.enable_eager_execution()


# [3, 2, 4]
a = np.asarray([[[1,1,1,1], [2,2,2,2]], [[3,3,3,3], [4,4,4,4]], [[5,5,5,5], [6,6,6,6]]])
atf = tf.constant([[[1,1,1,1], [2,2,2,2]], [[3,3,3,3], [4,4,4,4]], [[5,5,5,5], [6,6,6,6]]])
print('a = {}\nshape = {}\n\n'.format(a, a.shape))

# [3, 4]
b = np.asarray([[1,1,1,1], [2,2,2,2], [3,3,3,3]])
btf = tf.constant([[1,1,1,1], [2,2,2,2], [3,3,3,3]])
print('b = {}\nshape = {}\n\n'.format(b, b.shape))

# [3, 3, 2]
c =  np.einsum('ij,klj', b, a)
ctf = tf.einsum('ij,klj', btf, atf)
print('c = {}\nshape = {}\n'.format(c, c.shape))
print('ctf = {}\nshape = {}\n'.format(ctf, ctf.shape))

# [3, 2]
d = np.einsum('iij->ij', c)
idx_to_keep = [[i, i] for i in range(ctf.shape[0])]
print(idx_to_keep)
dtf = tf.gather_nd(c, idx_to_keep)
# dtf = tf.einsum('iij->ij', ctf)  # it doesn't work because tf.einsum != np.einsum
print('d = {}\nshape = {}\n'.format(d, d.shape))
print('dtf = {}\nshape = {}'.format(dtf, dtf.shape))
