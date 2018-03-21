import numpy as np


def save_array(array, name):
    np.savez('{}.npz'.format(name), array)


# arr1 = np.array([])
# if len(arr1):
#     arr1 = np.array([1, 3])
# else:
#     arr1 = np.array([1, 3])
# np.savez("test.npz", arr1)

# c = np.load('test.npz')
# print(c['arr_0'])
arr1 = np.array([1, 2, 3])

save_array(arr1, 'test')