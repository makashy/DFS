"""Contains dataset importer for NYU Depth Dataset V2"""

from __future__ import absolute_import, division, print_function

import numpy as np
import tables

class NYU:
    """Iterator for looping over elements of NYU Depth Dataset V2 backwards."""

    def __init__(self,
                 NYU_Depth_Dataset_V2_address,
                 batch_size=1,
                 repeater=False,
                 label_type='labels'):
        self.file = tables.open_file(NYU_Depth_Dataset_V2_address)
        self.index = len(self.file.root.images) -1
        self.batch_size = batch_size
        self.repeater = repeater
        self.label_type = label_type.decode()
        self.features = None
        self.labels = None

    def __iter__(self):
        return self

    # Python 3 compatibility
    def __next__(self):
        return self.next()

    def next(self):
        """Retrieve the next pairs from the dataset
        """
        if self.index - self.batch_size <= 0:
            if not self.repeater:
                raise StopIteration
            else:
                self.index = len(self.file.root.images) -1
        self.index = self.index - self.batch_size

        self.features = np.array(self.file.root.images[self.index:self.index+self.batch_size])
        self.features = np.transpose(self.features, [0, 3, 2, 1])

        if self.label_type == 'depths':
            label_set = self.file.root.depths
        elif self.label_type == 'labels':
            label_set = self.file.root.labels
        else:
            raise ValueError('invalid label type')

        self.labels = np.array(label_set[self.index:self.index+self.batch_size])
        self.labels = np.transpose(self.labels, [0, 2, 1])

        return self.features, self.labels
