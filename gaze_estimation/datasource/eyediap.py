"""EYEDIAP data source for gaze estimation."""
from threading import Lock
from typing import List

import cv2 as cv
import h5py
import numpy as np
import tensorflow as tf

from core import BaseDataSource
import util.gazemap


class EYEDIAPSource(BaseDataSource):
    """EYEDIAP data loading class (using h5py)."""

    def __init__(self,
                 tensorflow_session: tf.Session,
                 batch_size: int,
                 data_files: str,
                 vector_gt_files: str,
                 eyediap_path: str,
                 testing=False,
                 eye_image_shape=(36, 60),
                 **kwargs):
        """Create queues and threads to read and preprocess data from specified keys."""
        eyediap = eyediap.File(eyediap_path, 'r')
        self._short_name = 'EYEDIAP'
        if testing:
            self._short_name += ':test'

        # Cache some parameters
        self._eye_image_shape = eye_image_shape
        
        self._num_entries = 0
        self._mutex = Lock()
        self._current_index = 0
        super().__init__(tensorflow_session, batch_size=batch_size, testing=testing, **kwargs)

        # Set index to 0 again as base class constructor called HDF5Source::entry_generator once to
        # get preprocessed sample.
        self._current_index = 0

    @property
    def num_entries(self):
        """Number of entries in this data source."""
        return self._num_entries

    @property
    def short_name(self):
        """Short name specifying source HDF5."""
        return self._short_name

    def cleanup(self):
        """Close HDF5 file before running base class cleanup routine."""
        super().cleanup()

    def reset(self):
        """Reset index."""
        with self._mutex:
            super().reset()
            self._current_index = 0

    def entry_generator(self, yield_just_one=False):
        """Read entry from EyeDiap."""
        try:
            while range(1) if yield_just_one else True:
                with self._mutex:
                    if self._current_index >= self.num_entries:
                        if self.testing:
                            break
                        else:
                            self._current_index = 0
                    current_index = self._current_index
                    self._current_index += 1

                assert (len(data_files) > 0)
                data = []
                for n in data_files[0]:
                    if n != "":
                        data_tmp = read_data_file(n)
                        data = np.concatenate((data, data_tmp), axis=0)
                data = np.array([os.path.join(path, filepath) for filepath in data])
                if os.name == "posix":
                    data = np.array([str.replace(filepath, "\\", "/") for filepath in data])

                # Gaze GT (3D vector)
                vgt = None
                if vector_gt_files is not None:
                    assert (len(vector_gt_files) > 0)
                    vgt = np.empty((0, 3))
                    for n in vector_gt_files[0]:
                        if n != "":
                            vgt_tmp = list(np.loadtxt(n))
                            vgt = np.concatenate((vgt, vgt_tmp), axis=0)
                    assert (len(data) == len(vgt))

                
                entry = {}
                entry['gaze'] = vgt
                yield entry
        finally:
            # Execute any cleanup operations as necessary
            pass

    def preprocess_entry(self, entry):
        """Resize eye image and normalize intensities."""
        oh, ow = self._eye_image_shape

        entry['gazemaps'] = util.gazemap.from_gaze2d(
            entry['gaze'], output_size=(oh, ow), scale=0.5,
        ).astype(np.float32)
        if self.data_format == 'NHWC':
            np.transpose(entry['gazemaps'], (1, 2, 0))


        return entry
