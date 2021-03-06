#!/usr/bin/env python3.6
"""Main script for training a model for gaze estimation."""
import argparse

import coloredlogs
import tensorflow as tf

if __name__ == '__main__':

    # Set global log level
    parser = argparse.ArgumentParser(description='Train a gaze estimation model.')
    parser.add_argument('-v', type=str, help='logging level', default='info',
                        choices=['debug', 'info', 'warning', 'error', 'critical'])
    args = parser.parse_args()
    coloredlogs.install(
        datefmt='%d/%m %H:%M',
        fmt='%(asctime)s %(levelname)s %(message)s',
        level=args.v.upper(),
    )

    # Initialize Tensorflow session
    tf.logging.set_verbosity(tf.logging.ERROR)
    gpu_options = tf.GPUOptions(allow_growth=True)
    for i in range(0, 15):
        # Specify which people to train on, and which to test on
        person_id = 'p%02d' % i
        other_person_ids = ['p%02d' % j for j in range(15) if i != j]
        
    with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as session:

        # Declare some parameters
        batch_size = 32

        # Define some model-specific parameters
        elg_first_layer_stride = 1
        elg_num_modules = 3
        elg_num_feature_maps = 32

        # Define training data source
        from datasources import UnityEyes
        from datasources import HDF5Source
        from datasources import EYEDIAPSource
        unityeyes = UnityEyes(
            session,
            batch_size=batch_size,
            data_format='NCHW',
            unityeyes_path='/home/zhangbochen/imgs',
            min_after_dequeue=1000,
            generate_heatmaps=True,
            shuffle=True,
            staging=True,
            eye_image_shape=(36, 60),
            heatmaps_scale=1.0 / elg_first_layer_stride,
        )
        eyediap = EYEDIAPSource(
            session,
            batch_size=batch_size,
            data_format='',
            data_file='.../data_FT_S.txt',
            vector_gt_files='.../gtv_cam_FT_S.txt',
            eyediap_path='.../EyeDiap',
            testing=False,
            eye_image_shape=(, ),
        )
        unityeyes.set_augmentation_range('translation', 2.0, 10.0)
        unityeyes.set_augmentation_range('rotation', 1.0, 10.0)
        unityeyes.set_augmentation_range('intensity', 0.5, 20.0)
        unityeyes.set_augmentation_range('blur', 0.1, 1.0)
        unityeyes.set_augmentation_range('scale', 0.01, 0.1)
        unityeyes.set_augmentation_range('rescale', 1.0, 0.5)
        unityeyes.set_augmentation_range('num_line', 0.0, 2.0)
        unityeyes.set_augmentation_range('heatmap_sigma', 7.5, 2.5)

        # Define model
        from models import ELG
        model = ELG(
            # Tensorflow session
            session,

            
            first_layer_stride=elg_first_layer_stride,
            num_feature_maps=elg_num_feature_maps,
            num_modules=elg_num_modules,

            
            learning_schedule=[
                {
                    'loss_terms_to_optimize': {
                        'heatmaps_mse': ['hourglass'],
                        'radius_mse': ['radius'],
                        'combined_loss': ['hourglass', 'densenet'],
                    },
                    'metrics': ['gaze_mse', 'gaze_ang'],
                    'learning_rate': 1e-3,
                },
            ],

            # Data sources for training (and testing).
            train_data={'synthetic': unityeyes},
            
            test_data={
                    'mpi': HDF5Source(
                        session,
                        data_format='NCHW',
                        batch_size=batch_size,
                        keys_to_use=['test/' + person_id],
                        hdf_path='/home/zhangbochen/MPIIH5/MPIIGaze.h5',
                        eye_image_shape=(90, 150),
                        testing=True,
                    ),
                },
        )

        # Train this model for a set number of epochs
        model.train(
            num_epochs=100,
        )
