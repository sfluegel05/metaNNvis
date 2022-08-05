import torch
import glob

from matplotlib import pyplot as plt
from tf_keras_vis.utils.model_modifiers import ReplaceToLinear
from tf_keras_vis.utils.scores import CategoricalScore

from Main import perform_feature_visualization
from methods import method_keys
from toolsets import toolset_keys
import numpy as np
import tensorflow as tf


def activation_max():
    rng = np.random.default_rng()
    input_size = (64, 224, 224, 3)
    output_size = (64, 1000)
    np_seed = rng.random(input_size, float)
    np_x = rng.random(input_size, float)
    np_x = tf.keras.applications.imagenet_utils.preprocess_input(np_x)
    np_y = rng.integers(0, 1000, output_size)

    tf_net = tf.keras.applications.vgg16.VGG16()
    out = tf_net.predict(np_x)
    out = tf.keras.applications.imagenet_utils.decode_predictions(preds=out)
    print(out)

    attr = perform_feature_visualization(tf_net, method_keys.ACTIVATION_MAXIMIZATION, plot=True,
                                         toolset=toolset_keys.TF_KERAS_VIS,
                                         dummy_input=np_x,
                                         init_args={'model_modifier': ReplaceToLinear()},
                                         exec_args={'score': CategoricalScore(np_y.tolist()),
                                                    'seed_input': np_seed})
    print(attr.shape)

    img_paths = glob.glob('test_imgs/*')
    print(f'found {len(img_paths)} imgs')

    for i, img_path in enumerate(img_paths):
        print(f'img {img_path}')
        orig_image = plt.imread(img_path)
        image = tf.keras.preprocessing.image.load_img(img_path, target_size=(224, 224))
        image = np.expand_dims(image, axis=0)
        image = tf.keras.applications.imagenet_utils.preprocess_input(image)
        predictions = tf_net.predict(image)
        output = tf.keras.applications.imagenet_utils.decode_predictions(preds=predictions)
        print(f'output: {output}')
        plt.subplot(len(img_paths), 2, 2 * i + 1)
        plt.imshow(orig_image)
        plt.title(f"{output[0][0][1]}, {output[0][0][2] * 100:.3f}")
        plt.axis('off')

        attr = perform_feature_visualization(tf_net, method_keys.ACTIVATION_MAXIMIZATION, plot=True,
                                             toolset=toolset_keys.TF_KERAS_VIS,
                                             dummy_input=np_x,
                                             init_args={'model_modifier': ReplaceToLinear()},
                                             exec_args={'score': CategoricalScore(np_y.tolist()),
                                                        'seed_input': np_seed})
        print(attr.shape)
        plt.subplot(len(img_paths), 2, 2 * i + 2)
        plt.imshow(attr[0][0])
        plt.title('tf-keras-vis activation maximization')

    plt.savefig('imagenet_output.png')
    plt.show()


if __name__ == '__main__':
    activation_max()
