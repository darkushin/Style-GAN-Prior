import numpy as np
import imageio
import os
import argparse
import pickle
from tqdm import tqdm
import tensorflow as tf
import cv2
import dnnlib
import dnnlib.tflib as tflib
import config
from perceptual_model import PerceptualModel

STYLEGAN_MODEL_URL = 'https://drive.google.com/uc?id=1MEGjdvVpUsu1jB4zrXZN7Y4kBBOzizDQ'

GAUSSIAN_KERNEL_SIZE = 5
MOTION_KERNEL_SIZE = 7
UNIFORM_KERNEL_SIZE = 7


def blur_image(im, blur_kernel):
    """
    A function that blurs an image in the spatial domain using convolution with a gaussian kernel
    :param im: The image that should be blurred
    :param blur_kernel: The blur kernel that should be used to blur the input image
    :return: The blurred image
    """
    return tf.nn.depthwise_conv2d(im, blur_kernel, [1, 1, 1, 1],
                                  padding='SAME', data_format='NHWC')


def gauss_kernel(kernel_size, sigma=5):
    """
    Creates a gaussian blur kernel.
    :param kernel_size: the size of the kernel - larger kernel size implies a blurrier image
    :param sigma: the std that should be used for the gaussian
    :return: the kernel that should be used for gaussian blurring
    """
    ax = tf.range(-kernel_size // 2 + 1.0, kernel_size // 2 + 1.0)
    xx, yy = tf.meshgrid(ax, ax)
    kernel = tf.exp(-(xx ** 2 + yy ** 2) / (2.0 * sigma ** 2))
    kernel = kernel / tf.reduce_sum(kernel)
    kernel = tf.tile(kernel[..., tf.newaxis], [1, 1, 3])
    return kernel[..., tf.newaxis]


def motion_kernel(direction, kernel_size):
    """
    Creates a motion blur kernel, i.e a kernel of size (kernel_size, kernel_size, 3) with values 1/kernel_size on the
    diagonal and 0 elsewhere
    :param direction: the direction of the motion blur. Can be either horizontal blur or vertical blur
    :param kernel_size: the size of the kernel - larger kernel size implies a blurrier image
    :return: the kernel that should be used for motion blurring
    """
    kernel = np.zeros((kernel_size, kernel_size))
    if direction == 'motion-horizontal':
        kernel[(kernel_size // 2)-1, :] = np.ones(kernel_size) / kernel_size
    elif direction == 'motion-vertical':
        kernel[:, (kernel_size // 2)-1] = np.ones(kernel_size) / kernel_size
    else:
        raise Exception('ERROR: Invalid motion-direction blur. Optional values are motion-horizontal/motion-vertical')
    kernel = tf.convert_to_tensor(kernel, dtype=tf.float32)
    kernel = tf.tile(kernel[..., tf.newaxis], [1, 1, 3])
    return kernel[..., tf.newaxis]


def uniform_kernel(kernel_size):
    """
    Creates a uniform blur kernel of size kernel_size X kernel_size X 3
    :param kernel_size: the size of the kernel - larger kernel size implies a blurrier image
    :return: the kernel that should be used for uniform blurring
    """
    kernel = tf.ones((kernel_size, kernel_size)) / (kernel_size ** 2)
    kernel = tf.tile(kernel[..., tf.newaxis], [1, 1, 3])
    return kernel[..., tf.newaxis]


def create_blur_kernel(blur_method):
    """
    Creates a blur kernel according to the blur method that was passed as an argument
    :param blur_method: the method that should be used to create the blur kernel
    :return: the relevant blur kernel
    """
    if blur_method == 'gaussian':
        return gauss_kernel(GAUSSIAN_KERNEL_SIZE)
    elif 'motion' in blur_method:
        return motion_kernel(blur_method, MOTION_KERNEL_SIZE)
    elif blur_method == 'uniform':
        return uniform_kernel(UNIFORM_KERNEL_SIZE)
    else:
        raise Exception('ERROR: Incorrect kernel - optional blur kernels are '
                        'gaussian/motion-horizontal/motion-vertical/uniform')


def optimize_latent_codes(args):
    tflib.init_tf()

    with dnnlib.util.open_url(STYLEGAN_MODEL_URL, cache_dir=config.cache_dir) as f:
        _G, _D, Gs = pickle.load(f)

    latent_code = tf.get_variable(
        name='latent_code', shape=(1, 18, 512), dtype='float32', initializer=tf.initializers.zeros()
    )
    generated_img = Gs.components.synthesis.get_output_for(latent_code, randomize_noise=False)
    generated_img = tf.transpose(generated_img, [0, 2, 3, 1])
    generated_img = ((generated_img + 1) / 2) * 255

    generated_img = tf.image.resize_images(generated_img, tuple(args.blurred_img_size),
                                           method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)

    blur_kernel = create_blur_kernel(args.blur_kernel)

    generated_blurred_img = tf.image.resize_images(blur_image(generated_img, blur_kernel),
                                                   tuple(args.blurred_img_size),
                                                   method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)

    generated_img_for_display = tf.saturate_cast(generated_img, tf.uint8)

    orig_blurred_img = tf.placeholder(tf.float32, [None, args.blurred_img_size[0], args.blurred_img_size[1], 3])

    perceptual_model = PerceptualModel(img_size=args.blurred_img_size)
    generated_img_features = perceptual_model(generated_blurred_img)
    target_img_features = perceptual_model(orig_blurred_img)

    loss_op = tf.reduce_mean(tf.abs(generated_img_features - target_img_features))

    optimizer = tf.train.AdamOptimizer(learning_rate=args.learning_rate)
    train_op = optimizer.minimize(loss_op, var_list=[latent_code])

    sess = tf.get_default_session()

    img_names = sorted(os.listdir(args.blurred_imgs_dir))
    for img_name in img_names:
        img = imageio.imread(os.path.join(args.blurred_imgs_dir, img_name))

        sess.run(tf.variables_initializer([latent_code] + optimizer.variables()))

        progress_bar_iterator = tqdm(
            iterable=range(args.total_iterations),
            bar_format='{desc}: {percentage:3.0f}% |{bar}| {n_fmt}/{total_fmt}{postfix}',
            desc=img_name
        )

        for i in progress_bar_iterator:
            loss, _ = sess.run(
                fetches=[loss_op, train_op],
                feed_dict={
                    orig_blurred_img: img[np.newaxis, ...]
                }
            )

            progress_bar_iterator.set_postfix_str('loss=%.2f' % loss)

        deblurred_imgs, latent_codes = sess.run(
            fetches=[generated_img_for_display, latent_code],
            feed_dict={
                orig_blurred_img: img[np.newaxis, ...]
            }
        )

        imageio.imwrite(os.path.join(args.deblurred_imgs_dir, img_name), deblurred_imgs[0])
        np.savez(file=os.path.join(args.latents_dir, img_name + '.npz'), latent_code=latent_codes[0])


def create_blurry_imgs(args):
    """
    Creates the blurry images from the original images and saves them to the blurred-imgs-dir. The original images dir
    and the blur kernels that should be used are passed by the user, using the input arguments of the program.
    :param args: the arguments the user entered when calling the program
    """
    blur_method = args.blur_kernel
    img_names = sorted(os.listdir(args.original_imgs_dir))

    if blur_method == 'gaussian':
        kernel = gauss_kernel(GAUSSIAN_KERNEL_SIZE)
    elif 'motion' in blur_method:
        kernel = motion_kernel(blur_method, MOTION_KERNEL_SIZE)
    elif blur_method == 'uniform':
        kernel = uniform_kernel(UNIFORM_KERNEL_SIZE)
    else:
        raise Exception('ERROR: Invalid kernel input - optional kernel names are '
                        'gaussian/motion-horizontal/motion-vertical/uniform.')

    for img_name in img_names:
        im = imageio.imread(os.path.join(args.original_imgs_dir, img_name))
        orig_img = tf.constant(im[np.newaxis, ...], dtype=tf.float32)
        blurred_im = blur_image(orig_img, kernel)
        imageio.imwrite(os.path.join(args.blurred_imgs_dir, f'{blur_method}-{img_name}'), tf.Session().run(blurred_im[0]))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--blurred-imgs-dir', type=str, required=True)
    parser.add_argument('--deblurred-imgs-dir', type=str, required=True)
    parser.add_argument('--latents-dir', type=str, required=True)
    parser.add_argument('--blur-kernel', type=str, required=True)

    parser.add_argument('--original-imgs-dir', type=str, default='')

    parser.add_argument('--blurred-img-size', type=int, nargs=2, default=(256, 256))
    parser.add_argument('--learning-rate', type=float, default=1e-3)
    parser.add_argument('--total-iterations', type=int, default=1000)

    args = parser.parse_args()

    os.makedirs(args.deblurred_imgs_dir, exist_ok=True)
    os.makedirs(args.latents_dir, exist_ok=True)

    if args.original_imgs_dir:
        create_blurry_imgs(args)

    optimize_latent_codes(args)

