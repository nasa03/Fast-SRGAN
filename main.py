from argparse import ArgumentParser
from dataloader import DataLoader
from model import FastSRGAN
import tensorflow as tf
import os

parser = ArgumentParser()
parser.add_argument('--image_dir', default='/Users/hasty/Datasets/div2k', type=str, help='Path to high resolution image directory.')
parser.add_argument('--batch_size', default=16, type=int, help='Batch size for training.')
parser.add_argument('--epochs', default=100000, type=int, help='Number of epochs for training')
parser.add_argument('--hr_size', default=128, type=int, help='Low resolution input size.')
parser.add_argument('--lr', default=1e-4, type=float, help='Learning rate for optimizers.')
parser.add_argument('--save_iter', default=200, type=int,
                    help='The number of iterations to save the tensorboard summaries and models.')


@tf.function
def train_step(model, x, y):
    """Single train step function for the SRGAN.
    Args:
        model: An object that contains a tf keras compiled discriminator model.
        x: The low resolution input image.
        y: The desired high resolution output image.

    Returns:
        d_loss: The mean loss of the discriminator.
    """
    valid = tf.ones((x.shape[0], 1))

    with tf.GradientTape() as disc_tape, tf.GradientTape() as gen_tape:
        # Get fake image:
        fake_hr = model.generator(x)

        # Get discriminator predictions:
        valid_predictions = model.discriminator(y)
        fake_predictions = model.discriminator(fake_hr)

        # calculate loss of the discriminator
        valid_loss = valid_predictions - tf.math.reduce_mean(fake_predictions, axis=0, keepdims=True) - valid
        fake_loss = fake_predictions - tf.math.reduce_mean(valid_predictions, axis=0, keepdims=True) + valid
        valid_loss = tf.math.reduce_mean(tf.square(valid_loss))
        fake_loss = tf.math.reduce_mean(tf.square(fake_loss))
        d_loss = tf.divide(valid_loss + fake_loss, 2.0)

        # Get generator losses:
        valid_loss = valid_predictions - tf.math.reduce_mean(fake_predictions, axis=0, keepdims=True) + valid
        fake_loss = fake_predictions - tf.math.reduce_mean(valid_predictions, axis=0, keepdims=True) - valid
        valid_loss = tf.math.reduce_mean(tf.square(valid_loss))
        fake_loss = tf.math.reduce_mean(tf.square(fake_loss))
        adv_loss = tf.divide(valid_loss + fake_loss, 2.0)
        content_loss = model.content_loss(y, fake_hr)
        mse_loss = tf.reduce_mean(tf.reduce_mean(tf.square(fake_hr - y), axis=[1, 2, 3]))
        perceptual_loss = content_loss + adv_loss + mse_loss

    # Backprop on Discriminator
    disc_grads = disc_tape.gradient(d_loss, model.discriminator.trainable_variables)
    model.disc_optimizer.apply_gradients(zip(disc_grads, model.discriminator.trainable_variables))

    # Backprop on Generator
    gen_grads = gen_tape.gradient(perceptual_loss, model.generator.trainable_variables)
    model.gen_optimizer.apply_gradients(zip(gen_grads, model.generator.trainable_variables))

    return valid_loss, fake_loss, adv_loss, content_loss, mse_loss


def train(model, dataset, log_iter, writer):
    """
    Function that defines a single training step for the SR-GAN.
    Args:
        model: An object that contains tf keras compiled generator and
               discriminator models.
        dataset: A tf data object that contains low and high res images.
        log_iter: Number of iterations after which to add logs in 
                  tensorboard.
        writer: Summary writer
    """
    with writer.as_default():
        # Iterate over dataset
        for x, y in dataset:
            dReal, dFake, adv_loss, content_loss, mse_loss = train_step(model, x, y)
            # Log tensorboard summaries if log iteration is reached.
            if model.iterations % log_iter == 0:
                tf.summary.scalar('Adversarial Loss', adv_loss, step=model.iterations)
                tf.summary.scalar('Content Loss', content_loss, step=model.iterations)
                tf.summary.scalar('MSE Loss', mse_loss, step=model.iterations)
                tf.summary.scalar('Discriminator Loss Real', dReal, step=model.iterations)
                tf.summary.scalar('Discriminator Loss Fake', dFake, step=model.iterations)
                tf.summary.image('Low Res', tf.cast(255 * x, tf.uint8), step=model.iterations)
                tf.summary.image('High Res', tf.cast(255 * (y + 1.0) / 2.0, tf.uint8), step=model.iterations)
                tf.summary.image('Generated', tf.cast(255 * (model.generator.predict(x) + 1.0) / 2.0, tf.uint8),
                                 step=model.iterations)
                model.generator.save('models/generator.h5')
                model.discriminator.save('models/discriminator.h5')
                writer.flush()
            model.iterations += 1


def main():
    # Parse the CLI arguments.
    args = parser.parse_args()

    # create directory for saving trained models.
    if not os.path.exists('models'):
        os.makedirs('models')

    # Create the tensorflow dataset.
    ds = DataLoader(args.image_dir, args.hr_size).dataset(args.batch_size)

    # Define the directory for saving the SRGAN training tensorbaord summary.
    train_summary_writer = tf.summary.create_file_writer('logs/train')

    # Initialize the GAN object.
    gan = FastSRGAN(args)

    # Run training.
    for _ in range(args.epochs):
        train(gan, ds, args.save_iter, train_summary_writer)


if __name__ == '__main__':
    main()
