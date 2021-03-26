import mnist
import numpy as np 

def rescale_image(unscaled_image):
    """Rescale images by dividing each pixel's weight by 255."""
    image = [(px / 255) for px in unscaled_image.flatten().tolist()]
    image = np.array(image).reshape(1, -1) 
    return image


def digit_to_target(digit):
    """Convert a digit to a sequence of 0's and 1's of length 10, where a 1 is in the position
    corresponding to that digit and 0's are everywhere else."""
    target = np.zeros(10)
    target[digit] = 1
    target = target.reshape(1, -1)
    return target
    

# mnist dataset
mnist_training_images = mnist.train_images()
mnist_testing_images = mnist.test_images()
training_labels = mnist.train_labels()
testing_labels = mnist.test_labels()

# training images and testing images
training_images = [rescale_image(image) for image in mnist_training_images]
training_images = np.array(training_images)
testing_images = [rescale_image(image) for image in mnist_testing_images]
testing_images = np.array(testing_images)

# training targets and testing targets
training_targets = [digit_to_target(digit) for digit in training_labels]
training_targets = np.array(training_targets)
testing_targets = [digit_to_target(digit) for digit in testing_labels]
testing_targets = np.array(testing_targets) 

