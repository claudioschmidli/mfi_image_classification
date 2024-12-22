import tensorflow as tf
import numpy as np
import glob
import os

import matplotlib.pyplot as plt
from itertools import islice
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import confusion_matrix
from sklearn.utils import class_weight
from sklearn.metrics import get_scorer
from sklearn.metrics import f1_score


class Data:
    def __init__(
        self,
        data_dir,
        img_height,
        img_width,
        channels,
        resize_option,
        batch_size,
        data_within_0_1=False,
        verbose=2,
    ):
        # Save the passed parameters
        self.img_height = img_height
        self.img_width = img_width
        self.channels = channels
        self.resize_option = resize_option
        self.batch_size = batch_size
        self.data_within_0_1 = data_within_0_1
        self.verbose = verbose
        self.AUTOTUNE = tf.data.AUTOTUNE
        # Get datapaths and shuffle later in a separate step (4th line below)
        self.image_paths = tf.data.Dataset.list_files(
            data_dir + "/*/*.jpg", shuffle=False
        )
        self.image_count = len(self.image_paths)
        self.image_paths = self.image_paths.shuffle(
            self.image_count, reshuffle_each_iteration=False, seed=1
        )
        self.val_preds = None
        self.test_preds = None
        # Print some example images
        if self.verbose > 1:
            print("Examples of found images:")
            for f in self.image_paths.take(4):
                print(f.numpy())
        self.class_names = [
            os.path.basename(path) for path in glob.glob(data_dir + "/*")
        ]
        self.nbr_of_classes = len(self.class_names)

    def test_val_train_split(self, test_size_relative, val_size_relative):
        # First make test split
        train_size_all = int(self.image_count * (1 - test_size_relative))
        train_ds_all = self.image_paths.take(train_size_all)
        self.test_ds = self.image_paths.skip(train_size_all)
        # Then do train val split
        val_size = int(self.image_count * val_size_relative)
        self.train_ds = train_ds_all.skip(val_size)
        self.val_ds = train_ds_all.take(val_size)
        self.train_ds_size = len(self.train_ds)
        self.val_ds_size = len(self.val_ds)
        self.test_ds_size = len(self.test_ds)

        assert self.image_count == len(self.test_ds) + len(self.train_ds) + len(
            self.val_ds
        )

    def get_label(self, file_path):
        # Convert the path to a list of path components
        parts = tf.strings.split(file_path, os.path.sep)
        # The second to last is the class-directory
        one_hot = parts[-2] == self.class_names
        # Integer encode the label
        return tf.argmax(one_hot)

    def decode_img(self, img):
        # Convert the compressed string to a 3D uint8 tensor
        img = tf.io.decode_jpeg(img, channels=3)
        # Resize the image to the desired size
        if self.resize_option == "scale":
            rescaled_img = tf.image.resize(img, [self.img_height, self.img_width])
        elif self.resize_option == "crop_or_pad":
            rescaled_img = tf.image.resize_with_crop_or_pad(
                img, self.img_height, self.img_width
            )
        elif self.resize_option == "pad":
            rescaled_img = tf.image.resize_with_pad(
                img, self.img_height, self.img_width
            )
        else:
            raise ValueError(
                'Enter a valid string for parameter "resize_option"! (e.g. scale, crop_or_pad or pad)'
            )
        return rescaled_img

    def get_dataset(self, dataset="train"):
        if dataset == "train":
            data = self.train_ds
        elif dataset == "val":
            data = self.val_ds
        elif dataset == "test":
            data = self.test_ds
        else:
            raise ValueError(
                'Enter a valid string for parameter "dataset"! (e.g. train, val or test)'
            )
        return data

    def check_if_data_batched(self, dataset="train"):
        data = self.get_dataset(dataset)
        return_value = False
        item = next(iter(data))
        if type(item) is tuple:
            shape = item[0].shape
            if len(shape) == 4 and shape[0] > 1:
                return_value = True
        return return_value

    def get_unbatched_dataset(self, dataset="train"):
        data = self.get_dataset(dataset)
        if self.check_if_data_batched(dataset):
            data = data.unbatch()
        return data

    def get_class_occurences(self, dataset="train", output_as_arrays=False):
        data = self.get_unbatched_dataset(dataset)
        images, labels = tuple(zip(*data))
        labels = np.array(labels)
        labels, occurences = np.unique(labels, return_counts=True)
        output = labels, occurences
        if not output_as_arrays:
            output = dict(zip(self.class_names, output[1]))
        return output

    def calculate_class_weights(self):
        # Calculates class weights that could but must not be used later.
        labels, occurences = self.get_class_occurences(
            dataset="train", output_as_arrays="True"
        )
        classes = [i for i in range(len(self.class_names))]
        class_weights = class_weight.compute_class_weight(
            class_weight="balanced", classes=classes, y=self.get_labels("train")
        )
        self.class_weights = dict(zip(classes, class_weights))

    def get_labels(self, dataset):
        data = self.get_unbatched_dataset(dataset)
        images, labels = tuple(zip(*data))
        return np.array(labels)

    def process_path(self, file_path):
        label = self.get_label(file_path)
        # Load the raw data from the file as a string
        img = tf.io.read_file(file_path)
        img = self.decode_img(img)
        return img, label

    def parallel_process_path(self, train=True, val=True, test=True):
        # Set `num_parallel_calls` so multiple images are loaded/processed in parallel.
        if train:
            self.train_ds = self.train_ds.map(
                self.process_path, num_parallel_calls=self.AUTOTUNE
            )
        if val:
            self.val_ds = self.val_ds.map(
                self.process_path, num_parallel_calls=self.AUTOTUNE
            )
        if test:
            self.test_ds = self.test_ds.map(
                self.process_path, num_parallel_calls=self.AUTOTUNE
            )

    def apply_caching(self, train=True, val=True, test=True):
        if train:
            self.train_ds = self.train_ds.cache()
        if val:
            self.val_ds = self.val_ds.cache()
        if test:
            self.test_ds = self.test_ds.cache()

    def apply_batching(self, train=True, val=True, test=True):
        if train:
            self.train_ds = self.train_ds.batch(self.batch_size)
        if val:
            self.val_ds = self.val_ds.batch(self.batch_size)
        if test:
            self.test_ds = self.test_ds.batch(self.batch_size)

    @tf.function
    def augment_image_batch(self, data, seed):
        # https://towardsdatascience.com/image-augmentations-in-tensorflow-62967f59239d
        # Note: First, we batch the datasets, making any supported operation run
        # faster. This concept is called vectorizing and helps us apply the
        # same transformation to a batch of samples rather than
        # one-by-one augmenting the data
        image, label = data
        # Make a new seed.
        new_seed = tf.random.experimental.stateless_split(seed, num=1)[0, :]
        image = tf.image.stateless_random_flip_left_right(image, seed=new_seed)
        image = tf.image.stateless_random_flip_up_down(image, seed=new_seed)
        return image, label, seed

    """
    @tf.function
    def rotate_image(feature, label, seed):
        num_samples = int(tf.shape(feature)[0])
        degrees = tf.random.stateless_uniform(
            shape=(num_samples,), seed=seed, minval=-45, maxval=45
        )
        degrees = degrees * 0.017453292519943295  # convert the angle in degree to radians

        rotated_images = tfa.image.rotate(feature, degrees)

        return rotated_images, label
    """

    def apply_data_augmentation(self, train=True, val=False, test=False):
        # Create a `Counter` object and `Dataset.zip` it together with the training set.
        counter = tf.data.experimental.Counter()

        if train:
            train_ds = tf.data.Dataset.zip((self.train_ds, (counter, counter)))
            train_ds = train_ds.map(
                self.augment_image_batch, num_parallel_calls=self.AUTOTUNE
            )

            # we have to add the rotation operation manually; it's not yet part of native TensorFlow
            # train_ds = self.train_ds.map(
            #                                self.rotate_image
            #                                 )  # we have to add the rotation operation manually; it's not yet part of native TensorFlow
            # remove seed
            self.train_ds = train_ds.map(lambda image, label, seed: (image, label))
        if val:
            pass
        if test:
            pass

    def apply_prefetching(self, train=True, val=True, test=True):
        if train:
            self.train_ds = self.train_ds.prefetch(buffer_size=self.AUTOTUNE)
        if val:
            self.val_ds = self.val_ds.prefetch(buffer_size=self.AUTOTUNE)
        if test:
            self.test_ds = self.test_ds.prefetch(buffer_size=self.AUTOTUNE)

    def config_performance(
        self,
        caching=True,
        batching=True,
        prefetching=True,
        train=True,
        val=True,
        test=False,
        augment_train=True,
    ):
        # Let's make sure to use buffered prefetching so you can yield data from disk without having I/O become blocking.
        # see: https://www.tensorflow.org/tutorials/load_data/images#configure_the_dataset_for_performance
        # These are two important methods you should use when loading data:
        self.parallel_process_path(train, val, test)
        self.calculate_class_weights()
        if caching:
            self.apply_caching(train, val, test)

        if batching:
            self.apply_batching(train, val, test)
        if augment_train:
            self.apply_data_augmentation(True, False, False)
        if prefetching:
            self.apply_prefetching(train, val, test)

    def scale_0_to_1(self, train=True, val=True, test=True):
        normalization_layer = tf.keras.layers.Rescaling(scale=1.0 / 255, offset=0)
        if train:
            self.train_ds = self.train_ds.map(lambda x, y: (normalization_layer(x), y))
        if val:
            self.val_ds = self.val_ds.map(lambda x, y: (normalization_layer(x), y))
        if test:
            self.test_ds = self.test_ds.map(lambda x, y: (normalization_layer(x), y))
        self.data_within_0_1 = True

    def plot_example_images(self, dataset="train", seed=0, examples_per_class=4):
        nbr_of_classes = len(self.class_names)
        data = self.get_dataset(dataset)
        if self.check_if_data_batched(dataset):
            data = data.unbatch()
        labels_as_int = self.get_labels(dataset)
        plt.figure(figsize=(10, 10))
        for class_index in range(0, nbr_of_classes):
            example_indices = np.where((labels_as_int == class_index))[0]
            plt.suptitle(
                f"Example images from {dataset}-set",
                size="xx-large",
                va="bottom",
                y=0.92,
            )
            for i in range(examples_per_class):
                if (i + seed * nbr_of_classes) < len(example_indices):
                    example_index = example_indices[i + seed * nbr_of_classes]
                    gen = iter(data)
                    gen = islice(gen, example_index, example_index + 1, 1)
                    image, label_as_int = next(gen)
                    label = self.class_names[label_as_int]
                    if self.data_within_0_1:
                        image = image * 255
                    plot_index = class_index + i * (nbr_of_classes) + 1
                    ax = plt.subplot(examples_per_class, nbr_of_classes, plot_index)
                    plt.imshow(image.numpy().astype("uint8"))
                    plt.title(label)
                    plt.axis("off")

    def predict(self, model, dataset="val"):
        data = self.get_dataset(dataset)
        if not self.check_if_data_batched(dataset):
            raise Exception(
                f"Error: {dataset}-dataset is not batched! The model needs a batched dataset for predictions!"
            )
        if dataset == "val":
            self.val_preds = model.predict(data, verbose=0)
            self.val_preds_as_int = np.apply_along_axis(np.argmax, 1, self.val_preds)
            self.val_labels_as_int = self.get_labels(dataset)

        if dataset == "test":
            self.test_preds = model.predict(data, verbose=0)
            self.test_preds_as_int = np.apply_along_axis(np.argmax, 1, self.test_preds)
            self.test_labels_as_int = self.get_labels(dataset)

    def check_if_already_predicted(self, dataset):
        already_predicted = False
        if dataset == "val":
            if self.val_preds is not None:
                already_predicted = True
        elif dataset == "test":
            if self.test_preds is not None:
                already_predicted = True
        else:
            raise ValueError(
                'Enter a valid string for parameter "dataset"! (e.g. val or test)'
            )
        return already_predicted

    def plot_prediction(
        self, model, dataset="val", pred_correct=True, seed=0, examples_per_class=4
    ):
        nbr_of_classes = len(self.class_names)
        data = self.get_dataset(dataset)
        if not self.check_if_data_batched(dataset):
            raise Exception(
                f"Error: {dataset}-dataset is not batched! The model needs a batched dataset for predictions!"
            )
        if not self.check_if_already_predicted(dataset):
            self.predict(model, dataset)

        if dataset == "val":
            preds = self.val_preds
            preds_as_int = self.val_preds_as_int
            labels_as_int = self.val_labels_as_int

        elif dataset == "test":
            preds = self.test_preds
            preds_as_int = self.test_preds_as_int
            labels_as_int = self.test_labels_as_int

        else:
            raise ValueError(
                'Enter a valid string for parameter "dataset"! (e.g. val or test)'
            )

        data = data.unbatch()

        plt.figure(figsize=(10, 10))
        for class_index in range(0, nbr_of_classes):
            comparison = preds_as_int == labels_as_int
            if pred_correct:
                plt.suptitle(
                    "Correct predicted examples", size="xx-large", va="bottom", y=0.92
                )
                example_indices = np.where(comparison & (labels_as_int == class_index))[
                    0
                ]
            else:
                plt.suptitle(
                    "Incorrect predicted examples", size="xx-large", va="bottom", y=0.92
                )
                example_indices = np.where(
                    (comparison != True) & (labels_as_int == class_index)
                )[0]
            for i in range(examples_per_class):
                if (i + seed * nbr_of_classes) < len(example_indices):
                    example_index = example_indices[i + seed * nbr_of_classes]
                    gen = iter(data)
                    gen = islice(gen, example_index, example_index + 1, 1)
                    image, label_as_int = next(gen)
                    label = self.class_names[label_as_int]
                    pred = preds[example_index]
                    pred_label_as_int = tf.math.argmax(pred).numpy()
                    pred_label = self.class_names[pred_label_as_int]
                    pred_prob = pred[pred_label_as_int]
                    pred_prob = str(round(pred_prob, 2))
                    title = f"True: {label}\n{pred_label} ({pred_prob})"
                    plot_index = class_index + i * (nbr_of_classes) + 1
                    ax = plt.subplot(examples_per_class, nbr_of_classes, plot_index)
                    if self.data_within_0_1:
                        image = image * 255
                    plt.imshow(image.numpy().astype("uint8"))
                    plt.title(title)
                    plt.axis("off")

    def plot_confusion_matrix(self, dataset):
        if not self.check_if_already_predicted(dataset):
            raise Exception(
                f"Error: No predictions made for {dataset}-dataset. Use instance.predict(model, dataset."
            )

        if dataset == "val":
            preds_as_int = self.val_preds_as_int
            labels_as_int = self.val_labels_as_int

        elif dataset == "test":
            preds_as_int = self.test_preds_as_int
            labels_as_int = self.test_labels_as_int

        cm = confusion_matrix(labels_as_int, preds_as_int)
        disp = ConfusionMatrixDisplay(
            confusion_matrix=cm, display_labels=self.class_names
        )
        disp.plot(cmap=plt.cm.Blues)
        fig = disp.ax_.get_figure()
        fig.set_figwidth(9.5)
        fig.set_figheight(9.5)
        fig.suptitle(
            f"Confusion Matrix {dataset}-dataset", size="xx-large", va="bottom", y=0.88
        )
        plt.show()

    def show_img_counts(self):
        print("Number of images in train, val and test set:")
        print(f"Total nbr of images:\t {self.image_count}")
        print(f"Train set nbr of images: {self.train_ds_size}")
        print(f"Val set nbr of images: \t {self.val_ds_size}")
        print(f"Test set nbr of images:\t {self.test_ds_size}")
        print("\n")

    def show_class_weights(self):
        print(f"Class weights for training:")
        for key, value in self.class_weights.items():
            key = self.class_names[int(key)]
            if len(key) < 10:
                space = (10 - len(key)) * " "
            value = str(round(value, 4))
            print(f"{key}:{space}{value}")
        print("\n")

    def show_nbr_img_per_class(self, dataset):
        print(f"Number of images per class in {dataset}-set")
        dict = self.get_class_occurences(dataset)

        for key, value in dict.items():
            if len(key) < 10:
                space = (10 - len(key)) * " "
            print(f"{key}:{space}{value}")
        print("\n")

    def show_predictions(self, model, dataset):
        data = self.get_dataset(dataset)
        if not self.check_if_data_batched(dataset):
            raise Exception(
                f"Error: {dataset}-dataset is not batched! The model needs a batched dataset for predictions!"
            )
        if not self.check_if_already_predicted(dataset):
            self.predict(model, dataset)

        if dataset == "val":
            print(
                "Accuracy socre on validation set:\t",
                get_scorer("accuracy")._score_func(
                    self.val_labels_as_int, self.val_preds_as_int
                ),
            )
            print(
                "F1 macro socre on validation set:\t",
                f1_score(
                    self.val_labels_as_int, self.val_preds_as_int, average="macro"
                ),
            )

        elif dataset == "test":
            print(
                "Accuracy socre on test set:\t\t",
                get_scorer("accuracy")._score_func(
                    self.test_labels_as_int, self.test_preds_as_int
                ),
            )
            print(
                "F1 macro socre on test set:\t\t",
                f1_score(
                    self.test_labels_as_int, self.test_preds_as_int, average="macro"
                ),
            )
        else:
            raise ValueError(
                'Enter a valid string for parameter "dataset"! (e.g. val or test)'
            )


def get_preprocessed_data(
    data_dir,
    img_height=28,
    img_width=28,
    channels=3,
    resize_option="scale",
    batch_size=32,
    data_within_0_1=False,
    test_size_relative=0.10,
    val_size_relative=0.20,
    scale_to_0_1=True,
    verbose=2,
    augment_train=True,
):

    data = Data(
        data_dir,
        img_height,
        img_width,
        channels,
        resize_option,
        batch_size,
        data_within_0_1,
        verbose,
    )
    data.test_val_train_split(test_size_relative, val_size_relative)
    data.config_performance(
        caching=True,
        batching=True,
        prefetching=True,
        train=True,
        val=True,
        test=True,
        augment_train=augment_train,
    )
    if scale_to_0_1:
        data.scale_0_to_1()
    return data


def save_model(model, location):
    print("Saving model", location)
    save_locally = tf.saved_model.SaveOptions(experimental_io_device="/job:localhost")
    model.model.save(location, options=save_locally)


def load_models(paths, strategy):
    models = []
    for path in paths:
        with strategy.scope():
            load_locally = tf.saved_model.LoadOptions(
                experimental_io_device="/job:localhost"
            )
            model = tf.keras.models.load_model(path, options=load_locally)
            models.append(model)
    return models


class WeightedAverageLayer(tf.keras.layers.Layer):
    def __init__(self, w1, w2, **kwargs):
        super(WeightedAverageLayer, self).__init__(**kwargs)
        self.w1 = w1
        self.w2 = w2

    def call(self, inputs):
        return self.w1 * inputs[0] + self.w2 * inputs[1]


def predict_ensemble(data: Data, models, weights=(0.6, 0.4)):
    probabilities = []
    for model in models:
        probability = model.predict(data.test_ds, verbose=0)
        probabilities.append(probability)
    ensemble_output = WeightedAverageLayer(*weights)(probabilities)
    ensemble_output = np.apply_along_axis(np.argmax, 1, ensemble_output)
    return ensemble_output
