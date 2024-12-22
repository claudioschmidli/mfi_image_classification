import tensorflow as tf
import os

import matplotlib.pyplot as plt
import tensorflow_hub as hub
import os
from modules.dataprocessing import Data
from modules.plotmodels import PlotLearning


# Load compressed models from tensorflow_hub
os.environ["TFHUB_MODEL_LOAD_FORMAT"] = "COMPRESSED"
# parameterize to the values in the previous cell


class Transfer_Learning_MobileNet:
    # Pretrained MobileNET that is available on TensorFlowHub
    def __init__(
        self,
        data: Data,
        lrate=0.001,
        l1=0.0,
        l2=0.0,
        num_hidden=16,
        epochs=5,
        patience=1,
    ):
        self.data = data
        self.lrate = lrate
        self.l1 = l1
        self.l2 = l2
        self.num_hidden = num_hidden
        self.epochs = epochs
        self.patience = patience

    def train_and_evaluate(self):
        regularizer = tf.keras.regularizers.l1_l2(self.l1, self.l2)

        layers = [
            tf.keras.layers.Rescaling(
                1.0 / 255,
                input_shape=(
                    self.data.img_height,
                    self.data.img_width,
                    self.data.channels,
                ),
                name="rescaling",
            ),
            hub.KerasLayer(
                "https://tfhub.dev/google/tf2-preview/mobilenet_v2/feature_vector/4",
                trainable=False,
                name="mobilenet_embedding",
            ),
            tf.keras.layers.Dense(
                self.num_hidden,
                kernel_regularizer=regularizer,
                activation="relu",
                name="dense_hidden",
            ),
            tf.keras.layers.Dense(
                self.data.nbr_of_classes,
                kernel_regularizer=regularizer,
                activation="softmax",
                name="flower_prob",
            ),
        ]

        model = tf.keras.Sequential(layers, name="particle_classification")

        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=self.lrate),
            loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
            metrics=["accuracy"],
        )

        print(model.summary())

        self.history = model.fit(
            self.data.train_ds,
            validation_data=self.data.val_ds,
            epochs=self.epochs,
            callbacks=[
                tf.keras.callbacks.EarlyStopping(patience=self.patience),
                PlotLearning(),
            ],
            class_weight=self.data.class_weights,
            verbose=1,
        )
        self.model = model


class RESNET50_FINE_TUNE:
    def __init__(
        self,
        data: Data,
        strategy,
        LR_START,
        LR_MAX,
        LR_MIN,
        LR_RAMPUP_EPOCHS=0,
        LR_SUSTAIN_EPOCHS=0,
        LR_EXP_DECAY=0.93,
        EPOCHS=20,
        PATIENCE=5,
    ):
        self.data = data
        self.strategy = strategy
        self.LR_START = LR_START
        self.LR_MAX = LR_MAX
        self.LR_MIN = LR_MIN
        self.LR_RAMPUP_EPOCHS = LR_RAMPUP_EPOCHS
        self.LR_SUSTAIN_EPOCHS = LR_SUSTAIN_EPOCHS
        self.LR_EXP_DECAY = LR_EXP_DECAY
        self.EPOCHS = EPOCHS
        self.PATIENCE = PATIENCE
        self.lr_callback = tf.keras.callbacks.LearningRateScheduler(
            self.lrfn, verbose=0
        )

    def lrfn(self, epoch):
        if epoch < self.LR_RAMPUP_EPOCHS:
            self.lr = (
                self.LR_MAX - self.LR_START
            ) / self.LR_RAMPUP_EPOCHS * epoch + self.LR_START
        elif epoch < self.LR_RAMPUP_EPOCHS + self.LR_SUSTAIN_EPOCHS:
            self.lr = self.LR_MAX
        else:
            self.lr = (self.LR_MAX - self.LR_MIN) * self.LR_EXP_DECAY ** (
                epoch - self.LR_RAMPUP_EPOCHS - self.LR_SUSTAIN_EPOCHS
            ) + self.LR_MIN
        return self.lr

    def plot_learning_schedule(self):
        rng = [i for i in range(self.EPOCHS)]
        y = [self.lrfn(x) for x in rng]
        plt.plot(rng, y)
        plt.xlabel("Epoch")
        plt.ylabel("Learning rate")
        plt.suptitle(
            "Learning rate schedule: {:.3g} to {:.3g} to {:.3g}".format(
                y[0], max(y), y[-1]
            )
        )

    def create_model(self):
        with self.strategy.scope():

            pretrained_model = tf.keras.applications.ResNet50(
                weights="imagenet", include_top=False
            )

            self.model = tf.keras.Sequential(
                [
                    # convert image format from int [0,255] to the format expected by this model
                    tf.keras.layers.Lambda(
                        lambda data: tf.keras.applications.resnet.preprocess_input(
                            tf.cast(data, tf.float32)
                        ),
                        input_shape=(
                            self.data.img_height,
                            self.data.img_width,
                            self.data.channels,
                        ),
                    ),
                    pretrained_model,
                    tf.keras.layers.GlobalAveragePooling2D(),
                    tf.keras.layers.Dense(
                        len(self.data.class_names),
                        activation="softmax",
                        name="particle_prob",
                    ),
                ]
            )

            self.model.compile(
                optimizer="adam",
                loss="sparse_categorical_crossentropy",
                metrics=["sparse_categorical_accuracy"],
                steps_per_execution=8,
            )
            self.model.summary()

    def train_and_evaluate(self):
        STEPS_PER_EPOCH = (
            self.data.train_ds_size // self.data.batch_size
        )  # this is not working since the dataset is too small
        VALIDATION_STEPS = -(
            -self.data.val_ds_size // self.data.batch_size
        )  # The "-(-//)" trick rounds up instead of down :-)
        self.history = self.model.fit(
            self.data.train_ds,
            epochs=self.EPOCHS,
            validation_data=self.data.val_ds,
            validation_steps=VALIDATION_STEPS,
            callbacks=[
                self.lr_callback,
                tf.keras.callbacks.EarlyStopping(patience=self.PATIENCE),
            ],
            class_weight=self.data.class_weights,
            verbose=0,
        )


class RESNET50_FROM_ZERO:
    def __init__(
        self,
        data: Data,
        strategy,
        LR_START,
        LR_MAX,
        LR_MIN,
        LR_RAMPUP_EPOCHS=0,
        LR_SUSTAIN_EPOCHS=0,
        LR_EXP_DECAY=0.93,
        EPOCHS=20,
        PATIENCE=5,
    ):
        self.data = data
        self.strategy = strategy
        self.LR_START = LR_START
        self.LR_MAX = LR_MAX
        self.LR_MIN = LR_MIN
        self.LR_RAMPUP_EPOCHS = LR_RAMPUP_EPOCHS
        self.LR_SUSTAIN_EPOCHS = LR_SUSTAIN_EPOCHS
        self.LR_EXP_DECAY = LR_EXP_DECAY
        self.EPOCHS = EPOCHS
        self.PATIENCE = PATIENCE
        self.lr_callback = tf.keras.callbacks.LearningRateScheduler(
            self.lrfn, verbose=0
        )

    def lrfn(self, epoch):
        if epoch < self.LR_RAMPUP_EPOCHS:
            self.lr = (
                self.LR_MAX - self.LR_START
            ) / self.LR_RAMPUP_EPOCHS * epoch + self.LR_START
        elif epoch < self.LR_RAMPUP_EPOCHS + self.LR_SUSTAIN_EPOCHS:
            self.lr = self.LR_MAX
        else:
            self.lr = (self.LR_MAX - self.LR_MIN) * self.LR_EXP_DECAY ** (
                epoch - self.LR_RAMPUP_EPOCHS - self.LR_SUSTAIN_EPOCHS
            ) + self.LR_MIN
        return self.lr

    def plot_learning_schedule(self):
        rng = [i for i in range(self.EPOCHS)]
        y = [self.lrfn(x) for x in rng]
        plt.plot(rng, y)
        plt.xlabel("Epoch")
        plt.ylabel("Learning rate")
        plt.suptitle(
            "Learning rate schedule: {:.3g} to {:.3g} to {:.3g}".format(
                y[0], max(y), y[-1]
            )
        )

    def create_model(self):
        with self.strategy.scope():

            pretrained_model = tf.keras.applications.ResNet50(
                weights=None, include_top=False
            )

            self.model = tf.keras.Sequential(
                [
                    # convert image format from int [0,255] to the format expected by this model
                    tf.keras.layers.Lambda(
                        lambda data: tf.keras.applications.resnet.preprocess_input(
                            tf.cast(data, tf.float32)
                        ),
                        input_shape=(
                            self.data.img_height,
                            self.data.img_width,
                            self.data.channels,
                        ),
                    ),
                    pretrained_model,
                    tf.keras.layers.GlobalAveragePooling2D(),
                    tf.keras.layers.Dense(
                        len(self.data.class_names),
                        activation="softmax",
                        name="particle_prob",
                    ),
                ]
            )

            self.model.compile(
                optimizer="adam",
                loss="sparse_categorical_crossentropy",
                metrics=["sparse_categorical_accuracy"],
                steps_per_execution=8,
            )
            self.model.summary()

    def train_and_evaluate(self):
        STEPS_PER_EPOCH = (
            self.data.train_ds_size // self.data.batch_size
        )  # this is not working since the dataset is too small
        VALIDATION_STEPS = -(
            -self.data.val_ds_size // self.data.batch_size
        )  # The "-(-//)" trick rounds up instead of down :-)
        self.history = self.model.fit(
            self.data.train_ds,
            epochs=self.EPOCHS,
            validation_data=self.data.val_ds,
            validation_steps=VALIDATION_STEPS,
            callbacks=[
                self.lr_callback,
                tf.keras.callbacks.EarlyStopping(patience=self.PATIENCE),
            ],
            class_weight=self.data.class_weights,
            verbose=0,
        )


class INCEPTIONV3_FINE_TUNE:
    def __init__(
        self,
        data: Data,
        strategy,
        LR_START,
        LR_MAX,
        LR_MIN,
        LR_RAMPUP_EPOCHS=0,
        LR_SUSTAIN_EPOCHS=0,
        LR_EXP_DECAY=0.93,
        EPOCHS=20,
        PATIENCE=5,
    ):
        self.data = data
        self.strategy = strategy
        self.LR_START = LR_START
        self.LR_MAX = LR_MAX
        self.LR_MIN = LR_MIN
        self.LR_RAMPUP_EPOCHS = LR_RAMPUP_EPOCHS
        self.LR_SUSTAIN_EPOCHS = LR_SUSTAIN_EPOCHS
        self.LR_EXP_DECAY = LR_EXP_DECAY
        self.EPOCHS = EPOCHS
        self.PATIENCE = PATIENCE
        self.lr_callback = tf.keras.callbacks.LearningRateScheduler(
            self.lrfn, verbose=0
        )

    def lrfn(self, epoch):
        if epoch < self.LR_RAMPUP_EPOCHS:
            self.lr = (
                self.LR_MAX - self.LR_START
            ) / self.LR_RAMPUP_EPOCHS * epoch + self.LR_START
        elif epoch < self.LR_RAMPUP_EPOCHS + self.LR_SUSTAIN_EPOCHS:
            self.lr = self.LR_MAX
        else:
            self.lr = (self.LR_MAX - self.LR_MIN) * self.LR_EXP_DECAY ** (
                epoch - self.LR_RAMPUP_EPOCHS - self.LR_SUSTAIN_EPOCHS
            ) + self.LR_MIN
        return self.lr

    def plot_learning_schedule(self):
        rng = [i for i in range(self.EPOCHS)]
        y = [self.lrfn(x) for x in rng]
        plt.plot(rng, y)
        plt.xlabel("Epoch")
        plt.ylabel("Learning rate")
        plt.suptitle(
            "Learning rate schedule: {:.3g} to {:.3g} to {:.3g}".format(
                y[0], max(y), y[-1]
            )
        )

    def create_model(self):
        with self.strategy.scope():

            pretrained_model = tf.keras.applications.InceptionV3(
                weights="imagenet", include_top=False
            )

            self.model = tf.keras.Sequential(
                [
                    # convert image format from int [0,255] to the format expected by this model
                    tf.keras.layers.Lambda(
                        lambda data: tf.keras.applications.inception_v3.preprocess_input(
                            tf.cast(data, tf.float32)
                        ),
                        input_shape=(
                            self.data.img_height,
                            self.data.img_width,
                            self.data.channels,
                        ),
                    ),
                    pretrained_model,
                    tf.keras.layers.GlobalAveragePooling2D(),
                    tf.keras.layers.Dense(
                        len(self.data.class_names),
                        activation="softmax",
                        name="particle_prob",
                    ),
                ]
            )

            self.model.compile(
                optimizer="adam",
                loss="sparse_categorical_crossentropy",
                metrics=["sparse_categorical_accuracy"],
                steps_per_execution=8,
            )
            self.model.summary()

    def train_and_evaluate(self):
        STEPS_PER_EPOCH = (
            self.data.train_ds_size // self.data.batch_size
        )  # this is not working since the dataset is too small
        VALIDATION_STEPS = -(
            -self.data.val_ds_size // self.data.batch_size
        )  # The "-(-//)" trick rounds up instead of down :-)
        self.history = self.model.fit(
            self.data.train_ds,
            epochs=self.EPOCHS,
            validation_data=self.data.val_ds,
            validation_steps=VALIDATION_STEPS,
            callbacks=[
                self.lr_callback,
                tf.keras.callbacks.EarlyStopping(patience=self.PATIENCE),
            ],
            class_weight=self.data.class_weights,
            verbose=0,
        )


class ResNet34_Self_Made:
    def __init__(
        self,
        data: Data,
        strategy,
        LR_START,
        LR_MAX,
        LR_MIN,
        LR_RAMPUP_EPOCHS=0,
        LR_SUSTAIN_EPOCHS=0,
        LR_EXP_DECAY=0.93,
        EPOCHS=20,
        PATIENCE=5,
    ):
        self.data = data
        self.strategy = strategy
        self.LR_START = LR_START
        self.LR_MAX = LR_MAX
        self.LR_MIN = LR_MIN
        self.LR_RAMPUP_EPOCHS = LR_RAMPUP_EPOCHS
        self.LR_SUSTAIN_EPOCHS = LR_SUSTAIN_EPOCHS
        self.LR_EXP_DECAY = LR_EXP_DECAY
        self.EPOCHS = EPOCHS
        self.PATIENCE = PATIENCE
        self.lr_callback = tf.keras.callbacks.LearningRateScheduler(
            self.lrfn, verbose=0
        )

    def identity_block(self, x, filter):
        # copy tensor to variable called x_skip
        x_skip = x
        # Layer 1
        x = tf.keras.layers.Conv2D(filter, (3, 3), padding="same")(x)
        x = tf.keras.layers.BatchNormalization(axis=3)(x)
        x = tf.keras.layers.Activation("relu")(x)
        # Layer 2
        x = tf.keras.layers.Conv2D(filter, (3, 3), padding="same")(x)
        x = tf.keras.layers.BatchNormalization(axis=3)(x)
        # Add Residue
        x = tf.keras.layers.Add()([x, x_skip])
        x = tf.keras.layers.Activation("relu")(x)
        return x

    def convolutional_block(self, x, filter):
        # copy tensor to variable called x_skip
        x_skip = x
        # Layer 1
        x = tf.keras.layers.Conv2D(filter, (3, 3), padding="same", strides=(2, 2))(x)
        x = tf.keras.layers.BatchNormalization(axis=3)(x)
        x = tf.keras.layers.Activation("relu")(x)
        # Layer 2
        x = tf.keras.layers.Conv2D(filter, (3, 3), padding="same")(x)
        x = tf.keras.layers.BatchNormalization(axis=3)(x)
        # Processing Residue with conv(1,1)
        x_skip = tf.keras.layers.Conv2D(filter, (1, 1), strides=(2, 2))(x_skip)
        # Add Residue
        x = tf.keras.layers.Add()([x, x_skip])
        x = tf.keras.layers.Activation("relu")(x)
        return x

    def lrfn(self, epoch):
        if epoch < self.LR_RAMPUP_EPOCHS:
            self.lr = (
                self.LR_MAX - self.LR_START
            ) / self.LR_RAMPUP_EPOCHS * epoch + self.LR_START
        elif epoch < self.LR_RAMPUP_EPOCHS + self.LR_SUSTAIN_EPOCHS:
            self.lr = self.LR_MAX
        else:
            self.lr = (self.LR_MAX - self.LR_MIN) * self.LR_EXP_DECAY ** (
                epoch - self.LR_RAMPUP_EPOCHS - self.LR_SUSTAIN_EPOCHS
            ) + self.LR_MIN
        return self.lr

    def plot_learning_schedule(self):
        rng = [i for i in range(self.EPOCHS)]
        y = [self.lrfn(x) for x in rng]
        plt.plot(rng, y)
        plt.xlabel("Epoch")
        plt.ylabel("Learning rate")
        plt.suptitle(
            "Learning rate schedule: {:.3g} to {:.3g} to {:.3g}".format(
                y[0], max(y), y[-1]
            )
        )

    def create_model(self):
        with self.strategy.scope():
            # Step 1 (Setup Input Layer)
            x_input = tf.keras.layers.Input(
                (self.data.img_height, self.data.img_width, self.data.channels)
            )
            x = tf.keras.layers.ZeroPadding2D((3, 3))(x_input)
            # Step 2 (Initial Conv layer along with maxPool)
            x = tf.keras.layers.Conv2D(64, kernel_size=7, strides=2, padding="same")(x)
            x = tf.keras.layers.BatchNormalization()(x)
            x = tf.keras.layers.Activation("relu")(x)
            x = tf.keras.layers.MaxPool2D(pool_size=3, strides=2, padding="same")(x)
            # Define size of sub-blocks and initial filter size
            block_layers = [3, 4, 6, 3]
            filter_size = 64
            # Step 3 Add the Resnet Blocks
            for i in range(4):
                if i == 0:
                    # For sub-block 1 Residual/Convolutional block not needed
                    for j in range(block_layers[i]):
                        x = self.identity_block(x, filter_size)
                else:
                    # One Residual/Convolutional Block followed by Identity blocks
                    # The filter size will go on increasing by a factor of 2
                    filter_size = filter_size * 2
                    x = self.convolutional_block(x, filter_size)
                    for j in range(block_layers[i] - 1):
                        x = self.identity_block(x, filter_size)
            # Step 4 End Dense Network
            x = tf.keras.layers.AveragePooling2D((2, 2), padding="same")(x)
            x = tf.keras.layers.Flatten()(x)
            x = tf.keras.layers.Dense(512, activation="relu")(x)
            x = tf.keras.layers.Dense(len(self.data.class_names), activation="softmax")(
                x
            )
            self.model = tf.keras.models.Model(
                inputs=x_input, outputs=x, name="ResNet34"
            )

            self.model.compile(
                optimizer="adam",
                loss="sparse_categorical_crossentropy",
                metrics=["sparse_categorical_accuracy"],
                steps_per_execution=8,
            )
            self.model.summary()

    def train_and_evaluate(self):
        STEPS_PER_EPOCH = (
            self.data.train_ds_size // self.data.batch_size
        )  # this is not working since the dataset is too small
        VALIDATION_STEPS = -(
            -self.data.val_ds_size // self.data.batch_size
        )  # The "-(-//)" trick rounds up instead of down :-)
        self.history = self.model.fit(
            self.data.train_ds,
            epochs=self.EPOCHS,
            validation_data=self.data.val_ds,
            validation_steps=VALIDATION_STEPS,
            callbacks=[
                self.lr_callback,
                tf.keras.callbacks.EarlyStopping(patience=self.PATIENCE),
            ],
            class_weight=self.data.class_weights,
            verbose=0,
        )


class Simple_FC_Network:
    # Pretrained MobileNET that is available on TensorFlowHub
    def __init__(
        self,
        data: Data,
        lrate=0.001,
        l1=0.0,
        l2=0.0,
        num_hidden=[64, 16],
        dropout_prob=0.4,
        epochs=5,
        patience=1,
    ):
        self.data = data
        self.lrate = lrate
        self.l1 = l1
        self.l2 = l2
        self.num_hidden = num_hidden
        self.dropout_prob = dropout_prob
        self.epochs = epochs
        self.patience = patience
        self.regularizer = tf.keras.regularizers.l1_l2(self.l1, self.l2)

    def train_and_evaluate(self):
        layers = [
            tf.keras.layers.Rescaling(
                1.0 / 255,
                input_shape=(
                    self.data.img_height,
                    self.data.img_width,
                    self.data.channels,
                ),
                name="rescaling",
            ),
            tf.keras.layers.Flatten(name="input_pixels_flattened"),
        ]

        for hno, nodes in enumerate(self.num_hidden):
            layers.extend(
                [
                    tf.keras.layers.Dense(
                        nodes,
                        kernel_regularizer=self.regularizer,
                        name="hidden_dense_{}".format(hno),
                    ),
                    tf.keras.layers.BatchNormalization(
                        scale=False,  # ReLU
                        center=False,  # have bias in Dense
                        name="batchnorm_dense_{}".format(hno),
                    ),
                    # move activation to come after batchnorm
                    tf.keras.layers.Activation(
                        "relu", name="relu_dense_{}".format(hno)
                    ),
                    tf.keras.layers.Dropout(
                        rate=self.dropout_prob, name="dropout_dense_{}".format(hno)
                    ),
                ]
            )

        layers.append(
            tf.keras.layers.Dense(
                self.data.nbr_of_classes,
                kernel_regularizer=self.regularizer,
                activation="softmax",
                name="particle_type_prob",
            )
        )

        self.model = tf.keras.Sequential(layers, name="particle_classification")
        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=self.lrate),
            loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
            metrics=["accuracy"],
        )
        print(self.model.summary())
        self.history = self.model.fit(
            self.data.train_ds,
            validation_data=self.data.val_ds,
            epochs=10,
            callbacks=[
                tf.keras.callbacks.EarlyStopping(patience=self.patience),
                PlotLearning(),
            ],
        )
