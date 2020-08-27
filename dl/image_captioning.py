import tensorflow as tf

from sklearn.utils import shuffle

import numpy as np
import os
import json
import models
import time


class CaptionGenerator:

    def __init__(self):
        gpus = tf.config.experimental.list_physical_devices('GPU')
        if gpus:
            try:
                # Currently, memory growth needs to be the same across GPUs
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
                logical_gpus = tf.config.experimental.list_logical_devices('GPU')
                print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
            except RuntimeError as e:
                # Memory growth must be set before GPUs have been initialized
                print(e)

        # Download caption annotation files
        annotation_folder = './annotations/'
        if not os.path.exists(os.path.abspath('.') + annotation_folder):
            annotation_zip = tf.keras.utils.get_file('captions.zip',
                                                     cache_subdir=os.path.abspath('..'),
                                                     origin='http://images.cocodataset.org/annotations/annotations_trainval2014.zip',
                                                     extract=True)
            annotation_file = os.path.dirname(annotation_zip) + '/annotations/captions_train2014.json'
            os.remove(annotation_zip)
        else:
            annotation_file = os.path.abspath('.') + '/annotations/captions_train2014.json'

        # Read the json file
        with open(annotation_file, 'r') as f:
            annotations = json.load(f)

        # Store captions and image names in vectors
        all_captions = []

        for annot in annotations['annotations']:
            caption = '<start> ' + annot['caption'] + ' <end>'
            all_captions.append(caption)

        # Shuffle captions and image_names together
        # Set a random state, which always guaranteed to have the same shuffle
        train_captions = shuffle(all_captions, random_state=1)

        # Initialize Inception-V3 with pretrained weight
        image_model = tf.keras.applications.InceptionV3(include_top=False, weights='imagenet')
        new_input = image_model.input
        hidden_layer = image_model.layers[-1].output

        self.image_features_extract_model = tf.keras.Model(new_input, hidden_layer)

        # Choose the top 5000 words from the vocabulary
        num_words = 20000
        self.tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words=num_words, oov_token="<unk>", filters='!"#$%&()*+.,-/:;=?@[\]^_`{|}~ ')
        self.tokenizer.fit_on_texts(train_captions)

        self.tokenizer.word_index['<pad>'] = 0
        self.tokenizer.index_word[0] = '<pad>'

        train_seqs = self.tokenizer.texts_to_sequences(train_captions)
        self.max_length = self.calc_max_length(train_seqs)

        embedding_dim = 128
        units = 512
        vocab_size = num_words + 1

        self.encoder = models.CNN_Encoder(embedding_dim)
        self.decoder = models.RNN_Decoder(embedding_dim, units, vocab_size)
        self.optimizer = tf.keras.optimizers.Adam()

        # Checkpoints
        checkpoint_path = "./checkpoints"
        ckpt = tf.train.Checkpoint(encoder=self.encoder,
                                   decoder=self.decoder,
                                   optimizer=self.optimizer)
        ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_path, max_to_keep=10)

        if ckpt_manager.latest_checkpoint:
            ckpt.restore(ckpt_manager.latest_checkpoint)

    # Function for preprocessing
    def load_image(self, image_path):
        img = tf.io.read_file(image_path)
        img = tf.image.decode_jpeg(img, channels=3)
        img = tf.image.resize(img, (299, 299))
        img = tf.keras.applications.inception_v3.preprocess_input(img)
        return img, image_path

    # Find the maximum length of any caption in our dataset
    def calc_max_length(self, tensor):
        return max(len(t) for t in tensor)

    def evaluate(self, image):
        temp_input = tf.expand_dims(self.load_image(image)[0], 0)  # Expand batch axis
        img_tensor_val = self.image_features_extract_model(temp_input)
        img_tensor_val = tf.reshape(img_tensor_val, (img_tensor_val.shape[0], -1, img_tensor_val.shape[3]))

        features = self.encoder(img_tensor_val)  # shape = (1, 64, embedding_dim)

        hidden = self.decoder.reset_state(batch_size=1)
        dec_input = tf.expand_dims([self.tokenizer.word_index['<start>']], 0)
        result = []

        for i in range(self.max_length):
            predictions, hidden, attention_weights = self.decoder(dec_input, features, hidden)

            # predicted_id = tf.random.categorical(predictions, 1)[0][0].numpy()
            predicted_id = tf.argmax(predictions[0]).numpy()
            if self.tokenizer.index_word[predicted_id] == '<end>':
                return result

            result.append(self.tokenizer.index_word[predicted_id])
            dec_input = tf.expand_dims([predicted_id], 0)

        return result


if __name__ == '__main__':
    image_url = 'https://tensorflow.org/images/surf.jpg'
    image_extension = image_url[-4:]
    image_path = tf.keras.utils.get_file('image' + image_extension, origin=image_url)

    cg = CaptionGenerator()

    result = cg.evaluate(image_path)
    print('Prediction Caption :', ' '.join(result))
