import os
import json
from caption_generator.models import *
import image_to_numpy

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

image_model = tf.keras.applications.EfficientNetB4(include_top=False, weights='imagenet')
new_input = image_model.input
hidden_layer = image_model.layers[-1].output

image_features_extract_model = tf.keras.Model(new_input, hidden_layer)

with open(os.path.dirname(__file__) + '/tokenizer.json') as f:
    data = json.load(f)
    tokenizer = tf.keras.preprocessing.text.tokenizer_from_json(data)

MAX_LENGTH = 51

enc_layers = 6
dec_layers = 6
d_model = 512
dff = 2048
num_heads = 8
dropout_rate = 0.3
max_position_encodings = 256
vocab_size = tokenizer.num_words + 1

transformer = Transformer(enc_layers, dec_layers, d_model, num_heads, dff,
                          vocab_size,
                          pe_input=max_position_encodings,
                          pe_target=max_position_encodings,
                          dropout_rate=dropout_rate)

checkpoint_path = os.path.dirname(__file__) + '/checkpoints'

ckpt = tf.train.Checkpoint(transformoer=transformer)
ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_path, max_to_keep=5)

if ckpt_manager.latest_checkpoint:
    saved_epoch = int(ckpt_manager.latest_checkpoint.split('-')[-1])
    # restoring the latest checkpoint in checkpoint_path
    ckpt.restore(ckpt_manager.latest_checkpoint)
    print('Checkpoint restored:', saved_epoch)


# Function for preprocessing
def load_image(image_path):
    # img = tf.io.read_file(image_path)
    # img = tf.image.decode_jpeg(img, channels=3)

    img = image_to_numpy.load_image_file(image_path)

    # plt.imshow(img)
    # plt.show()

    img = tf.image.resize(img, (380, 380))  # (380, 380) for efficient-netB4
    img = tf.keras.applications.efficientnet.preprocess_input(img)
    return img, image_path


def create_masks(inp, tar):
    enc_padding_mask = create_padding_mask(tf.ones(shape=(tf.shape(inp)[0], tf.shape(inp)[1])))
    dec_padding_mask = create_padding_mask(tf.ones(shape=(tf.shape(inp)[0], tf.shape(inp)[1])))
    look_ahead_mask = create_look_ahead_mask(tf.shape(tar)[1])
    dec_target_padding_mask = create_padding_mask(tar)
    combined_mask = tf.maximum(dec_target_padding_mask, look_ahead_mask)

    return enc_padding_mask, combined_mask, dec_padding_mask


def evaluate(image):
    encoder_input = tf.expand_dims(load_image(image)[0], 0)  # Expand batch axis
    encoder_input = image_features_extract_model(encoder_input)
    encoder_input = tf.reshape(encoder_input, (encoder_input.shape[0], -1, encoder_input.shape[3]))

    # as the target is english, the first word to the transformer should be the english start token.
    decoder_input = [tokenizer.word_index['<start>']]
    output = tf.expand_dims(decoder_input, 0)

    for i in range(MAX_LENGTH):
        enc_padding_mask, combined_mask, dec_padding_mask = create_masks(encoder_input, output)

        # predictions.shape == (batch_size, seq_len, vocab_size)
        predictions = transformer(encoder_input,
                                  output,
                                  False,
                                  enc_padding_mask,
                                  combined_mask,
                                  dec_padding_mask)

        # select the last word from the seq_len dimension
        predictions = predictions[:, -1:, :]  # (batch_size, 1, vocab_size)
        predicted_id = tf.cast(tf.argmax(predictions, axis=-1), tf.int32)

        # return the result if the predicted_id is equal to the end token

        if predicted_id == tokenizer.word_index['<end>']:
            return tf.squeeze(output, axis=0)

        # concatenate the predicted_id to the output which is given to the decoder as its input.
        output = tf.concat([output, predicted_id], axis=-1)

    return tf.squeeze(output, axis=0)


def decode(seq):
    predicted_caption = [tokenizer.index_word[i] for i in seq[1:] if i < vocab_size]

    return ' '.join(predicted_caption)


def generate_caption(image):
    result = evaluate(image)

    return decode(result.numpy())


print(generate_caption('C:/Users/Sopiro/Desktop/20200825/uchan.jpg'))

# if __name__ == '__main__':
#     image_url = 'https://tensorflow.org/images/surf.jpg'
#     image_extension = image_url[-4:]
#     image_path = tf.keras.utils.get_file('image' + image_extension, origin=image_url)
#
#     print('Prediction Caption :', generate_caption(image_path))
