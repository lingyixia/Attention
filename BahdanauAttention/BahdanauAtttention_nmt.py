# coding:utf-8

# -------------------------------------------------------------------------------
# @Author        chenfeiyu01
# @Name:         BahdanauAttention.py
# @Project       BahdanauAttention
# @Product       PyCharm
# @DateTime:     2019-07-18 20:14
# @Contact       chenfeiyu01@baidu.com
# @Version       1.0
# @Description:
# REF:https://github.com/tensorflow/docs/blob/master/site/en/r2/tutorials/text/nmt_with_attention.ipynb
# -------------------------------------------------------------------------------
# -*- coding: utf-8 -*-
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from sklearn.model_selection import train_test_split
import unicodedata
import re, os, io, time, argparse
import numpy as np

parser = argparse.ArgumentParser(description='翻译Attention模型超参数设置')
parser.add_argument('--batch_size', type=int, default=64, help='batchsize')
parser.add_argument('--embedding_dim', type=int, default=256, help='embeddingsize')
parser.add_argument('--units', type=int, default=1024, help='hiddensize')
parser.add_argument('--num_examples', type=int, default=16000, help='hiddensize')
parser.add_argument('--data_path', type=str, default="./cmn-eng/cmn.txt", help='dataPath')


class DataHelper(object):
    def __init__(self):
        #         path_to_zip = tf.keras.utils.get_file(
        #             'spa-eng.zip', origin='http://www.manythings.org/anki/spa-eng.zip',
        #             extract=True)
        #         self.path_to_file = os.path.dirname(path_to_zip) + "/spa-eng/spa.txt"
        self.path_to_file = FLAGS.data_path

    def unicode_to_ascii(self, s):
        return ''.join(c for c in unicodedata.normalize('NFD', s) if unicodedata.category(c) != 'Mn')

    def preprocess_sentence(self, w):
        w = self.unicode_to_ascii(w.lower().strip())
        w = re.sub(r"([?.!,])", r" \1 ", w)
        w = re.sub(r'[" "]+', " ", w)
        w = re.sub(r"[^a-zA-Z?.!,]+", " ", w)
        w = w.rstrip().strip()
        w = '<start> ' + w + ' <end>'
        return w

    def preprocess_sentence_chinese(self, w):
        w = " ".join(list(w))
        w = w.rstrip().strip()
        w = '<start> ' + w + ' <end>'
        return w

    def create_dataset(self, path, num_examples):
        lines = io.open(path, encoding='UTF-8').read().strip().split('\n')
        word_pairs = [[self.preprocess_sentence(w.split('\t')[0]), self.preprocess_sentence_chinese(w.split('\t')[1])]
                      for w in lines]
        return zip(*word_pairs)

    def max_length(self, tensor):
        return max(len(t) for t in tensor)

    def tokenize(self, lang):
        lang_tokenizer = tf.keras.preprocessing.text.Tokenizer(filters='')
        lang_tokenizer.fit_on_texts(lang)
        tensor = lang_tokenizer.texts_to_sequences(lang)
        tensor = tf.keras.preprocessing.sequence.pad_sequences(tensor, padding='post')
        return tensor, lang_tokenizer

    def load_dataset(self, path, num_examples=None):
        targ_lang, inp_lang = self.create_dataset(path, num_examples)
        input_tensor, inp_lang_tokenizer = self.tokenize(inp_lang)
        target_tensor, targ_lang_tokenizer = self.tokenize(targ_lang)
        return input_tensor, target_tensor, inp_lang_tokenizer, targ_lang_tokenizer


class Encoder(tf.keras.Model):
    def __init__(self, vocab_size, embedding_dim, enc_units, batch_sz):
        super(Encoder, self).__init__()
        self.batch_sz = batch_sz
        self.enc_units = enc_units
        self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)
        self.gru = tf.keras.layers.GRU(self.enc_units,
                                       return_sequences=True,
                                       return_state=True,
                                       recurrent_initializer='glorot_uniform')

    def call(self, x, hidden):
        x = self.embedding(x)
        output, state = self.gru(x, initial_state=hidden)
        return output, state

    def initialize_hidden_state(self):
        return tf.zeros((self.batch_sz, self.enc_units))


class BahdanauAttention(tf.keras.Model):
    def __init__(self, units):
        super(BahdanauAttention, self).__init__()
        self.W1 = tf.keras.layers.Dense(units)
        self.W2 = tf.keras.layers.Dense(units)
        self.V = tf.keras.layers.Dense(1)

    # 关键代码
    def call(self, query, values):
        hidden_with_time_axis = tf.expand_dims(query, 1)
        score = self.V(tf.nn.tanh(self.W1(values) + self.W2(hidden_with_time_axis)))
        attention_weights = tf.nn.softmax(score, axis=1)
        context_vector = attention_weights * values
        context_vector = tf.reduce_sum(context_vector, axis=1)
        return context_vector, attention_weights


class Decoder(tf.keras.Model):
    def __init__(self, vocab_size, embedding_dim, dec_units, batch_sz):
        super(Decoder, self).__init__()
        self.batch_sz = batch_sz
        self.dec_units = dec_units
        self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)
        self.gru = tf.keras.layers.GRU(self.dec_units,
                                       return_sequences=True,
                                       return_state=True,
                                       recurrent_initializer='glorot_uniform')
        self.fc = tf.keras.layers.Dense(vocab_size)
        self.attention = BahdanauAttention(self.dec_units)

    def call(self, x, hidden, enc_output):
        context_vector, attention_weights = self.attention(hidden, enc_output)
        x = self.embedding(x)
        x = tf.concat([tf.expand_dims(context_vector, 1), x], axis=-1)
        output, state = self.gru(x)
        output = tf.reshape(output, (-1, output.shape[2]))
        x = self.fc(output)
        return x, state, attention_weights


class NMT(object):
    def __init__(self, example_input_batch, embedding_dim, units, batch_size):
        self.decoder = Decoder(vocab_tar_size, embedding_dim, units, batch_size)
        self.encoder = Encoder(vocab_inp_size, embedding_dim, units, batch_size)
        sample_hidden = self.encoder.initialize_hidden_state()
        sample_output, sample_hidden = self.encoder(example_input_batch, sample_hidden)
        self.sample_decoder_output, _, _ = self.decoder(tf.random.uniform((64, 1)), sample_hidden, sample_output)
        self.optimizer = tf.keras.optimizers.Adam()
        self.loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction='none')
        self.checkpoint_dir = './training_checkpoints'
        self.checkpoint_prefix = os.path.join(self.checkpoint_dir, "ckpt")
        self.checkpoint = tf.train.Checkpoint(optimizer=self.optimizer, encoder=self.encoder, decoder=self.decoder)

    def loss_function(self, real, pred):
        mask = tf.math.logical_not(tf.math.equal(real, 0))
        loss_ = self.loss_object(real, pred)
        mask = tf.cast(mask, dtype=loss_.dtype)
        loss_ *= mask
        return tf.reduce_mean(loss_)

    @tf.function
    def __train_step(self, inp, targ, enc_hidden, batch_size):
        loss = 0
        with tf.GradientTape() as tape:
            enc_output, enc_hidden = self.encoder(inp, enc_hidden)
            dec_hidden = enc_hidden
            dec_input = tf.expand_dims([targ_lang.word_index['<start>']] * batch_size, 1)
            for t in range(1, targ.shape[1]):
                predictions, dec_hidden, _ = self.decoder(dec_input, dec_hidden, enc_output)
                loss += self.loss_function(targ[:, t], predictions)
                dec_input = tf.expand_dims(targ[:, t], 1)
        batch_loss = (loss / int(targ.shape[1]))
        variables = self.encoder.trainable_variables + self.decoder.trainable_variables
        gradients = tape.gradient(loss, variables)
        self.optimizer.apply_gradients(zip(gradients, variables))
        return batch_loss

    def train(self, batch_size, dataset):
        EPOCHS = 3
        for epoch in range(EPOCHS):
            start = time.time()
            enc_hidden = self.encoder.initialize_hidden_state()
            total_loss = 0
            for (batch, (inp, targ)) in enumerate(dataset.take(steps_per_epoch)):
                batch_loss = self.__train_step(inp, targ, enc_hidden, batch_size)
                total_loss += batch_loss
                if batch % 100 == 0:
                    print('Epoch {} Batch {} Loss {:.4f}'.format(epoch + 1,
                                                                 batch,
                                                                 batch_loss.numpy()))
            if (epoch + 1) % 2 == 0:
                self.checkpoint.save(file_prefix=self.checkpoint_prefix)
            print('Epoch {} Loss {:.4f}'.format(epoch + 1, total_loss / steps_per_epoch))
            print('Time taken for 1 epoch {} sec\n'.format(time.time() - start))

    def translate(self, sentence, units):
        result, sentence, attention_plot = self.evaluate(sentence, units)
        print('Input: %s' % (sentence))
        print('Predicted translation: {}'.format(result))
        attention_plot = attention_plot[:len(result.split(' ')), :len(sentence.split(' '))]
        self.plot_attention(attention_plot, sentence.split(' '), result.split(' '))

    def plot_attention(self, attention, sentence, predicted_sentence):
        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(1, 1, 1)
        ax.matshow(attention, cmap='viridis')
        fontdict = {'fontsize': 14}
        ax.set_xticklabels([''] + sentence, fontdict=fontdict, rotation=90)
        ax.set_yticklabels([''] + predicted_sentence, fontdict=fontdict)
        ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
        ax.yaxis.set_major_locator(ticker.MultipleLocator(1))
        plt.show()

    def evaluate(self, sentence, units):
        attention_plot = np.zeros((max_length_targ, max_length_inp))
        inputs = [inp_lang.word_index[i] for i in sentence.split(' ')]
        inputs = tf.keras.preprocessing.sequence.pad_sequences([inputs], maxlen=max_length_inp, padding='post')
        inputs = tf.convert_to_tensor(inputs)
        result = ''
        hidden = [tf.zeros((1, units))]
        enc_out, enc_hidden = self.encoder(inputs, hidden)
        dec_hidden = enc_hidden
        dec_input = tf.expand_dims([targ_lang.word_index['<start>']], 0)
        for t in range(max_length_targ):
            predictions, dec_hidden, attention_weights = self.decoder(dec_input, dec_hidden, enc_out)
            attention_weights = tf.reshape(attention_weights, (-1,))
            attention_plot[t] = attention_weights.numpy()
            predicted_id = tf.argmax(predictions[0]).numpy()
            result += targ_lang.index_word[predicted_id] + ' '
            if targ_lang.index_word[predicted_id] == '<end>':
                return result, sentence, attention_plot
            dec_input = tf.expand_dims([predicted_id], 0)
        return result, sentence, attention_plot


if __name__ == '__main__':
    FLAGS = parser.parse_args()
    data_helper = DataHelper()
    input_tensor, target_tensor, inp_lang, targ_lang = data_helper.load_dataset(data_helper.path_to_file,
                                                                                FLAGS.num_examples)
    max_length_targ, max_length_inp = data_helper.max_length(target_tensor), data_helper.max_length(input_tensor)
    input_tensor_train, input_tensor_val, target_tensor_train, target_tensor_val = train_test_split(input_tensor,
                                                                                                    target_tensor,
                                                                                                    test_size=0.2)

    print(len(input_tensor_train), len(target_tensor_train), len(input_tensor_val), len(target_tensor_val))
    BUFFER_SIZE = len(input_tensor_train)
    steps_per_epoch = len(input_tensor_train) // FLAGS.batch_size
    vocab_inp_size = len(inp_lang.word_index) + 1
    vocab_tar_size = len(targ_lang.word_index) + 1
    dataset = tf.data.Dataset.from_tensor_slices((input_tensor_train, target_tensor_train)).shuffle(BUFFER_SIZE)
    dataset = dataset.batch(FLAGS.batch_size, drop_remainder=True)
    example_input_batch, example_target_batch = next(iter(dataset))
    nmt = NMT(example_input_batch, FLAGS.embedding_dim, FLAGS.units, FLAGS.batch_size)
    nmt.train(FLAGS.batch_size, dataset)
    nmt.checkpoint.restore(tf.train.latest_checkpoint(nmt.checkpoint_dir))
    origin = data_helper.preprocess_sentence('你 现 在 在 哪 里 ?')
    nmt.translate(origin, FLAGS.units)
