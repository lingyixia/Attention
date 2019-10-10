# coding:utf-8

# -------------------------------------------------------------------------------
# @Author        chenfeiyu01
# @Name:         Transform.py
# @Project       Attention
# @Product       PyCharm
# @DateTime:     2019-07-18 20:17
# @Contact       chenfeiyu01@baidu.com
# @Version       1.0
# @Description:
# -------------------------------------------------------------------------------
# -*- coding: utf-8 -*-
import tensorflow_datasets as tfds
import tensorflow as tf

import time, argparse, functools
import numpy as np
import matplotlib.pyplot as plt

tf.enable_eager_execution()
parser = argparse.ArgumentParser(description='翻译Transform模型超参数设置')
parser.add_argument('--batch_size', type=int, default=64, help='batchsize')
parser.add_argument('--embedding_dim', type=int, default=256, help='embeddingsize')
parser.add_argument('--max_length', type=int, default=40, help='max_length')
parser.add_argument('--buffer_size', type=int, default=20000, help='buffer_size')
parser.add_argument('--data_path', type=str, default="./cmn-eng/cmn.txt", help='dataPath')
parser.add_argument('--num_layers', type=int, default=4, help='dataPath')
parser.add_argument('--d_model', type=int, default=128, help='dataPath')
parser.add_argument('--dff', type=int, default=512, help='dataPath')
parser.add_argument('--num_heads', type=int, default=8, help='dataPath')
parser.add_argument('--dropout_rate', type=int, default=0.1, help='dataPath')


class DataHelper(object):
    def __init__(self):
        examples = tfds.load('ted_hrlr_translate/pt_to_en', as_supervised=True)
        self.train_examples, self.val_examples = examples['train'], examples['validation']
        self.tokenizer_en = tfds.features.text.SubwordTextEncoder.build_from_corpus(
            (en.numpy() for pt, en in self.train_examples), target_vocab_size=2 ** 13)
        self.tokenizer_pt = tfds.features.text.SubwordTextEncoder.build_from_corpus(
            (pt.numpy() for pt, en in self.train_examples), target_vocab_size=2 ** 13)

    def get_datasets(self, max_length, batch_size):
        train_dataset = data_helper.train_examples.map(self.__tf_encode)
        train_dataset = train_dataset.filter(
            functools.partial(self.__filter_max_length, max_length=max_length))
        train_dataset = train_dataset.cache()
        train_dataset = train_dataset.shuffle(FLAGS.buffer_size).padded_batch(batch_size, padded_shapes=([-1], [-1]))
        train_dataset = train_dataset.prefetch(tf.data.experimental.AUTOTUNE)
        return train_dataset

    def __encode(self, lang1, lang2):
        lang1 = [self.tokenizer_pt.vocab_size] + self.tokenizer_pt.encode(
            lang1.numpy()) + [self.tokenizer_pt.vocab_size + 1]
        lang2 = [self.tokenizer_en.vocab_size] + self.tokenizer_en.encode(
            lang2.numpy()) + [self.tokenizer_en.vocab_size + 1]
        return lang1, lang2

    def __filter_max_length(self, x, y, max_length):
        return tf.logical_and(tf.size(x) <= max_length, tf.size(y) <= max_length)

    def __tf_encode(self, pt, en):
        return tf.py_function(self.__encode, [pt, en], [tf.int64, tf.int64])


class MultiHeadAttention(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.d_model = d_model
        self.depth = d_model // self.num_heads
        self.wq = tf.keras.layers.Dense(d_model)
        self.wk = tf.keras.layers.Dense(d_model)
        self.wv = tf.keras.layers.Dense(d_model)
        self.dense = tf.keras.layers.Dense(d_model)

    def split_heads(self, x, batch_size):
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.depth))
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def call(self, v, k, q, mask):
        batch_size = tf.shape(q)[0]
        q = self.wq(q)  # (batch_size, seq_len, d_model)
        k = self.wk(k)  # (batch_size, seq_len, d_model)
        v = self.wv(v)  # (batch_size, seq_len, d_model)
        q = self.split_heads(q, batch_size)  # (batch_size, num_heads, seq_len_q, depth)
        k = self.split_heads(k, batch_size)  # (batch_size, num_heads, seq_len_k, depth)
        v = self.split_heads(v, batch_size)  # (batch_size, num_heads, seq_len_v, depth)
        scaled_attention, attention_weights = self.scaled_dot_product_attention(q, k, v, mask)
        scaled_attention = tf.transpose(scaled_attention, perm=[0, 2, 1, 3])
        concat_attention = tf.reshape(scaled_attention, (batch_size, -1, self.d_model))
        output = self.dense(concat_attention)  # (batch_size, seq_len_q, d_model)
        return output, attention_weights

    def scaled_dot_product_attention(self, q, k, v, mask):
        matmul_qk = tf.matmul(q, k, transpose_b=True)  # (..., seq_len_q, seq_len_k)
        dk = tf.cast(tf.shape(k)[-1], tf.float32)
        scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)
        if mask is not None:
            scaled_attention_logits += (mask * -1e9)
        attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)  # (..., seq_len_q, seq_len_k)
        output = tf.matmul(attention_weights, v)  # (..., seq_len_q, depth_v)
        return output, attention_weights


class EncoderLayer(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads, dff, rate=0.1):
        super(EncoderLayer, self).__init__()
        self.mha = MultiHeadAttention(d_model, num_heads)
        self.ffn = self.point_wise_feed_forward_network(d_model, dff)
        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = tf.keras.layers.Dropout(rate)
        self.dropout2 = tf.keras.layers.Dropout(rate)

    def point_wise_feed_forward_network(self, d_model, dff):
        return tf.keras.Sequential([
            tf.keras.layers.Dense(dff, activation='relu'),  # (batch_size, seq_len, dff)
            tf.keras.layers.Dense(d_model)  # (batch_size, seq_len, d_model)
        ])

    def call(self, x, training, mask):
        attn_output, _ = self.mha(x, x, x, mask)  # (batch_size, input_seq_len, d_model)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(x + attn_output)  # 第一个残差忘了  # (batch_size, input_seq_len, d_model)
        ffn_output = self.ffn(out1)  # (batch_size, input_seq_len, d_model)
        ffn_output = self.dropout2(ffn_output, training=training)
        out2 = self.layernorm2(out1 + ffn_output)  # 第二个残差网络  # (batch_size, input_seq_len, d_model)
        return out2


class DecoderLayer(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads, dff, rate=0.1):
        super(DecoderLayer, self).__init__()
        self.mha1 = MultiHeadAttention(d_model, num_heads)
        self.mha2 = MultiHeadAttention(d_model, num_heads)
        self.ffn = self.point_wise_feed_forward_network(d_model, dff)
        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm3 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = tf.keras.layers.Dropout(rate)
        self.dropout2 = tf.keras.layers.Dropout(rate)
        self.dropout3 = tf.keras.layers.Dropout(rate)

    def point_wise_feed_forward_network(self, d_model, dff):  # 前馈网络其实就是先正达维度在减少到原来的维度，前后shape不变
        return tf.keras.Sequential([
            tf.keras.layers.Dense(dff, activation='relu'),  # (batch_size, seq_len, dff)
            tf.keras.layers.Dense(d_model)  # (batch_size, seq_len, d_model)
        ])

    def call(self, x, enc_output, training, look_ahead_mask, padding_mask):
        attn1, attn_weights_block1 = self.mha1(x,
                                               x,
                                               x,
                                               look_ahead_mask)  # Self-Attention:v，k，q一样(batch_size, target_seq_len, d_model)
        attn1 = self.dropout1(attn1, training=training)
        out1 = self.layernorm1(attn1 + x)  # 第一个残差网络
        attn2, attn_weights_block2 = self.mha2(enc_output,
                                               enc_output,
                                               out1,
                                               padding_mask)  # 普通Attention  # (batch_size, target_seq_len, d_model)
        attn2 = self.dropout2(attn2, training=training)
        out2 = self.layernorm2(attn2 + out1)  # 第二个残差网络  # (batch_size, target_seq_len, d_model)
        ffn_output = self.ffn(out2)  # (batch_size, target_seq_len, d_model)
        ffn_output = self.dropout3(ffn_output, training=training)
        out3 = self.layernorm3(ffn_output + out2)  # (batch_size, target_seq_len, d_model)
        return out3, attn_weights_block1, attn_weights_block2


class Encoder(tf.keras.layers.Layer):
    def __init__(self, num_layers, d_model, num_heads, dff, input_vocab_size, rate=0.1):
        super(Encoder, self).__init__()
        self.d_model = d_model
        self.num_layers = num_layers
        self.embedding = tf.keras.layers.Embedding(input_vocab_size, d_model)
        self.pos_encoding = self.positional_encoding(input_vocab_size, self.d_model)
        self.enc_layers = [EncoderLayer(d_model, num_heads, dff, rate) for _ in range(num_layers)]
        self.dropout = tf.keras.layers.Dropout(rate)

    def call(self, x, training, mask):
        seq_len = tf.shape(x)[1]
        x = self.embedding(x)  # (batch_size, input_seq_len, d_model)
        x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
        x += self.pos_encoding[:, :seq_len, :]
        x = self.dropout(x, training=training)
        for i in range(self.num_layers):
            x = self.enc_layers[i](x, training, mask)
        return x  # (batch_size, input_seq_len, d_model)

    def get_angles(self, pos, i, d_model):
        angle_rates = 1 / np.power(10000, (2 * (i // 2)) / np.float32(d_model))
        return pos * angle_rates

    def positional_encoding(self, position, d_model):
        angle_rads = self.get_angles(np.arange(position)[:, np.newaxis], np.arange(d_model)[np.newaxis, :], d_model)
        sines = np.sin(angle_rads[:, 0::2])
        cosines = np.cos(angle_rads[:, 1::2])
        pos_encoding = np.concatenate([sines, cosines], axis=-1)
        pos_encoding = pos_encoding[np.newaxis, ...]
        return tf.cast(pos_encoding, dtype=tf.float32)


class Decoder(tf.keras.layers.Layer):
    def __init__(self, num_layers, d_model, num_heads, dff, target_vocab_size,
                 rate=0.1):
        super(Decoder, self).__init__()
        self.d_model = d_model
        self.num_layers = num_layers
        self.embedding = tf.keras.layers.Embedding(target_vocab_size, d_model)
        self.pos_encoding = self.positional_encoding(target_vocab_size, self.d_model)
        self.dec_layers = [DecoderLayer(d_model, num_heads, dff, rate) for _ in range(num_layers)]
        self.dropout = tf.keras.layers.Dropout(rate)

    def call(self, x, enc_output, training, look_ahead_mask, padding_mask):
        seq_len = tf.shape(x)[1]
        attention_weights = {}
        x = self.embedding(x)  # (batch_size, target_seq_len, d_model)
        x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
        x += self.pos_encoding[:, :seq_len, :]
        x = self.dropout(x, training=training)
        for i in range(self.num_layers):
            x, block1, block2 = self.dec_layers[i](x, enc_output, training, look_ahead_mask, padding_mask)
            attention_weights['decoder_layer{}_block1'.format(i + 1)] = block1
            attention_weights['decoder_layer{}_block2'.format(i + 1)] = block2
        return x, attention_weights

    def get_angles(self, pos, i, d_model):
        angle_rates = 1 / np.power(10000, (2 * (i // 2)) / np.float32(d_model))
        return pos * angle_rates

    def positional_encoding(self, position, d_model):
        angle_rads = self.get_angles(np.arange(position)[:, np.newaxis], np.arange(d_model)[np.newaxis, :], d_model)
        sines = np.sin(angle_rads[:, 0::2])
        cosines = np.cos(angle_rads[:, 1::2])
        pos_encoding = np.concatenate([sines, cosines], axis=-1)
        pos_encoding = pos_encoding[np.newaxis, ...]
        return tf.cast(pos_encoding, dtype=tf.float32)


class Transformer(tf.keras.Model):
    def __init__(self, num_layers, d_model, num_heads, dff, input_vocab_size, target_vocab_size, rate=0.1):
        super(Transformer, self).__init__()
        self.encoder = Encoder(num_layers, d_model, num_heads, dff, input_vocab_size, rate)
        self.decoder = Decoder(num_layers, d_model, num_heads, dff, target_vocab_size, rate)
        self.final_layer = tf.keras.layers.Dense(target_vocab_size)

    def call(self, inp, tar, training, enc_padding_mask, look_ahead_mask, dec_padding_mask):
        enc_output = self.encoder(inp, training, enc_padding_mask)  # (batch_size, inp_seq_len, d_model)
        dec_output, attention_weights = self.decoder(tar, enc_output, training, look_ahead_mask, dec_padding_mask)
        final_output = self.final_layer(dec_output)  # (batch_size, tar_seq_len, target_vocab_size)
        return final_output, attention_weights


class CustomSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, d_model, warmup_steps=4000):
        super(CustomSchedule, self).__init__()
        self.d_model = d_model
        self.d_model = tf.cast(self.d_model, tf.float32)
        self.warmup_steps = warmup_steps

    def __call__(self, step):
        arg1 = tf.math.rsqrt(step)
        arg2 = step * (self.warmup_steps ** -1.5)
        return tf.math.rsqrt(self.d_model) * tf.math.minimum(arg1, arg2)


train_step_signature = [tf.TensorSpec(shape=(None, None), dtype=tf.int64),
                        tf.TensorSpec(shape=(None, None), dtype=tf.int64)]


class NMT(object):
    def __init__(self, tokenizer_pt, tokenizer_en, max_length, num_layers, d_model, dff, num_heads, dropout_rate):
        self.max_length = max_length
        self.tokenizer_pt = tokenizer_pt
        self.tokenizer_en = tokenizer_en
        self.train_loss = tf.keras.metrics.Mean(name='train_loss')
        self.train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')
        self.transformer = Transformer(num_layers,
                                       d_model,
                                       num_heads,
                                       dff,
                                       tokenizer_pt.vocab_size + 2,
                                       tokenizer_en.vocab_size + 2,
                                       dropout_rate)
        checkpoint_path = "./checkpoints/train"
        learning_rate = CustomSchedule(d_model)
        self.optimizer = tf.keras.optimizers.Adam(learning_rate, beta_1=0.9, beta_2=0.98, epsilon=1e-9)
        ckpt = tf.train.Checkpoint(transformer=self.transformer, optimizer=self.optimizer)
        self.ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_path, max_to_keep=5)
        if self.ckpt_manager.latest_checkpoint:
            ckpt.restore(self.ckpt_manager.latest_checkpoint)
            print('Latest checkpoint restored!!')

    @tf.function(input_signature=train_step_signature)
    def train_step(self, inp, tar):
        tar_inp = tar[:, :-1]
        tar_real = tar[:, 1:]
        enc_padding_mask, combined_mask, dec_padding_mask = self.create_masks(inp, tar_inp)
        with tf.GradientTape() as tape:
            predictions, _ = self.transformer(inp, tar_inp,
                                              True,
                                              enc_padding_mask,
                                              combined_mask,
                                              dec_padding_mask)
            loss = self.loss_function(tar_real, predictions)
        gradients = tape.gradient(loss, self.transformer.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.transformer.trainable_variables))
        self.train_loss(loss)
        self.train_accuracy(tar_real, predictions)

    def loss_function(self, real, pred):
        loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction='none')
        mask = tf.math.logical_not(tf.math.equal(real, 0))
        loss_ = loss_object(real, pred)
        mask = tf.cast(mask, dtype=loss_.dtype)
        loss_ *= mask
        return tf.reduce_mean(loss_)

    def train(self, train_dataset):
        EPOCHS = 2
        for epoch in range(EPOCHS):
            start = time.time()
            self.train_loss.reset_states()
            self.train_accuracy.reset_states()
            for (batch, (inp, tar)) in enumerate(train_dataset):
                self.train_step(inp, tar)
                if batch % 50 == 0:
                    print('Epoch {} Batch {} Loss {:.4f} Accuracy {:.4f}'.format(
                        epoch + 1, batch, self.train_loss.result(), self.train_accuracy.result()))
            if (epoch + 1) % 5 == 0:
                ckpt_save_path = self.ckpt_manager.save()
                print('Saving checkpoint for epoch {} at {}'.format(epoch + 1, ckpt_save_path))
            print(
                'Epoch {} Loss {:.4f} Accuracy {:.4f}'.format(epoch + 1, self.train_loss.result(),
                                                              self.train_accuracy.result()))
            print('Time taken for 1 epoch: {} secs\n'.format(time.time() - start))

    def translate(self, sentence, plot=''):
        result, attention_weights = self.evaluate(sentence)
        predicted_sentence = self.tokenizer_en.decode([i for i in result if i < self.tokenizer_en.vocab_size])
        print('Input: {}'.format(sentence))
        print('Predicted translation: {}'.format(predicted_sentence))
        if plot:
            self.plot_attention_weights(attention_weights, sentence, result, plot)

    def plot_attention_weights(self, attention, sentence, result, layer):
        fig = plt.figure(figsize=(16, 8))
        sentence = self.tokenizer_pt.encode(sentence)
        attention = tf.squeeze(attention[layer], axis=0)
        for head in range(attention.shape[0]):
            ax = fig.add_subplot(2, 4, head + 1)
            ax.matshow(attention[head][:-1, :], cmap='viridis')
            fontdict = {'fontsize': 10}
            ax.set_xticks(range(len(sentence) + 2))
            ax.set_yticks(range(len(result)))
            ax.set_ylim(len(result) - 1.5, -0.5)
            ax.set_xticklabels(['<start>'] + [self.tokenizer_pt.decode([i]) for i in sentence] + ['<end>'],
                               fontdict=fontdict,
                               rotation=90)
            ax.set_yticklabels([self.tokenizer_en.decode([i]) for i in result if i < self.tokenizer_en.vocab_size],
                               fontdict=fontdict)
            ax.set_xlabel('Head {}'.format(head + 1))
        plt.tight_layout()
        plt.show()

    def evaluate(self, inp_sentence):
        start_token = [self.tokenizer_pt.vocab_size]
        end_token = [self.tokenizer_pt.vocab_size + 1]
        inp_sentence = start_token + self.tokenizer_pt.encode(inp_sentence) + end_token
        encoder_input = tf.expand_dims(inp_sentence, 0)
        decoder_input = [self.tokenizer_en.vocab_size]
        output = tf.expand_dims(decoder_input, 0)
        for i in range(self.max_length):
            enc_padding_mask, combined_mask, dec_padding_mask = self.create_masks(encoder_input, output)
            predictions, attention_weights = self.transformer(encoder_input,
                                                              output,
                                                              False,
                                                              enc_padding_mask,
                                                              combined_mask,
                                                              dec_padding_mask)
            predictions = predictions[:, -1:, :]
            predicted_id = tf.cast(tf.argmax(predictions, axis=-1), tf.int32)
            if tf.equal(predicted_id, self.tokenizer_en.vocab_size + 1):
                return tf.squeeze(output, axis=0), attention_weights
            output = tf.concat([output, predicted_id], axis=-1)
        return tf.squeeze(output, axis=0), attention_weights

    def create_padding_mask(self, seq):
        seq = tf.cast(tf.math.equal(seq, 0), tf.float32)
        return seq[:, tf.newaxis, tf.newaxis, :]  # (batch_size, 1, 1, seq_len)

    def create_look_ahead_mask(self, size):
        mask = 1 - tf.linalg.band_part(tf.ones((size, size)), -1, 0)
        return mask  # (seqth_len, seq_len)

    def create_masks(self, inp, tar):
        enc_padding_mask = self.create_padding_mask(inp)
        dec_padding_mask = self.create_padding_mask(inp)
        look_ahead_mask = self.create_look_ahead_mask(tf.shape(tar)[1])
        dec_target_padding_mask = self.create_padding_mask(tar)
        combined_mask = tf.maximum(dec_target_padding_mask, look_ahead_mask)
        return enc_padding_mask, combined_mask, dec_padding_mask


if __name__ == '__main__':
    FLAGS = parser.parse_known_args()[0]
    data_helper = DataHelper()
    train_dataset = data_helper.get_datasets(max_length=FLAGS.max_length, batch_size=FLAGS.batch_size)
    nmt = NMT(data_helper.tokenizer_pt,
              data_helper.tokenizer_en,
              FLAGS.max_length,
              FLAGS.num_layers,
              FLAGS.d_model,
              FLAGS.dff,
              FLAGS.num_heads,
              FLAGS.dropout_rate)
    nmt.train(train_dataset)
    nmt.translate("este é um problema que temos que resolver.")
    print("Real translation: this is a problem we have to solve .")
