#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/9/4 15:14
# @Author  : zhoujl
# @Site    : 
# @File    : word2vec_basic.py
# @Software: PyCharm
import os
import sys
from tempfile import gettempdir
import argparse
import random
import collections
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import tensorflow as tf
from tensorflow.contrib.tensorboard.plugins import projector
from data_producing import maybe_download, read_data, build_dataset

# 添加'--log_dir'参数，指定TensorBoard的summaries储存地址
# 默认为当前目录下的'log/'目录，若不存在则创建指定目录
current_path = os.path.dirname(os.path.realpath(sys.argv[0]))
parser = argparse.ArgumentParser()
parser.add_argument(
    '--log_dir',
    type=str,
    default=os.path.join(current_path, 'log'),
    help='The log directory for TensorBoard summaries.')
FLAGS, unparsed = parser.parse_known_args()
if not os.path.exists(FLAGS.log_dir):
    os.makedirs(FLAGS.log_dir)


# Step-1: 下载/提取数据
filename = maybe_download('text8.zip', 31344016)
vocabulary = read_data(filename)
vocabulary_size = 50000


# Step-2: 构建字典，稀有词用UNK替换
data, count, dictionary, reversed_dictionary = build_dataset(vocabulary, vocabulary_size)
print('Filename: {}'.format(filename))
print('Data size: {}'.format(len(vocabulary)))
print('Most common words (+UNK): {}'.format(count[:5]))
print('Sample data:\n{}\n{}'.format(data[:15], [reversed_dictionary[i] for i in data[:15]]))
del vocabulary  # 删除原始文本单词列表，释放内存


# Step-3: 生成训练batch和labels
data_index = 0
def generate_batch(batch_size, num_skips, skip_window):
    global data_index
    assert batch_size % num_skips == 0
    assert num_skips <= 2 * skip_window
    batch = np.ndarray(shape=(batch_size), dtype=np.int32)
    labels = np.ndarray(shape=(batch_size, 1), dtype=np.int32)
    span = 2 * skip_window + 1
    # 若超出数据长度，则索引清零
    if data_index + span > len(data):
        data_index = 0
    # 最多存放span个元素的双向队列，作为缓存
    buffer = collections.deque(maxlen=span)
    buffer.extend(data[data_index:data_index + span])
    data_index += span
    for i in range(batch_size // num_skips):
        # 上下文词在此buffer中的位置列表
        context_words = [w for w in range(span) if w != skip_window]
        words_to_use = random.sample(context_words, num_skips)
        # 构建batch(包含目标词)和对应labels(包含上下文词)
        for j, context_word in enumerate(words_to_use):
            batch[i * num_skips + j] = buffer[skip_window]
            labels[i * num_skips + j, 0] = buffer[context_word]
        # 若索引指向尾部，则从头更新缓存，更新索引
        if data_index == len(data):
            buffer.extend(data[0:span])
            data_index = span
        # 否则，更新一位缓存，索引加一
        else:
            buffer.append(data[data_index])
            data_index += 1
    # 保存全局变量data_index的值
    data_index = (data_index + len(data) - span) % len(data)
    return batch, labels

batch, labels = generate_batch(batch_size=8, num_skips=2, skip_window=1)
for i in range(8):
    print('{}: {} -> {}: {}'.format(batch[i], reversed_dictionary[batch[i]],
                                    labels[i, 0], reversed_dictionary[labels[i, 0]]))


# Step-4: 构建skip-gram模型
batch_size = 128
embedding_size = 128
skip_window = 1
num_skips = 2
num_sampled = 64
valid_size = 16
valid_window = 100
valid_examples = np.random.choice(valid_window, valid_size, replace=False)

graph = tf.Graph()
with graph.as_default():
    # 输入占位
    with tf.name_scope('inputs'):
        train_inputs = tf.placeholder(tf.int32, shape=[batch_size])
        train_labels = tf.placeholder(tf.int32, shape=[batch_size, 1])
        valid_dataset = tf.constant(valid_examples, dtype=tf.int32)
    # 权重与偏置
    with tf.device('/cpu:0'):
        with tf.name_scope('embeddings'):
            embeddings = tf.Variable(tf.random_uniform([vocabulary_size, embedding_size], -1.0, 1.0))
            embed = tf.nn.embedding_lookup(embeddings, train_inputs)
        with tf.name_scope('weights'):
            nce_weights = tf.Variable(tf.truncated_normal([vocabulary_size, embedding_size],
                                                          stddev=1.0 / np.sqrt(embedding_size)))
        with tf.name_scope('biases'):
            nce_biases = tf.Variable(tf.zeros([vocabulary_size]))
    # 损失函数nce
    with tf.name_scope('loss'):
        loss = tf.reduce_mean(
            tf.nn.nce_loss(
                weights=nce_weights,
                biases=nce_biases,
                labels=train_labels,
                inputs=embed,
                num_sampled=num_sampled,
                num_classes=vocabulary_size))
    tf.summary.scalar('loss', loss)
    # 优化器
    with tf.name_scope('optimizer'):
        optimizer = tf.train.GradientDescentOptimizer(1.0).minimize(loss)
    # 余弦相似度
    norm = tf.sqrt(tf.reduce_sum(tf.square(embeddings), 1, keepdims=True))
    normalized_embeddings = embeddings / norm
    valid_embeddings = tf.nn.embedding_lookup(normalized_embeddings, valid_dataset)
    similarity = tf.matmul(valid_embeddings, normalized_embeddings, transpose_b=True)
    # 合并summary，初始化，保存图
    merged = tf.summary.merge_all()
    init = tf.global_variables_initializer()
    saver = tf.train.Saver()


# Step-5: 训练skip-gram模型
num_steps = 100001
with tf.Session(graph=graph) as sess:
    writer = tf.summary.FileWriter(FLAGS.log_dir, sess.graph)
    init.run()
    print('Initialized!')

    average_loss = 0
    for step in range(num_steps):
        batch_inputs, batch_labels = generate_batch(batch_size, num_skips, skip_window)
        feed_dict = {train_inputs: batch_inputs, train_labels: batch_labels}
        run_metadata = tf.RunMetadata()
        _, summary, loss_val = sess.run([optimizer, merged, loss],
                                        feed_dict=feed_dict,
                                        run_metadata=run_metadata)
        average_loss += loss_val

        # 写入summary
        writer.add_summary(summary, step)
        if step == (num_steps - 1):
            writer.add_run_metadata(run_metadata, 'step{}'.format(step))
        if step % 2000 == 0:
            if step > 0:
                average_loss /= 2000
            print('Average loss at step {}: {}'.format(step, average_loss))
            average_loss = 0

        # 打印相似单词
        if step % 10000 == 0:
            sim = similarity.eval()
            for i in range(valid_size):
                valid_word = reversed_dictionary[valid_examples[i]]
                top_k = 8
                nearst = (-sim[i, :]).argsort()[1:top_k + 1]
                log_str = 'Nearest to {}:'.format(valid_word)
                for k in range(top_k):
                    closed_word = reversed_dictionary[nearst[k]]
                    log_str = '{} {},'.format(log_str, closed_word)
                print(log_str)
    # 最终的词向量
    final_embeddings = normalized_embeddings.eval()

    # 写入embeddings相应labels
    with open(FLAGS.log_dir + '/metadata.tsv', 'w') as f:
        for i in range(vocabulary_size):
            f.write(reversed_dictionary[i] + '\n')

    saver.save(sess, os.path.join(FLAGS.log_dir, 'model.ckpt'))

    # 词向量可视化
    config = projector.ProjectorConfig()
    embedding_config = config.embeddings.add()
    embedding_config.tensor_name = embeddings.name
    embedding_config.metadata_path = os.path.join(FLAGS.log_dir, 'metadata.tsv')
    projector.visualize_embeddings(writer, config)

writer.close()


# Step-6: 可视化词向量之间的距离(二维坐标)
def plot_with_labels(low_dim_embs, labels, filename):
    assert low_dim_embs.shape[0] >= len(labels), 'More labels than embeddings'
    plt.figure(figsize=(18, 18))
    for i, label in enumerate(labels):
        x, y = low_dim_embs[i, :]
        plt.scatter(x, y)
        plt.annotate(label, xy=(x, y), xytext=(5, 2), textcoords='offset points', ha='right', va='bottom')
    plt.savefig(filename)

tsne = TSNE(perplexity=30, n_components=2, init='pca', n_iter=5000, method='exact')
plot_only = 500
low_dim_embs = tsne.fit_transform(final_embeddings[:plot_only, :])
labels = [reversed_dictionary[i] for i in range(plot_only)]
plot_with_labels(low_dim_embs, labels, os.path.join(gettempdir(), 'tsne.png'))
