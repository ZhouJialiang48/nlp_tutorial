#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/9/4 20:10
# @Author  : zhoujl
# @Site    : 
# @File    : data_producing.py
# @Software: PyCharm
import collections
import os
from tempfile import gettempdir
import zipfile
from six.moves import urllib
import tensorflow as tf

# STEP 1: Download the data.
url = 'http://mattmahoney.net/dc/'


def maybe_download(filename, expected_bytes):
    """
    若文件不存在则重新下载，并检查文件大小是否完整
    :param filename: 文件名
    :param expected_bytes: 完整文件大小
    :return: local_filename - 文件绝对路径
    """
    local_filename = os.path.join(gettempdir(), filename)
    if not os.path.exists(local_filename):
        local_filename, _ = urllib.request.urlretrieve(url + filename, local_filename)
    statinfo = os.stat(local_filename)
    if statinfo.st_size == expected_bytes:
        print('Found and verified', filename)
    else:
        print(statinfo.st_size)
        raise Exception('Failed to verify ' + local_filename
                        + '. Can you get to it with a browser?')
    return local_filename


def read_data(zip_filename):
    """
    提取压缩文件中第一个文本文件，构建单词列表
    :param filename: 压缩文件名
    :return: 单词列表
    """
    with zipfile.ZipFile(zip_filename) as f:
        text_file = f.namelist()[0]
        data = tf.compat.as_str(f.read(text_file)).split()
    return data


def build_dataset(words, n_words):
    """
    将原始输入文本转换为可用数据集
    :param words: 原始文本单词列表
    :param n_words: 返回字典的大小
    :return: data - 词频列表（按原文顺序排列）
             count - [单词, 词频]列表，从高到低排列
             dicitonary - {单词: 词频排名}
             reversed_dictionary - {词频排名: 单词}
    """
    # most_common()方法以[(elem, count),...]的形式返回前n个出现次数最多的元素列表
    # count为前(n_words - 1)个最常见词语列表
    count = [['UNK', -1]]
    count.extend(collections.Counter(words).most_common(n_words - 1))

    # 将常见词以及词频存入dictionary字典
    dictionary = dict()
    for word, _ in count:
        dictionary[word] = len(dictionary)

    # 统计UNK词频并改写count对应值
    data = list()
    unk_count = 0
    for word in words:
        index = dictionary.get(word, 0)
        if not index:
            unk_count += 1
        data.append(index)
    count[0][1] = unk_count

    reversed_dictionary = dict(zip(dictionary.values(), dictionary.keys()))
    return data, count, dictionary, reversed_dictionary
