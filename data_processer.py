import os
import re
from collections import Counter
from collections import defaultdict
import numpy as np

from tqdm import tqdm


class StringProcess(object):

    def __init__(self):
        self.other_char = re.compile(r"[^A-Za-z0-9(),!?\'\`]", flags=0)
        self.num = re.compile(r"[+-]?\d+\.?\d*", flags=0)
        # self.url = re.compile(r"[a-z]*[:.]+\S+|\n|\s+", flags=0)
        self.url = re.compile(
            r"(https?|ftp|file)://[-A-Za-z0-9+&@#/%?=~_|!:,.;]+[-A-Za-z0-9+&@#/%=~_|]",
            flags=0)
        self.stop_words = None
        self.nlp = None

    def clean_str(self, string):
        string = re.sub(self.other_char, " ", string)
        string = re.sub(r"\'s", " \'s", string)
        string = re.sub(r"\'ve", " \'ve", string)
        string = re.sub(r"n\'t", " n\'t", string)
        string = re.sub(r"\'re", " \'re", string)
        string = re.sub(r"\'d", " \'d", string)
        string = re.sub(r"\'ll", " \'ll", string)
        string = re.sub(r",", " , ", string)
        string = re.sub(r"!", " ! ", string)
        string = re.sub(r"\(", " ( ", string)
        string = re.sub(r"\)", " ) ", string)
        string = re.sub(r"\?", " ? ", string)
        string = re.sub(r"\s{2,}", " ", string)

        return string.strip().lower()

    def norm_str(self, string):
        string = re.sub(self.other_char, " ", string)

        if self.nlp is None:
            from spacy.lang.en import English
            self.nlp = English()

        new_doc = list()
        doc = self.nlp(string)
        for token in doc:
            if token.is_space or token.is_punct:
                continue
            if token.is_digit:
                token = "[num]"
            else:
                token = token.text

            new_doc.append(token)

        return " ".join(new_doc).lower()

    def lean_str_sst(self, string):
        """
            Tokenization/string cleaning for the SST yelp_dataset
            Original taken from https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
            """
        string = re.sub(self.other_char, " ", string)
        string = re.sub(r"\s{2,}", " ", string)
        return string.strip().lower()

    def remove_stopword(self, string):
        if self.stop_words is None:
            from nltk.corpus import stopwords
            self.stop_words = set(stopwords.words('english'))

        if type(string) is str:
            string = string.split()

        new_string = list()
        for word in string:
            if word in self.stop_words:
                continue
            new_string.append(word)

        return " ".join(new_string)

    def replace_num(self, string):
        result = re.sub(self.num, '<num>', string)
        return result

    def replace_urls(self, string):
        result = re.sub(self.url, '<url>', string)
        result = ' '.join(re.split(' +|\n+', result)).strip()
        return result


def remove_less_word(lines_str, word_st):
    return " ".join([word for word in lines_str.split() if word in word_st])


class CorpusProcess:

    def __init__(self, dataset, encoding=None):
        corpus_path = "data/text_dataset/corpus"
        clean_corpus_path = "data/text_dataset/clean_corpus"
        if not os.path.exists(clean_corpus_path):
            os.makedirs(clean_corpus_path)

        self.dataset = dataset
        self.corpus_name = f"{corpus_path}/{dataset}.txt"
        self.save_name = f"{clean_corpus_path}/{dataset}.txt"
        self.context_dct = defaultdict(dict)

        self.encoding = encoding
        self.clean_text()

    def clean_text(self):
        # 处理字符串
        sp = StringProcess()
        # 存储单词列表
        word_lst = list()
        # 读取mr.txt文本
        with open(self.corpus_name, mode="rb", encoding=self.encoding) as fin:
            # tqdm装饰一个iterable object,只是会有一个进度条
            for indx, item in tqdm(enumerate(fin), desc="clean the text"):
                data = item.strip().decode('latin1')
                data = sp.clean_str(data)
                if self.dataset not in {"mr"}:
                    data = sp.remove_stopword(data)
                word_lst.extend(data.split())

        # 如果不是mr数据集,就去掉词频低于5的单词
        word_st = set()
        if self.dataset not in {"mr"}:
            for word, value in Counter(word_lst).items():
                if value < 5:
                    continue
                word_st.add(word)
        # 如果是mr数据集,直接变成集合。所有单词的词频都变成1
        else:
            word_st = set(word_lst)

        doc_len_lst = list()
        # 创建保存文件的stream
        with open(self.save_name, mode='w') as fout:
            with open(self.corpus_name, mode="rb",
                      encoding=self.encoding) as fin:
                # 逐行读取
                for line in tqdm(fin):
                    # 对每一行进行处理
                    lines_str = line.strip().decode('latin1')
                    lines_str = sp.clean_str(lines_str)
                    if self.dataset not in {"mr"}:
                        lines_str = sp.remove_stopword(lines_str)
                        lines_str = remove_less_word(lines_str, word_st)

                    fout.write(lines_str)
                    fout.write(" \n")
                    # 把每一行的长度记录
                    doc_len_lst.append(len(lines_str.split()))

        print("Average length:", np.mean(doc_len_lst))
        print("doc count:", len(doc_len_lst))
        print("Total number of words:", len(word_st))


def main():
    # CorpusProcess("R52")
    # CorpusProcess("20ng")
    # CorpusProcess("mr")
    CorpusProcess("ohsumed")
    # CorpusProcess("R8")
    # pass


if __name__ == '__main__':
    main()
