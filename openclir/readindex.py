import re
from whoosh.fields import *
import os
import io
import sys
from whoosh.index import open_dir
from whoosh.query import *
import gc
from whoosh.qparser import QueryParser, OrGroup
from whoosh import scoring
from whoosh.scoring import WeightingModel, BaseScorer, WeightLengthScorer
from gensim.models import Word2Vec


def load_dictionary(path):
    """
    Return a torch tensor of size (n, 2) where n is the size of the
    loader dictionary, and sort it by source word frequency.
    """
    assert os.path.isfile(path)

    word_dict = {}
    with io.open(path, 'r', encoding='utf-8') as f:
        for _, line in enumerate(f):
            # print(line)
            try:
                word1, word2 = line.rstrip().split()
                word_dict[word1] = word2
            except:
                pass
    return word_dict


def retrieve(path):
    ix = open_dir("index_file")
    reader = ix.reader()
    texts = reader.all_terms()
    for fieldname, text in texts:
        if fieldname == 'content':
            tinfo = reader.term_info(fieldname, text)
            # print(text)
            # print(tinfo.doc_frequency())

    query_dict = {}
    with open('/Users/zhoudong/Experiments/CLIR/openclir/bweswq.txt', 'r', encoding='utf-8') as rf:
        lines = rf.readlines()
        i = 0
        for line in lines:
            text1 = line.strip().split(' ')
            query_id = text1[0]  # 表示取第一个元素
            query_string = text1[1]  # 表示取第二个元素
            i = i + 1

            query_dict[query_id] = query_string
    print(i)

    with open(path, 'w', encoding='utf-8') as wf:
        with ix.searcher() as searcher:
            parser = QueryParser("content", schema=ix.schema, group=OrGroup)
            for query in query_dict.items():
                myquery = parser.parse(query[1])
                # print(myquery)
                # print(query[0])

                results = searcher.search(myquery, limit=None)
                query1 = []

                for result in results:
                    print(query[0] + ' ' + result['id'] + ' ' + str('%.5f' % result.score) + ' BM25\n')
                    wf.write(query[0] + ' ' + result['id'] + ' ' + str('%.5f' % result.score) + ' BM25\n')


def translate():
    path = 'F:/MUSE-master/data1/en-sw.0-5000.txt'#en-sw词典路径
    word_dict = load_dictionary(path)
    # print(word_dict)
    wf = open('H:/open_clir/sw3.txt', 'w', encoding='utf-8')#翻译en为sw的存储路径
    with open('H:/open_clir/en.txt', 'r', encoding='utf-8') as rf:#en查询的存储路径
        lines = rf.readlines()
        for line in lines:
            text1 = line.strip().split('\t')
            query_id = text1[0]
            query_string = text1[1]
            new_q = ''
            tokens = query_string.strip().split(' ')
            for token in tokens:
                if word_dict.get(token) is not None:
                    new_q = new_q + ' ' + word_dict.get(token)
                else:
                    new_q = new_q + ' ' + token
            print(query_string)
            print(new_q.strip())
            wf.write(query_id + '\t' + new_q.strip() + " \n")


# translate()
path = 'H:/open_clir/weBM25.txt'#利用BM25模型检索的结果
retrieve(path)
