from whoosh.scoring import WeightingModel,BaseScorer,WeightLengthScorer
from whoosh.reading import IndexReader
import whoosh
import numpy as np
import gensim
from gensim.models import Word2Vec
import sys


#加载已经训练好的词向量模型
sb=np.load('F:/test/clir_experiment/doc_vec/nl_dict_doc300.npy').item()
model=Word2Vec.load('F:/CLEF2003/we-vs_new/size600/we_vs1.model')
size=600
#自定义LDA部分
#计算两个向量之间的相似度
# def wevs(a,b):
#     num=np.matmul(a,b.T)
#     num=float(num)
#     denom_a=np.linalg.norm(a)
#     denom_b=np.linalg.norm(b)
#     sim=num/(denom_a+denom_b)
#     sim=0.5+0.5*sim
#     return sim
#计算两个向量之间的相似度
def wevs(a,b):
    vector_a=np.mat(a)
    vector_b=np.mat(b)
    num=float(vector_a*vector_b.T)
    denom=np.linalg.norm(vector_a)*np.linalg.norm(vector_b)
    cos=num/denom
    sim=0.5+0.5*cos
    return sim

class WEvs(WeightingModel):
    # def __init__(self,u):
    #     self.u=u
    #     print(query)
    def scorer(self, searcher, fieldname, text,qf=1):
        if not searcher.schema[fieldname].scorable:
            return WeightScorer.for_(searcher, fieldname, text)
        print(text)

        return WEvsScorer(searcher, fieldname, text, qf=qf)


class WEvsScorer(BaseScorer):
    # self.matcher=matcher
    def __init__(self, searcher, fieldname, text,qf=1):
        parent = searcher.get_parent()  # Returns self if no parent
        self.se=parent
        self.text=text
        print(text.decode('utf-8'))
        if text.decode('utf-8') in model.wv.vocab:
            self.query=model[text.decode('utf-8')]
            # print(self.query)
        else:
            self.query=np.zeros(size)
            # print(self.query)

    def supports_block_quality(self):
        return True

    def score(self,matcher):
        print(matcher.id())
        #vec=np.zeros(size)
        doc=self.se.reader().stored_fields(matcher.id())
        id=doc['id']
        if id in sb.keys():
            vec=sb[id]
        else:
            vec=np.zeros(size)
        #text=doc['content']
        # print(text)
        # sys.exit(0)
        #for word in text.split():
        #    if word in model.wv.vocab:
         #       vec=vec+model[word]
        # print(vec)
        # print("******************************")
        # print(self.query)
        return wevs(vec,self.query)


    def max_quality(self):
        return self._maxquality

    def block_quality(self, matcher):
        return matcher.block_max_weight() * self.idf

# class WEReader(IndexReader):

#     def vector(self,docnum,fieldname,format_=None):
#         searcher.


#自定义LM部分
def lm(tf,dl,cf,fl,u):
    #tf代表词在该文档中出现次数
    #dl代表文档长度
    #cf代表词在文档集合中出现次数
    #fl代表所有文档的长度之和
    #u代表参数
    return tf/(dl+u)+(1-dl/(dl+u))*(cf/fl)

class LM(WeightingModel):
    # def __init__(self,u):
    #     self.u=1000
    def scorer(self, searcher, fieldname, text, u=200,qf=1):
        if not searcher.schema[fieldname].scorable:
            return WeightScorer.for_(searcher, fieldname, text)
        # print(fieldname)
        # print(text)
    

class LMScorer(WeightLengthScorer):
    def __init__(self, searcher, fieldname, text, u,qf=1):
        # for item in searcher.document(fieldname,text):
        #     print(item)
        parent = searcher.get_parent()  # Returns self if no parent
        # print(fieldname)
        # print(text)
        reader=parent.reader()
        term_info=reader.term_info(fieldname,text)        
        self.cf = term_info.weight()
        self.fl = parent.field_length(fieldname)
        # self._maxquality = maxweight * idf
        self.u=u
        # print(fieldname)

        self.setup(searcher, fieldname, text)

    def supports_block_quality(self):
        return True

    def _score(self, weight,length):
        # print(weight)
        # print(length)
        return lm(weight,length,self.cf,self.fl,self.u)

    def max_quality(self):
        return self._maxquality

    def block_quality(self, matcher):
        return matcher.block_max_weight() * self.lm


# #加载训练好的LDA模型和训练集词汇表
# lda=LdaModel.load('F:/CLEF2003/lda/lda_k1000.model')
# dct=Dictionary.load('F:/CLEF2003/lda/deep.dict')
# #获取LDA训练模型的主题词分布
# pp=lda.get_topics()


# # #加载查询词的词袋表示
# # query_items=np.load('E:/corpus/daima/npy/query_items.npy').item()
# #加载测试文档的主题分布
#dict_file=np.load('E:/corpus/daima/dict_file.npy').item()

#自定义LDA部分
def lda(tf,dl,cf,fl,u):
    #tf代表词在该文档中出现次数
    #dl代表文档长度
    #cf代表词在文档集合中出现次数
    #fl代表所有文档的长度之和
    #u代表参数
    return tf/(dl+u)+(1-dl/(dl+u))*(cf/fl)

class LDA(WeightingModel):
    # def __init__(self,u):
    #     self.u=1000
    def scorer(self, searcher, fieldname, text,qf=1):
        if not searcher.schema[fieldname].scorable:
            return WeightScorer.for_(searcher, fieldname, text)
        # print(fieldname)
        # print(text)
        return LDAScorer(searcher, fieldname, text, qf=qf)
    # def scorer(self, searcher, fieldname, text, u=1000):
    #     # IDF is a global statistic, so get it from the top-level searcher
    #     parent = searcher.get_parent()  # Returns self if no parent
    #     self.cf = parent.weight(fieldname, text)
    #     self.fl = parent.field_length(fieldname)

    #     maxweight = searcher.term_info(fieldname, text).max_weight()
    #     return LMScorer(maxweight, idf)


class LDAScorer(BaseScorer):
    # self.matcher=matcher
    def __init__(self, searcher, fieldname, text,qf=1):
        parent = searcher.get_parent()  # Returns self if no parent
        self.se=parent
        self.text=text
        print(text)

    def supports_block_quality(self):
        return True

    def score(self,matcher):
        print(matcher.id())
        doc=self.se.reader().stored_fields(matcher.id())
        docno=doc['id']
        topic_distribute=dict_file[docno]
        p=0
        # print(topic_distribute)
        for item in topic_distribute:
            p1=item[1]
            if self.text.decode('utf-8') in dct.token2id.keys():
                p2=pp[item[0]][dct.token2id[self.text.decode('utf-8')]]
            else:
                break
            p=p+p1*p2
        return p
        # print(matcher.value().decode('utf-8'))
        # sys.exit(0)

    # def _score(self, weight,length):
    #     # print(weight)
    #     # print(length)
    #     return lda(weight,length,self.cf,self.fl,self.u)

    def max_quality(self):
        return self._maxquality

    def block_quality(self, matcher):
        return matcher.block_max_weight() * self.idf




