import io
import numpy as np
def read_txt_embeddings(emb_path):
    """
    Reload pretrained embeddings from a text file.
    """
    # word2id = {}
    vectors = {}

    # load pretrained embeddings
    lang = 'en'
    _emb_dim_file = 300
    with io.open(emb_path, 'r', encoding='utf-8', newline='\n', errors='ignore') as f:
        for i, line in enumerate(f):
            if i == 0:
                split = line.split()
            else:
                word, vect = line.rstrip().split(' ', 1)
                word = word.lower()
                vect = np.fromstring(vect, sep=' ')
                if np.linalg.norm(vect) == 0:  # avoid to have null embeddings
                    vect[0] = 0.01
                else:
                    if not vect.shape == (_emb_dim_file,):
                        # logger.warning("Invalid dimension (%i) for %s word '%s' in line %i."
                        #                % (vect.shape[0], 'source' if source else 'target', word, i))
                        continue
                    assert vect.shape == (_emb_dim_file,), i
                    vectors[word]=vect

    # compute new vocabulary / embeddings
    # embeddings = np.concatenate(vectors, 0)
    # embeddings = torch.from_numpy(embeddings).float()
    # embeddings = embeddings.cuda() if (params.cuda and not full_vocab) else embeddings
    return vectors

def cos_sim(vector_a, vector_b):
    """
    计算两个向量之间的余弦相似度
    :param vector_a: 向量 a 
    :param vector_b: 向量 b
    :return: sim
    """
    vector_a = np.mat(vector_a)
    vector_b = np.mat(vector_b)
    num = float(vector_a * vector_b.T)
    denom = np.linalg.norm(vector_a) * np.linalg.norm(vector_b)
    cos = num / denom
    sim = 0.5 + 0.5 * cos
    return sim

def get_SWword(word_vectors,embeddings):
    words_similarity={}
    for word,vectors in embeddings.items():
        simi=cos_sim(word_vectors,vectors)
        words_similarity[word]=simi
    #对每个词按照相似度降序排序
    new=sorted(words_similarity.items(),key = lambda x:x[1],reverse = True)
    return new[0][0]


en_vectors=read_txt_embeddings('F:/MUSE-master/dumped/debug/gf5i4wmxj4/vectors-en.txt')
sw_vectors=read_txt_embeddings('F:/MUSE-master/dumped/debug/gf5i4wmxj4/vectors-sw.txt')

#读取英文查询文件
ids=[]
queries=[]
with open('G:/Openclir-Data/query.txt','r',encoding='utf-8') as rf:
    lines=rf.readlines()
    for line in lines:
        #id,words=line.lsplit()
        text=line.strip().split()
        id=text[0]
        words=text[1]
        queries.append(words)
        ids.append(id)

#遍历所有的query,将每个query中的词翻译成斯瓦西里语，主要是利用NN，计算词向量相似度
sw_queries=[]
for query in queries:
    word_list=query.rstrip().split()
    sw_query=''
    for word in word_list:
        if word in en_vectors.keys():
            word_vectors=en_vectors[word]
            sw_word=get_SWword(word_vectors,sw_vectors)
            sw_query=sw_query+sw_word+' '
    if sw_query=='':
        print('没有找到翻译词')
        sw_query=query
    sw_query=sw_query.strip()
    sw_queries.append(sw_query)

#enumerate函数用于遍历序列中的元素以及其下标
with open('G:/Openclir-Data/new_query.txt','w',encoding='utf-8') as wf:
    for i,query in enumerate(sw_queries):
        wf.write(ids[i]+' '+query+'\n')



