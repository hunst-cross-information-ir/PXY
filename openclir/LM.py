import os
import sys
import os.path
import codecs
import numpy as np
import unicodedata
from collection_extractors import extract_dutch
import constants as c
from multiprocessing.pool import Pool
from functools import partial
from collections import Counter
from timer import Timer as PPrintTimer
from text2vec import clean
from text2vec import tokenize
from itertools import repeat

timer = PPrintTimer()


def _word_overlap(document, query):
    """
    Take items from document and keep if they are in the query, i.e. return intersection of query and document words
    and take the frequency from the document.
    :param document:
    :param query:
    :return:
    """
    document = dict(document)
    query = dict(query)
    return {k: v for k, v in document.items() if k in query.keys()}


def _count_words(document):
    tokens = document
    # print(Counter(tokens))
    # {k: v/n_d for k, v in Counter(tokens).items()}
    return dict(Counter(tokens))

#加载文档
def load_clef_documents(path):
    #tree = _decode_xml(path)
    documents = []
    ids = []
    names=os.listdir(path)
    for name in names:
    	file_dir=path+name
    	with open(file_dir,'r',encoding='utf-8')as f:
    		text=f.read()
    	index=name.rfind('.')
    	file_id=name[:index]
    	text=text.replace('\n','')
    	text=text.lower()
    	ids.append(file_id)
    	documents.append(text)

    return ids, documents

#加载查询
def load_queries(path):
    queries = []
    ids = []
    with open(path,'r',encoding='utf-8')as rf:
    	lines=rf.readlines()
    	for line in lines:
    		text=line.strip().split('	')
    		query_id=text[0]
    		query_string=text[1].lower()
    		queries.append(query_string)
    		ids.append(query_id)
    	return ids, queries


def _save_ranking(all_rankings, path):
    """
    Stores ranking
    """
    # wf.write(query[0] + ' Q0 ' + result['id'] + ' ' + str(i) + ' ' + str(results.score(i - 1)) + ' LM\n')
    file_content = []
    for query, ranking_with_doc_ids in all_rankings:
        queries=[]
        for ranking, score in ranking_with_doc_ids:
            queries.append(score)
        min=queries[0]
        max=queries[0]
        for q in queries[1:]:
            if(max<q):
                max=q
            if(min>q):
                min=q
        for ranking, score in ranking_with_doc_ids:
            if min!=0 and max!=0:
                score=(score-min)/(max-min)#归一化 处理
                #score=score/max
                one_line = str(query) + ' ' + str(ranking) + ' ' + str('%.5f' % score) + ' LM\n'
            else:
                one_line = str(query) + ' ' + str(ranking) + ' ' + str('%.5f' % score) + ' LM\n'
           
            file_content.append(one_line)
            
    file_content = ''.join(file_content)
    with open(path, mode="w") as ranking_file:
        ranking_file.write(file_content)
    pass


def _score_doc_unigram_lm(data, mu=1000):
    document_distribution, query_distr, collection_dist, dc = data
    n_d = sum(query_distr.values())  # document length

    smoothing_term = n_d / (n_d + mu)#平滑项
    document_score = 0

    for query_term, occurrences in query_distr.items():
        if query_term in collection_dist:
            query_freq_in_doc = document_distribution.get(query_term, 0)
            P_q_d = query_freq_in_doc / n_d

            query_freq_in_collection = collection_dist.get(query_term, 0)
            assert query_freq_in_collection != 0
            P_q_dc = query_freq_in_collection / dc

            score = smoothing_term * P_q_d + (1 - smoothing_term) * P_q_dc
            document_score += (np.log(score) * occurrences)

    # calculations up to here were done in log-space
    document_score = np.exp(document_score) if document_score != 0 else 0
    return document_score


def prepare_experiment(doc_dir, query_file):

    documents = []
    doc_ids = []
    query_ids, queries = load_queries(query_file)
    doc_ids,documents=load_clef_documents(doc_dir)
    
    return doc_ids, documents, query_ids, queries


def evaluate_clef(query_ids, doc_ids, all_rankings, scores_for_all_query):
    """
    Evaluates results for queries in terms of Mean Average Precision (MAP). Evaluation gold standard is
    loaded from the relevance assessments.
    :param query_ids: internal id of query
    :param doc_ids: internal id of document
    :param relass: gold standard (expected) rankings
    :param all_rankings: (actual) rankings retrieved
    :return:
    """

    rankings_with_doc_ids = []
    for j in range(len(query_ids)):
        # for the ith query
        query_id = query_ids[j]
        
        scores = scores_for_all_query[j].tolist()
        ranking = all_rankings[j].tolist()
        ranking_with_docm_ids=[]
        for i in ranking:
            ranking_with_docm_ids.append((doc_ids[i], scores[i]))
        rankings_with_doc_ids.append((query_id, ranking_with_docm_ids))
    return rankings_with_doc_ids


def run_unigram_lm(query_lang, doc_lang,experiment_data, processes=40, most_common=None):

    doc_ids, documents, query_ids, queries=experiment_data
    pool = Pool(processes=processes)

    print("Start preprocessing data %s" % timer.pprint_lap())
    clean_to_lower = partial(clean, to_lower=True)
    tokenize_doc_language = partial(tokenize, language=doc_lang, exclude_digits=True)
    documents = pool.map(clean_to_lower, documents)
    documents = pool.map(tokenize_doc_language, documents)
    print("Documents preprocessed %s" % (timer.pprint_lap()))

    tokenize_query_language = partial(tokenize, language=query_lang, exclude_digits=True)
    queries = pool.map(clean_to_lower, queries)
    queries = pool.map(tokenize_query_language, queries)
    print("queries preprocessed %s" % timer.pprint_lap())

    # word frequency distribution per document
    document_distributions = pool.map(_count_words, documents)
    print("Document conditional counts collected %s" % timer.pprint_lap())

    # word frequency distribution per query
    query_distributions = pool.map(_count_words, queries)
    print("Query conditional counts collected %s" % timer.pprint_lap())
    # print(query_distributions)

    collection_size = sum([sum(document.values()) for document in document_distributions])
    collection_distribution = Counter()
    for document in document_distributions:
        collection_distribution.update(document)  # { token: frequency }
    if most_common is not None:
        collection_distribution.most_common(most_common)
    collection_distribution = dict(collection_distribution)
    print("Marginal counts collected %s" % timer.pprint_lap())
    # print(collection_size)

    np.random.seed(10)
    random_ranking = np.random.permutation(len(documents))
    doc_count = len(document_distributions)
    broadcasted_collection_size = [collection_size] * doc_count
    # print(doc_count)

    results = []
    scores_for_all_query = []
    print("start evaluation %s" % timer.pprint_lap())
    for i, query in enumerate(query_distributions, 1):
        query_id = query_ids[i - 1]
        suffix = ""
        if query_id in query_ids:
            doc_subset_distributions = pool.starmap(_word_overlap, zip(document_distributions, repeat(query)))
            # print(doc_subset_distributions)
            col_subset_distribution = _word_overlap(collection_distribution, query)

            scores_for_query = pool.map(_score_doc_unigram_lm, zip(doc_subset_distributions,  # {word_d: freq}
                                                                   repeat(query),  # {word_q: freq}
                                                                   repeat(col_subset_distribution),
                                                                   broadcasted_collection_size))  # {word_dc: freq}
            # condition for random ranking if all documents score zero
            any_score_non_zero = sum(scores_for_query) > 0
            # https://stackoverflow.com/questions/20197990/how-to-make-argsort-result-to-be-random-between-equal-values
            # ranking_for_query = np.lexsort((-np.array(scores_for_query),random_ranking))
            # sort first by argsort then by random ranking

            ranking_for_query = np.argsort(-np.array(scores_for_query)) if any_score_non_zero else random_ranking
            results.append(ranking_for_query)
            scores_for_all_query.append(scores_for_query)
        else:
            results.append(random_ranking)  # query without relevant documents is not fired
            suffix = " --> no relevant docs for q_id %s" % str(query_id)
        if i % 10 == 0:
            print("%s  queries processed (%s) %s" % (i, timer.pprint_lap(), suffix))

    pool.close()
    pool.join()
    all_rankings = evaluate_clef(query_ids=query_ids, doc_ids=doc_ids,
                                 all_rankings=np.array(results),
                                 scores_for_all_query=np.array(scores_for_all_query))

    return all_rankings


def main():
    #path_documents = 'H:/OPEN_CLIR/data/OPEN-CLEF/ANALYSISN/OPENCLIR_2019-1A/ANALYSIS/text/src/'#analysis

    path_documents='G:/Openclir-Data/OPENCLIR_2019-1A/EVAL/text/src/'#Eval

    #path_queries = 'H:/open_clir/sw3.txt'#analysis
    path_queries='G:/Openclir-Data/new_query.txt'#Eval
    

    #result_path = 'H:/OPEN_CLIR/result/result_2500.txt'#analysis
    result_path='G:/Openclir-Data/result/result_1000.txt'#Eval

    experiment_data = prepare_experiment(doc_dir=path_documents, query_file=path_queries)

    query_lang = 'english'
    doc_lang = 'english'

    all_rankings = run_unigram_lm(query_lang, doc_lang, experiment_data, processes=40, most_common=None)

    _save_ranking(all_rankings, result_path)


if __name__ == "__main__":
    main()

