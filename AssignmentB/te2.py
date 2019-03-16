import nltk
from nltk.classify.util import accuracy
from nltk.classify.maxent import MaxentClassifier
from nltk.classify.scikitlearn import SklearnClassifier
from sklearn.svm import LinearSVC
from nltk.classify.decisiontree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import rte as rte_corpus
import numpy as np
import Assignment1.DataPreprocessing as mpipeline
# from gensim.models import KeyedVectors
# from gensim.models import Word2Vec

def rte_featurize(rte_pairs, training, test_id=0):
    id = 0
    rl = []
    for pair in rte_pairs:
        rl.append((rte_features(pair, id, training, test_id), pair.value))
        id += 1
    return rl


def rte_features(rtepair, id, training=True, test_id=0):
    true_test_id = id + test_id*800
    extractor = nltk.RTEFeatureExtractor(rtepair)
    features = {}
    features['alwayson'] = True
    features['word_overlap'] = len(extractor.overlap('word'))
    features['word_hyp_extra'] = len(extractor.hyp_extra('word'))
    features['ne_overlap'] = len(extractor.overlap('ne'))
    features['ne_hyp_extra'] = len(extractor.hyp_extra('ne'))
    features['neg_txt'] = len(extractor.negwords & extractor.text_words)
    features['neg_hyp'] = len(extractor.negwords & extractor.hyp_words)
    if training:
        features['tfidf_sim'] = tfidf_sim_list_train[id]
        features['w2v_ne_sim'] = ne_w2v_train[id]
        features['w2v_noun_sim'] = noun_w2v_train[id]
        features['w2v_adj_sim'] = adj_w2v_train[id]
        features['w2v_verb_sim'] = verb_w2v_train[id]
    else:
        features['tfidf_sim'] = tfidf_sim_list_test[true_test_id]
        features['w2v_ne_sim'] = ne_w2v_test[true_test_id]
        features['w2v_noun_sim'] = noun_w2v_test[true_test_id]
        features['w2v_adj_sim'] = adj_w2v_test[true_test_id]
        features['w2v_verb_sim'] = verb_w2v_test[true_test_id]
    return features


def tf_idf(which):
    cv = TfidfVectorizer(binary=False, decode_error='ignore', stop_words='english')
    if which == 'train':
        vec = cv.fit_transform(pairs_to_list(rte_corpus.pairs(['rte1_dev.xml', 'rte2_dev.xml', 'rte3_dev.xml'])))
        # print vec.toarray()
        return vec
    else:
        vec = cv.fit_transform(pairs_to_list(rte_corpus.pairs(['rte1_test.xml', 'rte2_test.xml', 'rte3_test.xml'])))
        return vec


def make_tfidf_sim(train_or_test):
    tfidf = tf_idf(train_or_test)
    simMatrix = (tfidf * tfidf.T).A
    sim_list = []
    for i in range(0, tfidf.shape[0], 2):
        sim_list.append(simMatrix[i, i+1])
    return sim_list


def ne_word2vec_sim(pretrain_model, train=True):
    lst = ['rte1_test.xml', 'rte2_test.xml', 'rte3_test.xml']
    if train:
        lst = ['rte1_dev.xml', 'rte2_dev.xml', 'rte3_dev.xml']
    sents = pairs_to_list(rte_corpus.pairs(lst))
    nes = mpipeline.detect_v(sents)
    w2v_sim = []
    for i in range(0, len(nes), 2):
        if nes[i] == [] or nes[i+1] == []:
            w2v_sim.append(0.0)
        else:
            total_text = []
            total_hyp = []
            mdict = {}
            for ne in nes[i]:
                # print ne.encode('gbk', "ignore")
                tks = nltk.word_tokenize(ne)
                for word in tks:
                    try:
                        mdict[word] = pretrain_model.wv[word]
                    except KeyError:
                        pass
                        # print "sth not in vocabulary"
                ne_vec = sum([mdict[word] for word in tks if word in mdict.keys()])
                total_text.append(ne_vec)  # get a entity vector, append in
            # print total_text
            for ne in nes[i+1]:  # entities in hyp
                # print ne.encode('gbk', "ignore")
                # print nes[i+1]
                # print "======================="
                tks = nltk.word_tokenize(ne)
                for word in tks:
                    try:
                        mdict[word] = pretrain_model.wv[word]
                    except KeyError:
                        pass
                        # print "sth not in vocabulary"
                ne_vec = sum([mdict[word] for word in tks if word in mdict.keys()])
                # print "append in ", ne_vec
                total_hyp.append(ne_vec)  # get a entity vector, append in
                # print 'TOTAL ne_vec: ', total_hyp
            # print "total a hyp of a pair", total_hyp
            # print "total a text of a pair", total_text
            # print "SUM: ", sum(total_hyp)
            sim = vec_cosine_sim(sum(total_text), sum(total_hyp))  # a pair
            # print sim
            w2v_sim.append(sim)
    fo = file('./w2v_verb_sim_train.txt', 'w')
    for i in w2v_sim:
        if type(i) == type(np.zeros(3)):
            fo.write("0.0\n")
        else:
            fo.write(str(i)+'\n')
    return w2v_sim


def pairs_to_list(pairs):
    l = []
    for p in pairs:
        l.append(p.text)
        l.append(p.hyp)
    return l


def vec_cosine_sim(a1, a2):
    return np.dot(a1, a2) / (np.linalg.norm(a1) * np.linalg.norm(a2))


def read_w2v(training=True, type="noun"):
    if training:
        if type == 'noun':
            return [float(v) for v in open('./w2v_noun_sim_train.txt', 'r').readlines()]
        if type == 'adj':
            return [float(v) for v in open('./w2v_adj_sim_train.txt', 'r').readlines()]
        if type == 'v':
            return [float(v) for v in open('./w2v_verb_sim_train.txt', 'r').readlines()]
        return [float(v) for v in open('./w2v_sim.txt', 'r').readlines()]
    else:
        if type == 'noun':
            return [float(v) for v in open('./w2v_noun_sim_test.txt', 'r').readlines()]
        if type == 'adj':
            return [float(v) for v in open('./w2v_adj_sim_test.txt', 'r').readlines()]
        if type == 'v':
            return [float(v) for v in open('./w2v_verb_sim_test.txt', 'r').readlines()]
        return [float(v) for v in open('./w2v_sim_tset.txt', 'r').readlines()]


def m_accuracy(classifier, gold, test):
    results = classifier.classify_many([fs for (fs, ll) in gold])
    correct = [ll == r for ((fs, ll), r) in zip(gold, results)]
    for r in zip(gold, results, test):
        print r[2].text.encode('gbk', 'ignore')
        print r[2].hyp.encode('gbk', 'ignore')
        print r
    if correct:
        return sum(correct) / len(correct)
    else:
        return 0


def rte_classifier():
    train_set = rte_corpus.pairs(['rte1_dev.xml', 'rte2_dev.xml', 'rte3_dev.xml'])
    test_set = rte_corpus.pairs(['rte1_test.xml'])
    featurized_train_set = rte_featurize(train_set, True)
    featurized_test_set = rte_featurize(test_set, False, test_id=0)
    print('Training classifier...')
    svm = SklearnClassifier(LinearSVC())
    clf_svm = svm.train(featurized_train_set)
    # clf_nb = nltk.NaiveBayesClassifier.train(featurized_train_set)
    # clf_gis = MaxentClassifier.train(featurized_train_set, 'GIS')
    # clf_iis = MaxentClassifier.train(featurized_train_set, 'IIS')
    # clf_dt = SklearnClassifier(RandomForestClassifier(random_state=0)).train(featurized_train_set)
    # clf_dt = DecisionTreeClassifier.train(featurized_train_set)
    print('Testing classifier...')
    # acc = m_accuracy(clf, featurized_test_set, test_set)

    # acc_dt = accuracy(clf_dt, featurized_test_set)
    # acc_gis = accuracy(clf_gis, featurized_test_set)
    # acc_iis = accuracy(clf_iis, featurized_test_set)
    acc_svm = accuracy(clf_svm, featurized_test_set)
    # acc_nb = accuracy(clf_nb, featurized_test_set)
    # print('rf Accuracy: %8.4f' % acc_dt)
    print('svm Accuracy: %8.4f' % acc_svm)
    # print('nb Accuracy: %8.4f' % acc_nb)
    # print('gis Accuracy: %8.4f' % acc_gis)
    # print('iis Accuracy: %8.4f' % acc_iis)
    print '==================================='

    # return clf


if __name__ == '__main__':
    # model = Word2Vec.load("word2vec.model")
    # pretrain_model = KeyedVectors.load_word2vec_format('D:/workplace_py/Install Package/GoogleNews-vectors-negative300.bin', binary=True)
    tfidf_sim_list_train = make_tfidf_sim("train")
    tfidf_sim_list_test = make_tfidf_sim("test")
    ne_w2v_train = read_w2v(True, 'ne')
    ne_w2v_test = read_w2v(False, 'ne')[0:]
    noun_w2v_train = read_w2v(True, 'noun')
    noun_w2v_test = read_w2v(False, 'noun')[1600:]
    adj_w2v_train = read_w2v(True, 'adj')
    adj_w2v_test = read_w2v(False, 'adj')[1600:]
    verb_w2v_train = read_w2v(True, 'v')
    verb_w2v_test = read_w2v(False, 'v')[1600:]
    rte_classifier()
    # ne_word2vec_sim(pretrain_model)

"""
1:
rf Accuracy:   0.5225
svm Accuracy:   0.5387
nb Accuracy:   0.5337
gis Accuracy:   0.5262
iis Accuracy:   0.5288
2:
rf Accuracy:   0.5288
svm Accuracy:   0.5925
nb Accuracy:   0.5613
gis Accuracy:   0.5450
iis Accuracy:   0.5513
"""
