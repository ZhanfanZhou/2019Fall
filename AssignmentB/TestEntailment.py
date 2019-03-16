import nltk
from nltk.classify.maxent import MaxentClassifier
from nltk.classify.scikitlearn import SklearnClassifier
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import rte as rte_corpus
import numpy as np
import Assignment1.DataPreprocessing as mpipeline
# from gensim.models import KeyedVectors
# from gensim.models import Word2Vec
from nltk.corpus.reader.rte import RTECorpusReader

train_set = rte_corpus.pairs(['rte1_dev.xml', 'rte2_dev.xml', 'rte3_dev.xml'])
rte_newtest = RTECorpusReader('D:\workplace_py\TestWN\AssignmentB',
                              ['COMP6751-RTE-10_TEST-SET_gold.xml', 'COMP6751-RTE-30_TEST-SET_gold.xml'])
test_set_1 = rte_corpus.pairs(['rte1_test.xml'])
test_set_2 = rte_corpus.pairs(['rte2_test.xml'])
test_set_3 = rte_corpus.pairs(['rte3_test.xml'])
new_1 = rte_newtest.pairs(['COMP6751-RTE-10_TEST-SET_gold.xml'])
new_2 = rte_newtest.pairs(['COMP6751-RTE-30_TEST-SET_gold.xml'])


def rte_featurize(rte_pairs, training, test_id=0):  # construct feature list
    id = 0
    rl = []
    for pair in rte_pairs:
        rl.append((rte_features(pair, id, training, test_id), pair.value))
        id += 1
    return rl


def rte_features(rtepair, id, training=True, test_id=0):  # construct feature of a pair
    if test_id == 4:
        true_test_id = 2410 + id
    else:
        true_test_id = id + test_id*800
    extractor = nltk.RTEFeatureExtractor(rtepair)
    features = {}
    features['word_overlap'] = len(extractor.overlap('word'))
    features['word_hyp_extra'] = len(extractor.hyp_extra('word'))
    features['ne_overlap'] = len(extractor.overlap('ne'))
    features['ne_hyp_extra'] = len(extractor.hyp_extra('ne'))
    features['neg_txt'] = len(extractor.negwords & extractor.text_words)
    features['neg_hyp'] = len(extractor.negwords & extractor.hyp_words)
    if training:
        features['tfidf_sim'] = tfidf_sim_list_train[id]
        # features['w2v_ne_sim'] = ne_w2v_train[id]
        # features['w2v_noun_sim'] = noun_w2v_train[id]
        # features['w2v_adj_sim'] = adj_w2v_train[id]
        # features['w2v_verb_sim'] = verb_w2v_train[id]
    else:
        features['tfidf_sim'] = tfidf_sim_list_test[true_test_id]
        # features['w2v_ne_sim'] = ne_w2v_test[true_test_id]
        # features['w2v_noun_sim'] = noun_w2v_test[true_test_id]
        # features['w2v_adj_sim'] = adj_w2v_test[true_test_id]
        # features['w2v_verb_sim'] = verb_w2v_test[true_test_id]
    return features


def tf_idf(which):  # construct tf-idf vectors
    cv = TfidfVectorizer(binary=False, decode_error='ignore', stop_words='english')
    if which == 'train':
        vec = cv.fit_transform(pairs_to_list(train_set))
        return vec
    else:
        # vec = cv.fit_transform(pairs_to_list(rte_corpus.pairs(['rte1_test.xml', 'rte2_test.xml', 'rte3_test.xml'])))
        vec = cv.fit_transform(pairs_to_list(rte_corpus.pairs(['rte1_test.xml', 'rte2_test.xml', 'rte3_test.xml'])+new_1+new_2))
        return vec


def make_tfidf_sim(train_or_test):  # construct tf-idf similarities within pairs, store in a list
    tfidf = tf_idf(train_or_test)
    simMatrix = (tfidf * tfidf.T).A
    sim_list = []
    for i in range(0, tfidf.shape[0], 2):
        sim_list.append(simMatrix[i, i+1])
    return sim_list


def word2vec_sim(pretrain_model, train=True, output_file='./new.txt'):  # construct word2vec similarities store in a list
    if train:
        sents = pairs_to_list(train_set)
    else:
        # sents = pairs_to_list(rte_corpus.pairs(['rte1_test.xml', 'rte2_test.xml', 'rte3_test.xml']))
        sents = pairs_to_list(new_1+new_2)
    nes = mpipeline.detect_n(sents)  # call detect_n/detect_v/detect_adj/detect_ne
    w2v_sim = []
    for i in range(0, len(nes), 2):
        if nes[i] == [] or nes[i+1] == []:
            w2v_sim.append(0.0)
        else:
            total_text = []
            total_hyp = []
            mdict = {}
            for ne in nes[i]:
                tks = nltk.word_tokenize(ne)
                for word in tks:
                    try:
                        mdict[word] = pretrain_model.wv[word]
                    except KeyError:
                        pass
                ne_vec = sum([mdict[word] for word in tks if word in mdict.keys()])
                total_text.append(ne_vec)  # get a entity vector, append in
            for ne in nes[i+1]:  # entities in hyp
                tks = nltk.word_tokenize(ne)
                for word in tks:
                    try:
                        mdict[word] = pretrain_model.wv[word]
                    except KeyError:
                        pass
                ne_vec = sum([mdict[word] for word in tks if word in mdict.keys()])
                total_hyp.append(ne_vec)  # get a entity vector, append in
            sim = vec_cosine_sim(sum(total_text), sum(total_hyp))  # a pair
            w2v_sim.append(sim)
    fo = file(output_file, 'w')
    for i in w2v_sim:
        if type(i) == type(np.zeros(3)):
            fo.write("0.0\n")
        else:
            fo.write(str(i)+'\n')
    return w2v_sim


def pairs_to_list(pairs):  # convert
    l = []
    for p in pairs:
        l.append(p.text)
        l.append(p.hyp)
    return l


def vec_cosine_sim(a1, a2):
    return np.dot(a1, a2) / (np.linalg.norm(a1) * np.linalg.norm(a2))


def read_w2v(training=True, type="noun"):  # read similarity in stored file
    if training:
        if type == 'noun':
            return [float(v) for v in open('./w2v_noun_sim_train.txt', 'r').readlines()]
        if type == 'adj':
            return [float(v) for v in open('./w2v_adj_sim_train.txt', 'r').readlines()]
        if type == 'v':
            return [float(v) for v in open('./w2v_verb_sim_train.txt', 'r').readlines()]
        return [float(v) for v in open('./w2v_ne_sim_train.txt', 'r').readlines()]
    else:
        if type == 'noun':
            return [float(v) for v in open('./w2v_noun_sim_test.txt', 'r').readlines()]
        if type == 'adj':
            return [float(v) for v in open('./w2v_adj_sim_test.txt', 'r').readlines()]
        if type == 'v':
            return [float(v) for v in open('./w2v_verb_sim_test.txt', 'r').readlines()]
        return [float(v) for v in open('./w2v_sim_tset.txt', 'r').readlines()]


def m_accuracy(classifier, gold, test=False, printable=False):  # print test details
    results = classifier.classify_many([fs for (fs, l) in gold])
    final = zip(gold, results)
    correct = [l == r for ((fs, l), r) in final]
    fp = [l == 0 and r == 1 for ((fs, l), r) in final]
    fn = [l == 1 and r == 0 for ((fs, l), r) in final]
    acc = float(sum(correct)) / len(correct)
    pre = float(sum(correct)) / float(sum(correct) + sum(fp))
    rec = float(sum(correct)) / float(sum(correct) + sum(fn))
    f1 = 2*(pre*rec)/(pre+rec)
    if printable:
        for r in zip(gold, results, test):
            print r[2].text.encode('gbk', 'ignore')
            print r[2].hyp.encode('gbk', 'ignore')
            print r
    if correct:
        print('Accuracy: %8.4f' % acc)
        print('Precision: %8.4f' % pre)
        print('Recall: %8.4f' % rec)
        print('F1-measure: %8.4f' % f1)
    else:
        return 0


def rte_classifier():  # classifier
    featurized_train_set = rte_featurize(train_set, True)
    featurized_test_set_1 = rte_featurize(test_set_1, False, test_id=0)
    featurized_test_set_2 = rte_featurize(test_set_2, False, test_id=1)
    featurized_test_set_3 = rte_featurize(test_set_3, False, test_id=2)
    featurized_new_1 = rte_featurize(new_1, False, test_id=3)
    featurized_new_2 = rte_featurize(new_2, False, test_id=4)
    testing = [featurized_test_set_1, featurized_test_set_2, featurized_test_set_3, featurized_new_1, featurized_new_2]
    print('Training classifier...')
    clf_svm = SklearnClassifier(LinearSVC()).train(featurized_train_set)
    clf_nb = nltk.NaiveBayesClassifier.train(featurized_train_set)
    clf_gis = MaxentClassifier.train(featurized_train_set, 'GIS')
    clf_iis = MaxentClassifier.train(featurized_train_set, 'IIS')
    clf_rf = SklearnClassifier(RandomForestClassifier(random_state=0)).train(featurized_train_set)
    print('Testing classifier...')
    # acc = m_accuracy(clf_rf, featurized_new_2, new_2)
    for testset in testing:
        print "=====Random Forest====="
        m_accuracy(clf_rf, testset)
        print "=====SVM====="
        m_accuracy(clf_svm, testset)
        print "=====Naive Bayes====="
        m_accuracy(clf_nb, testset)
        print "=====MaxEnt GIS====="
        m_accuracy(clf_gis, testset)
        print "======MaxEnt IIS======"
        m_accuracy(clf_iis, testset)
        print '==================================='


if __name__ == '__main__':
    # model = Word2Vec.load("word2vec.model")
    # pretrain_model = KeyedVectors.load_word2vec_format('D:/workplace_py/Install Package/GoogleNews-vectors-negative300.bin', binary=True)
    # word2vec_sim(pretrain_model, False)
    tfidf_sim_list_train = make_tfidf_sim("train")
    tfidf_sim_list_test = make_tfidf_sim("test")
    ne_w2v_train = read_w2v(True, 'ne')
    ne_w2v_test = read_w2v(False, 'ne')
    noun_w2v_train = read_w2v(True, 'noun')
    noun_w2v_test = read_w2v(False, 'noun')
    adj_w2v_train = read_w2v(True, 'adj')
    adj_w2v_test = read_w2v(False, 'adj')
    verb_w2v_train = read_w2v(True, 'v')
    verb_w2v_test = read_w2v(False, 'v')
    rte_classifier()
