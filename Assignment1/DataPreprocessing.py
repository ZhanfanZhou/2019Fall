import nltk
from nltk.tree import Tree
from nltk.corpus import movie_reviews as mrs
import pandas as pd
from nltk.chunk import ne_chunk
import re
from nltk.tag import StanfordNERTagger
from nltk.parse.stanford import StanfordParser
from nltk.parse.stanford import StanfordDependencyParser

OUTPUT_FILE = False
stanford_parser = StanfordParser('D:/workplace_py/Install Package/stanford-parser-full-2018-02-27/stanford-parser.jar',
                                 'D:/workplace_py/Install Package/stanford-parser-full-2018-02-27/stanford-parser-3.9.1-models.jar')
stanford_ner = StanfordNERTagger("D:/workplace_py/Install Package/stanford-ner-2018-02-27/classifiers/english.muc.7class.distsim.crf.ser.gz",
                                 "D:/workplace_py/Install Package/stanford-ner-2018-02-27/stanford-ner.jar")
stanford_dp = StanfordDependencyParser('D:/workplace_py/Install Package/stanford-parser-full-2018-02-27/stanford-parser.jar',
                                 'D:/workplace_py/Install Package/stanford-parser-full-2018-02-27/stanford-parser-3.9.1-models.jar')
chunk_grammar_re = r"""
  NP: {<DT|PP\$>?<JJ>+<NN>}   # chunk determiner/possessive, adjectives and noun
      {<NNP>+}                # chunk sequences of proper nouns
      {<DT|PP\$>?<JJ|CD>*<NNS|NNPS>+}
      {<DT|PP\$>?<JJ|CD>*<NN>+}
"""
regexp_grammar = r"""
  NE: {<DT|PRP>?<JJ|CD>+<NNP|NN|NNS>+}
      {<NNP><CD><,><CD>}
      {<SYM><NN|NNP|NNS>+<CC>?<NN|NNP|NNS>*<SYM>}
      {<DT|PRP>?<NNP|NN|NNS>+<IN><DT|PRP\$|PRP>?<NNP|NN|NNS>}
      {<DT>?<NNP|NN|NNS>{3,}}
"""
parser_grammar = nltk.CFG.fromstring("""
  S -> NP VP ED| PP NP VP ED
  VP -> V NP | V NP PP | Adv V
  PP -> P NP | P N | P Det N
  V -> "saw" | "ate" | "walked"
  NP -> "John" | "Mary" | "Bob" | Det N | Det N PP | N N | Adj N
  Det -> "a" | "an" | "the" | "my"|'his'
  N -> "man" | "dog" | "cat" | "apple" | "Monday"|'refrigerator'|'apple' | 'office' | "table" | "fridge"|"week"|"took"
  P -> "in" | "On" | "by" | "with" | "at" | "on"| "from"| "to"|"over"
  ED -> "."|"..."|"!"|"?"
  Adv -> "finally"
  Adj -> "Last"
  """)

def dataReader_fromCorpus():
    columns_name = ['label', 'para_name', 'raw_text']
    # ['neg/cv000_29416.txt', 'neg/cv001_19502.txt', 'neg/cv002_17424.txt'...]
    r1 = []
    r2 = []
    r3 = []
    for fileid in mrs.fileids()[2:6]:
        r1.append(fileid.split('/')[0])
        r2.append(fileid.split('/')[1])
        r3.append(mrs.raw(fileid))
    data = pd.DataFrame({columns_name[0]: r1, columns_name[1]: r2, columns_name[2]: r3})
    print ('=====READ IN CORPUS!=====')
    print data.head()
    data.info()
    return data

# all_words = nltk.FreqDist(w.lower() for w in mrs.words()[0:100])
# print(list(all_words))
# for i in all_words:
#     print i, all_words[i]


# input:[(u'what', 'WP'), (u"'", "''"), (u's', 'VBD'), (u'the', 'DT'), (u'deal', 'NN'), (u'?', '.')]
def tag_correcter(tag_list):
    new_tag = []
    for tag in tag_list:
        if tag[0] == u'"':
            new_tag.append((u'"', 'SYM'))
        elif tag[0] == u'i':
            new_tag.append((u'i', 'PRP'))
        else:
            new_tag.append(tag)
    return new_tag


# .*?[!?.]+
# [^.]+?[!?.]+
# input string
# output ['sent1','sent2'...]
def sentence_re_spliter(raw_text_string):
    pattern = re.compile(r"(\b[^!.? ][\W\w]+?[!?.][ ]*\.*[ ]*\.*)", re.X)
    # print [sent.strip() for sent in pattern.findall(raw_text_string)]
    return [sent.strip() for sent in pattern.findall(raw_text_string)]


# input: sent string like "what's the, mind ?"
# output: ['what', "'s", 'the', ',', 'mind', '?']
def word_ez_tokenizer(sent_string, punc_pattern=r"[',:;.][stdvrm]?e?"):#?->{1}
    sp = sent_string.split()
    i = 0
    while i < len(sp):
        if i in range(1, len(sp) - 2) and sp[i] == sp[i + 2] == sp[i + 1] == u'.':
            sp[i] = u'...'
            sp[i + 2] = sp[i + 1] = u''
        else:
            result = re.search(re.compile(pattern=punc_pattern), sp[i])
        if result is not None:
            sp.insert(i + 1, result.group())#inset abbreviation
            sp[i] = sp[i][0:-len(result.group())]#modify former
            i += 2
        else:
            i += 1
    sp_not_empty = [value for value in sp if value is not u'']
    return sp_not_empty


# arg2:[(u'they', 'PRP'), (u'get', 'VBP'), (u'into', 'IN'), (u'an', 'DT'), (u'accident', 'NN'), (u'.', '.')]
def chunking_with_re(grammar_re, POSed_sentence):
    cp = nltk.RegexpParser(grammar_re)
    print cp.parse(POSed_sentence)


def parse_using_stanfordparser(tokenized_sent, display_tree=False, printNP=False, printLeave=False):
    result = stanford_parser.tagged_parse(tokenized_sent)
    for item in result:
        # print item
        if display_tree:
            Tree.draw(item)
        if printNP:
            NPs = list(Tree.subtrees(item, filter=lambda x: x.label() == 'NP' and x.height() <= 6))
            for n in NPs:
                if printLeave:
                    candidate = Tree.leaves(n)
                    s = ' '.join(candidate)
                    if len(candidate) == 1:
                        if re.search(re.compile(r'[A-Z_-]+', re.X), s):
                            print s
                    else:
                        print s
                else:
                    tags = []
                    for t in Tree.subtrees(n):
                        if t.label() not in ['NP', 'S', 'VP']:
                            tags.append(t.label())
                    tagged = []
                    for w in range(len(Tree.leaves(n))):
                        tagged.append((Tree.leaves(n)[w], tags[w].encode('gbk')))
                    regexp_ner_m2(regexp_grammar, tagged)


def denpendency_stanford_dp(tokenized_sent, display_tree=False):
    result = list(stanford_dp.tagged_parse(tokenized_sent))
    for item in result[0].triples():
        if display_tree:
            Tree.draw(item)
        print item


def grammar_parser(sent):
    rd_parser = nltk.RecursiveDescentParser(parser_grammar)
    for tree in rd_parser.parse(sent):
        print(tree)


# input:['what', "'s", 'the', ',', 'mind', '?']
def ner_stanford(one_tokenized_sent):
    result = stanford_ner.tag(one_tokenized_sent)
    print 'Stanford NER: ', result
    for item in result:
        if item[1] != u'O':
            print item[0]

def transform(file_lines, preprocessing):
    if preprocessing:
        result = [line for line in file_lines if line[0] not in ['*', '%'] and line != '\n']
        return result
    else:
        return file_lines


def start_pipline():
    # file_out = open('./pos_after.log', 'w')
    df = dataReader_fromCorpus()
    print '=====START PROCESSING====='
    for raw_text in df['raw_text']:
        #do sentence spliting and word tokenize
        sents = []
        for sent in sentence_re_spliter(raw_text_string=raw_text):
            sents.append(word_ez_tokenizer(sent))
        print 'Overview: ', len(sents), 'sentences: ', sents
        for sent in sents:
            # do POS tagging
            tags = nltk.pos_tag(sent)
            corrected_tag = tag_correcter(tags)
            if OUTPUT_FILE:
                # file_out.writelines(str(corrected_tag)+'\n')
                pass
            print corrected_tag
        # do parsing
            # parse_using_stanfordparser(corrected_tag)
        # do NP chunking & NER
            ner_stanford(sent)
        #     print chunking_with_re(chunk_grammar_re, corrected_tag)
            # print ne_chunk(corrected_tag)
            print '------FINISHED A SENTENCE------'


def start_validation(path, preprocessing=False):
    validation_file = open(path, 'r')
    # file_out = open('./pos_test.log', 'w')
    print '=====START PROCESSING====='
    # read in file & format
    raw_list = transform(validation_file.readlines(), preprocessing)
    for raw_text in raw_list:
        #do sentence spliting and word tokenize
        sents = []
        for sent in sentence_re_spliter(raw_text_string=raw_text.strip()):
            sents.append(word_ez_tokenizer(sent))
        print 'Overview: ', len(sents), 'sentences: ', sents
        for sent in sents:
            # do POS tagging
            tags = nltk.pos_tag(sent)
            corrected_tag = tag_correcter(tags)
            print 'POS tagging: ', corrected_tag
            # file_out.writelines(str(corrected_tag) + '\n')
        # do parsing
            # grammar_parser([item for item in sent if item not in [u',']])
            parse_using_stanfordparser(corrected_tag)
        # do dependency analysis
        #     denpendency_stanford_dp(corrected_tag)
        # do NP chunking & NER
        #     ner_stanford(sent)
            # print chunking_with_re(chunk_grammar_re, corrected_tag)
            # print ne_chunk(corrected_tag)
            print '------FINISHED A SENTENCE------'


def regexp_ner_m1(sentence_list):
    result = []
    index = 0
    pattern_1 = re.compile(r'[A-Z_-]+', re.X)
    pattern_2 = re.compile(r'((19|20)[0-9]{2})|[0-9]{1,2}[stndrdth]{2}', re.X)
    for i in range(len(sentence_list)):
        if sentence_list[i].lower() in nltk.corpus.stopwords.words('english'):
            continue
        if sentence_list[i] in ['-', '--']:
            result.extend([[sentence_list[i+1], i+1, 'NE']])
        elif sentence_list[i] == ':' and i != 0:
            result.extend([[sentence_list[i - 1], i - 1, 'NE']])
        elif pattern_1.search(sentence_list[i]) is not None or pattern_2.search(sentence_list[i]) is not None:
            if i-1 >= 0 and sentence_list[i-1].lower() in ['the', 'a']:
                result.append([sentence_list[i-1], i-1, 'NE'])
            result.append([sentence_list[i], i, 'NE'])
    # print result
    while index < len(result):
        result[index][2] = 'B-NE'
        index += 1
        while index < len(result):
            if result[index-1][1] == result[index][1] - 1:
                result[index][2] = 'I-NE'
                index += 1
            else:
                break
    if result == []:
        print "(M1)NER: not found"
    else:
        print "(M1)NER: ", result


def regexp_ner_m2(grammar_re, tagged_sentence):
    result = []
    cp = nltk.RegexpParser(grammar_re)
    result_tree = cp.parse(tagged_sentence)
    nps = list(Tree.subtrees(result_tree, filter=lambda x: x.label() == 'NE' and x.height() <= 5))
    if nps is not []:
        # print "(M2)NE found: "
        for n in nps:
            ne_list = [i[0] for i in Tree.leaves(n)]
            s = ' '.join(ne_list)
            result.append(s)
            # print s
    return result

def ner_three_modules(path, preprocessing=False):
    # validation_file = open(path, 'r')
    # print '=====START PROCESSING====='
    # raw_list = transform(validation_file.readlines(), preprocessing)
    df = dataReader_fromCorpus()
    # for raw_text in raw_list:
    for raw_text in df['raw_text']:
        sents = []
        for sent in sentence_re_spliter(raw_text_string=raw_text.strip()):
            sents.append(word_ez_tokenizer(sent))
        for sent in sents:
            # print sent
            # regexp_ner_m1(sent)
            corrected_tag = tag_correcter(nltk.pos_tag(sent))
            # print 'POS tagging: ', corrected_tag
            # regexp_ner_m2(regexp_grammar, corrected_tag)
            try:
                parse_using_stanfordparser(corrected_tag, printNP=True)
            except Exception as e:
                print str(e)


def test_montrealgazette(path, preprocessing=False):
    validation_file = open(path, 'r')
    raw_list = transform(validation_file.readlines(), preprocessing)
    for raw_text in raw_list:
        sents = []
        for sent in sentence_re_spliter(raw_text_string=raw_text.strip()):
            sents.append(word_ez_tokenizer(sent))
        # print 'Overview: ', len(sents), 'sentences: ', sents
        for sent in sents:
            print sent
            # regexp_ner_m1(sent)
            corrected_tag = tag_correcter(nltk.pos_tag(sent))
            # print 'POS tagging: ', corrected_tag
            # regexp_ner_m2(regexp_grammar, corrected_tag)
            try:
                parse_using_stanfordparser(corrected_tag, printNP=True, printLeave=True)
            except Exception as e:
                print str(e)


def detect_ne(raw_list):
    result = []
    for raw_text in raw_list:
        # regexp_ner_m1(sent)
        corrected_tag = tag_correcter(nltk.pos_tag(nltk.word_tokenize(raw_text)))
        result.append(regexp_ner_m2(regexp_grammar, corrected_tag))
    return result


def detect_n(raw_list):
    result = []
    for raw_text in raw_list:
        # regexp_ner_m1(sent)
        corrected_tag = tag_correcter(nltk.pos_tag(nltk.word_tokenize(raw_text)))
        list_of_nouns = [w[0] for w in corrected_tag if w[1] in ['NN', 'NNP', 'NNPS', 'NNS']]
        result.append(list_of_nouns)  # list of nouns
    return result


def detect_adj(raw_list):
    result = []
    for raw_text in raw_list:
        # regexp_ner_m1(sent)
        corrected_tag = tag_correcter(nltk.pos_tag(nltk.word_tokenize(raw_text)))
        list_of_nouns = [w[0] for w in corrected_tag if w[1] in ['JJ', 'JJR', 'JJS']]
        result.append(list_of_nouns)  # list of nouns
    return result


def detect_v(raw_list):
    result = []
    for raw_text in raw_list:
        # regexp_ner_m1(sent)
        corrected_tag = tag_correcter(nltk.pos_tag(nltk.word_tokenize(raw_text)))
        list_of_nouns = [w[0] for w in corrected_tag if w[1] in ['VB', 'VBD', 'VBG', 'VBN', 'VBZ', 'VBP']]
        result.append(list_of_nouns)  # list of nouns
    return result


if __name__ == '__main__':
    # start_pipline()
    # start_validation('D:/workplace_py/TestWN/Assignment1/test.txt')
    # start_validation('D:/workplace_py/TestWN/Assignment1/Nokia_6610.txt', True)
    ner_three_modules('D:/workplace_py/TestWN/Assignment1/Nokia_6610.txt', True)
    # test_montrealgazette('D:/workplace_py/TestWN/Assignment1/montrealgazette.txt', False)
