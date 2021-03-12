import nltk
import re
import pandas
import warnings
import gensim
import numpy as np
from sklearn import preprocessing
from nltk.stem import WordNetLemmatizer
from nltk.tag import pos_tag
from nltk.corpus import stopwords
from sklearn import linear_model, svm, naive_bayes, ensemble
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import cross_validate
from sklearn.model_selection import RepeatedStratifiedKFold

warnings.filterwarnings('ignore')
wnl = WordNetLemmatizer()


def sample():
    # Specify the file path of the dataset
    file_path = "../dataset/decisions.xlsx"

    data = pandas.read_excel(file_path, 0, names=['text', 'type', 'label'])

    return data


def get_stop_word_list():
    stopWordList = set(stopwords.words('English'))

    # We self-defined several stop words
    self_define_stop_words = ['might', 'may', 'would', 'must', 'lgtm', 'could', 'good', 'bad', 'great',
                              'nice', 'well', 'better', 'best', 'worse', 'worst', 'easy', 'simple', "i'll", "ill",
                              'im', "i'm", "they're", 'theyre', 'youre', "that's", 'btw', 'thats',
                              'us', 'theres', 'shouldnt', 'didnt', 'dont', 'rather', 'also', 'next', 'early',
                              'doesnt', 'wasnt', 'pm', 'ive', 'eg', 'imho', 'etc', 'yes', 'cant', 'ok', 'ie',
                              'really', 'every', 'anyway', 'many', 'little', 'today', 'tomorrow'
                              ]
    for single_word in self_define_stop_words:
        if single_word in stopWordList:
            print(single_word)
        stopWordList.add(single_word)

    return stopWordList


def lemmatization(words):
    word_tags = pos_tag(words)

    # Lemmatize each word based on its part-of-speech
    for i in range(len(words)):
        word = words[i]
        tag = (word_tags[i])[1]

        if tag.startswith('NN'):
            # Noun
            word_ = wnl.lemmatize(word, pos='n')
        elif tag.startswith('VB'):
            # Verb
            word_ = wnl.lemmatize(word, pos='v')
        elif tag.startswith('JJ'):
            # ADJ
            word_ = wnl.lemmatize(word, pos='a')
        elif tag.startswith('R'):
            # ADV
            word_ = wnl.lemmatize(word, pos='r')
        else:
            word_ = word

        words[i] = word_

    return words


def text_preprocess(dataset):
    tokenizer = nltk.WordPunctTokenizer()

    stop_words = get_stop_word_list()
    comp = re.compile('[^A-Z^a-z ]')

    for item in dataset.iterrows():
        i = item[0]

        decision = dataset['text'][i]

        # Clean sentence by a regular expression
        decision = comp.sub('', decision)

        # Spilt the sentence into words
        decision = tokenizer.tokenize(str(decision).lower())

        # Lemmatization
        decision = lemmatization(decision)

        # Remove not needed words
        decision = [word for word in decision if word not in stop_words]

        dataset['text'][i] = decision


def feature_selection(text, label, k):
    term_count_dict = {}
    type_count_dict = {}

    n = len(text)

    for dt in set(label):
        type_count_dict[dt] = label.count(dt)

    for i in range(len(text)):
        decision_type = label[i]
        for term in set(text[i]):
            type_dict = term_count_dict.get(decision_type, {})
            term_frequency = type_dict.get(term, 0)

            type_dict[term] = term_frequency + 1
            term_count_dict[decision_type] = type_dict

    # filter
    for decision_type in type_count_dict.keys():
        for term in list(term_count_dict[decision_type].keys()):
            if term_count_dict[decision_type][term] < 2:
                del term_count_dict[decision_type][term]

    term_chi2_dict = {}

    for decision_type in type_count_dict.keys():
        for term in term_count_dict[decision_type].keys():
            a = term_count_dict[decision_type][term]

            b = 0
            for dt in type_count_dict.keys():
                if dt != decision_type:
                    b += term_count_dict[dt].get(term, 0)

            c = type_count_dict[decision_type] - a

            d = n - a - b - c

            chi2_value = (n * (a * d - b * c) ** 2) / ((a + c) * (b + d) * (a + b) * (c + d))

            type_dict = term_chi2_dict.get(decision_type, {})
            type_dict[term] = chi2_value
            term_chi2_dict[decision_type] = type_dict

    total_feature_terms = set()

    for decision_type in type_count_dict.keys():
        sort_after = dict(sorted(term_chi2_dict[decision_type].items(), key=lambda item: item[1], reverse=True))
        feature_terms_for_decision_type = list(sort_after.keys())

        topN = round(len(feature_terms_for_decision_type) * (k / 100))
        total_feature_terms |= set(feature_terms_for_decision_type[:topN])

    for i in range(len(text)):
        filtered_decision = []
        for t in text[i]:
            if t in total_feature_terms:
                filtered_decision.append(t)

        text[i] = ' '.join(filtered_decision)

    return text


def bow(data):
    bow_vector = CountVectorizer()

    data_bow = bow_vector.fit_transform(data)

    return data_bow


def tf_idf(data):
    tf_idf_vector = TfidfVectorizer()

    data_tf_idf = tf_idf_vector.fit_transform(data)

    return data_tf_idf


def word2vec(data):
    decision_sentences = []

    for s in data:
        decision_sentences.append(str(s).split(' '))

    vec_size = 100

    model = gensim.models.word2vec.Word2Vec(decision_sentences, size=vec_size)

    word2vec_data = []

    for s in decision_sentences:
        v = np.zeros(vec_size)
        count = 0

        for word in s:
            try:
                count += 1
                v += model[word]
            except KeyError:
                continue
        v /= count
        word2vec_data.append(v)

    # Since Multinomial NB fails when training negative values, we normalize features to [0, 1] range.
    min_max_scaler = preprocessing.MinMaxScaler()

    word2vec_data = min_max_scaler.fit_transform(word2vec_data)

    return word2vec_data


def training_and_evaluation(technique, name, model, data, label, report):
    pre_results = cross_validate(model, data, label,
                                 cv=RepeatedStratifiedKFold(n_splits=10, n_repeats=10, random_state=0),
                                 scoring=['precision_weighted', 'recall_weighted', 'f1_weighted'],
                                 return_estimator=True)

    # Evaluate classifiers by weighted precision, recall, and F1-score
    weighted_avg_precision = np.mean(pre_results['test_precision_weighted'])
    weighted_avg_recall = np.mean(pre_results['test_recall_weighted'])
    weighted_avg_f1 = np.mean(pre_results['test_f1_weighted'])

    if report:
        print("feature extraction technique: " + technique + "  ML:" + name)
        print("precision: %.3f" % weighted_avg_precision)
        print("recall: %.3f" % weighted_avg_recall)
        print("f1-score: %.3f" % weighted_avg_f1)
        print("\n")


def train_classifiers(feature_extraction_technique, data, label, report):
    # Four base classifiers and five ensemble classifiers using soft voting

    classifiers = [
        ('NB', naive_bayes.MultinomialNB(alpha=1.0)),
        ('LR', linear_model.LogisticRegression(C=1.0, max_iter=100, random_state=0)),
        ('SVM', svm.SVC(kernel='linear', C=1.0, random_state=0, probability=True)),
        ('RF', ensemble.RandomForestClassifier(n_estimators=100, criterion='gini', random_state=0)),
        ('SVE1', ensemble.VotingClassifier(
            estimators=[('NB', naive_bayes.MultinomialNB(alpha=1.0)),
                        ('LR', linear_model.LogisticRegression(C=1.0, max_iter=100, random_state=0)),
                        ('SVM', svm.SVC(kernel='linear', C=1.0, random_state=0, probability=True)),
                        ('RF', ensemble.RandomForestClassifier(n_estimators=100, criterion='gini', random_state=0))
                        ],
            voting='soft')
         ),
        ('SVE2', ensemble.VotingClassifier(estimators=[('NB', naive_bayes.MultinomialNB(alpha=1.0)),
                                                       ('LR', linear_model.LogisticRegression(C=1.0, max_iter=100,
                                                                                              random_state=0)),
                                                       ('SVM',
                                                        svm.SVC(kernel='linear', C=1.0, random_state=0,
                                                                probability=True))],
                                           voting='soft')),
        ('SVE3', ensemble.VotingClassifier(
            estimators=[('LR', linear_model.LogisticRegression(C=1.0, max_iter=100, random_state=0)),
                        ('SVM',
                         svm.SVC(kernel='linear', C=1.0, random_state=0, probability=True)),
                        ('RF', ensemble.RandomForestClassifier(n_estimators=100, criterion='gini', random_state=0))],
            voting='soft')),
        ('SVE4', ensemble.VotingClassifier(
            estimators=[('NB', naive_bayes.MultinomialNB(alpha=1.0)),
                        ('SVM', svm.SVC(kernel='linear', C=1.0, random_state=0, probability=True)),
                        ('RF', ensemble.RandomForestClassifier(n_estimators=100, criterion='gini', random_state=0))
                        ],
            voting='soft')
         ),
        ('SVE5', ensemble.VotingClassifier(
            estimators=[('NB', naive_bayes.MultinomialNB(alpha=1.0)),
                        ('LR', linear_model.LogisticRegression(C=1.0, max_iter=100, random_state=0)),
                        ('RF', ensemble.RandomForestClassifier(n_estimators=100, criterion='gini', random_state=0))
                        ],
            voting='soft')
         )

    ]

    for name, model in classifiers:
        training_and_evaluation(feature_extraction_technique, name, model, data, label, report)


def experiment():
    # Input data
    dataset = sample()

    # Preprocess data
    text_preprocess(dataset)

    data = dataset['text'].tolist()
    label = dataset['label'].tolist()

    data = feature_selection(data, label, 50)

    # Feature extraction techniques
    feature_extraction_techniques = [
        ('BoW', bow(data)),
        ('TF-IDF', tf_idf(data)),
        ('Word2Vec', word2vec(data)),
    ]

    for technique, data in feature_extraction_techniques:
        train_classifiers(technique, data, label, True)


if __name__ == "__main__":
    # Run this function to get our experiments results
    experiment()
