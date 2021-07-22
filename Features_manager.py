import numpy
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
import re
from scipy.sparse import csr_matrix, hstack
from sklearn.preprocessing import normalize
from corpora_resource_hateeval import HatEval
from corpora_resource_hateeval_atalay_sp import Atalay
from corpora_resource_hateeval_fermi_sp import Fermi
from linguistic_resource_morality_in_knowledge_graph import MoralityInKnowledgeGraph
from linguistic_resource_hurtlex import HurtLex
from linguistic_resource_emoji_based import EmojiFeature
from linguistic_resource_no_swearing import NoSwearing
from linguistic_resource_racial_slur import RacialSlur
from ner_vocabulary_wikipedia import NerWikipedia
from linguistic_resource_communicative_style import CommunicationBehavior
from corpora_resource_morality_prediction import MoralAttitude


class Features_manager(object):

    def __init__(self):
        """You could add new feature types in global_feature_types_list
            global_feature_types_list is  a dictionary containing the feature space matrix for each feature type

            if you want to add a new feature:
            1. chose a keyword for defining the  feature type
            2. define a function function_name(self,tweets,tweet_test=None) where:

                tweets: Array of  tweet objects belonging to the training set
                tweet_test: Optional. Array of tweet objects belonging to the test set

                return:
                X_train: The feature space of the training set (numpy.array)
                X_test: The feature space of the test set, if test  set was defined (numpy.array)
                feature_names: An array containing the names of the features used for creating the feature space (array)

        """
        self.global_feature_types_list = {
             "ngrams":                        self.get_ngrams_features,
             "user_morality_profile":         self.get_user_morality_profile_features,
             "hurtlex":                       self.get_user_hurtlex_profile_features,
             "no_swearing":                   self.get_user_no_swearing_profile_features,
             "racial_slur":                   self.get_user_racial_slur_profile_features,
             "emoji_profile":                 self.get_bag_of_emoji_features,
             "bio_profile":                   self.get_bio_profile_features,
             "hateeval_prediction":           self.get_hateeval_prediction_features,
             "morality_prediction_en":        self.get_morality_prediction_features,
             "atalay_sp":                     self.get_atalay_prediction_features,
             "fermi_en":                      self.get_fermi_prediction_features,
             "numhashtag":                    self.get_numhashtag_features,
             "nummention":                    self.get_nummention_features,
             "ner":                           self.get_user_ner_features,
             "puntuactionmarks":              self.get_puntuaction_marks_features,
             "length":                        self.get_length_features,
             "uppercase":                     self.get_uppercase_features,
             "communicative_styles":          self.get_user_communicative_style,
        }

        return

    def get_availablefeaturetypes(self):
        """
        Return un array containing the keyword corresponding to  available feature types
        :return: un array containing the keyword corresponding to  available feature types
        """
        return np.array([x for x in self.global_feature_types_list.keys()])

    def get_label(self, users):
        """
        Return un array containing the label for each tweet
        :param users:  Array of Tweet Objects
        :return: Array of label
        """
        return [user.label for user in users]

    # features extractor
    def create_feature_space(self, tweets, feature_types_list=None, tweet_test=None):
        """

        :param tweets: Array of  tweet objects belonging to the training set
        :param feature_types_list: Optional. array of keyword corresponding to global_feature_types_list (accepted  values are:
            "ngrams"
            "ngramshashtag"
            "chargrams"
            "numhashtag"
            "puntuactionmarks"
            "length")
            If not defined, all available features are used


            You could add new features in global_feature_types_list. See def __init__(self).


        :param tweet_test: Optional. Array of tweet objects belonging to the test set
        :return:

        X: The feature space of the training set
        X_test: The feature space of the test set, if test  set was defined
        feature_names: An array containing the names of the features used  for  creating the feature space
        feature_type_indexes: A numpy array of length len(feature_type).
                           Given feature_type i, feature_type_indexes[i] contains
                           the list of the index columns of the feature space matrix for the feature type i

        How to use the output, example:

        feature_type=feature_manager.get_avaiblefeaturetypes()

        print(feature_type)  # available feature types
        [
          "puntuactionmarks",
          "length",
          "numhashtag",
        ]
        print(feature_name)  # the name of all feature corresponding to the  number of columns of X
        ['feature_exclamation', 'feature_question',
        'feature_period', 'feature_comma', 'feature_semicolon','feature_overall',
        'feature_charlen', 'feature_avgwordleng', 'feature_numword',
        'feature_numhashtag' ]

        print(feature_type_indexes)
        [ [0,1,2,3,4,5],
          [6,7,8],
          [9]
        ]

        print(X) #feature space 3X10 using "puntuactionmarks", "length", and "numhashtag"
        numpy.array([
        [0,1,0,0,0,1,1,0,0,1], # vector rapresentation of the document 1
        [0,1,1,1,0,1,1,0,0,1], # vector rapresentation of the document 2
        [0,1,0,1,0,1,0,1,1,1], # vector rapresentation of the document 3

        ])

        # feature space 3X6 obtained using only "puntuactionmarks"
        print(X[:, feature_type_indexes[feature_type.index("puntuactionmarks")]])
        numpy.array([
        [0,1,0,0,0,1], # vector representation of the document 1
        [0,1,1,1,0,1], # vector representation of the document 2
        [0,1,0,1,0,1], # vector representation of the document 3

        ])

        """

        if feature_types_list is None:
            feature_types_list = self.get_availablefeaturetypes()

        if tweet_test is None:
            all_feature_names = []
            all_feature_index = []
            all_X = []
            index = 0
            for key in feature_types_list:
                X, feature_names = self.global_feature_types_list[key](tweets)

                current_feature_index = []
                for i in range(0, len(feature_names)):
                    current_feature_index.append(index)
                    index += 1
                all_feature_index.append(current_feature_index)

                all_feature_names = np.concatenate((all_feature_names, feature_names))
                if all_X != []:
                    all_X = csr_matrix(hstack((all_X, X)))
                else:
                    all_X = X

            return all_X, all_feature_names, np.array(all_feature_index)
        else:
            all_feature_names = []
            all_feature_index = []
            all_X = []
            all_X_test = []
            index = 0
            for key in feature_types_list:
                X, X_test, feature_names = self.global_feature_types_list[key](tweets, tweet_test)
                print(key, X.shape, X_test.shape)
                current_feature_index = []
                for i in range(0, len(feature_names)):
                    current_feature_index.append(index)
                    index += 1
                all_feature_index.append(current_feature_index)

                all_feature_names = np.concatenate((all_feature_names, feature_names))
                if all_X != []:
                    all_X = csr_matrix(hstack((all_X, X)))
                    all_X_test = csr_matrix(hstack((all_X_test, X_test)))
                else:
                    all_X = X
                    all_X_test = X_test

            return all_X, all_X_test, all_feature_names, np.array(all_feature_index)

    def get_ngrams_features(self, users, users_test=None):
        """

        :param users: Array of  Tweet objects. Training set.
        :param users_test: Optional Array of  Tweet objects. Test set.
        :return:

        X_train: The feature space of the training set
        X_test: The feature space of the test set, if test  set was defined
        feature_names:  An array containing the names of the features used  for  creating the feature space
        """
        # CountVectorizer return a numpy matrix
        # row number of tweets
        # column number of 1-3gram in the dictionary
        print("Calculating binary ngram_range=(1,3) feature...")

        tfidfVectorizer = CountVectorizer(ngram_range=(1, 3),
                                          analyzer="word",
                                          # stop_words="english",
                                          lowercase=True,
                                          binary=True,
                                          max_features=500000)

        if users_test is None:
            feature = []
            for user in users:
                texts = ''
                for tweet in user.tweets:
                    texts += tweet.text + " "
                feature.append(texts)

            tfidfVectorizer = tfidfVectorizer.fit(feature)

            X = tfidfVectorizer.transform(feature)

            feature_names = tfidfVectorizer.get_feature_names()

            return X, feature_names
        else:
            feature = []
            feature_test = []
            for user in users:
                texts = ''
                for tweet in user.tweets:
                    texts += tweet.text + " "
                feature.append(texts)

            for user in users_test:
                texts = ''
                for tweet in user.tweets:
                    texts += tweet.text + " "
                feature_test.append(texts)

            tfidfVectorizer = tfidfVectorizer.fit(feature)

            X_train = tfidfVectorizer.transform(feature)
            X_test = tfidfVectorizer.transform(feature_test)

            feature_names = tfidfVectorizer.get_feature_names()

            return X_train, X_test, feature_names

    def get_user_morality_profile_features(self, users, users_test=None):
        print("Calculating user morality profile feature...")
        mkg = MoralityInKnowledgeGraph(users[0].lang)

        if users_test is None:
            feature = []
            for user in users:
                concepts, values = mkg.get_user_morality_profile([tweet.text for tweet in user.tweets])
                feature.append(values)

            feature_names = concepts

            return csr_matrix(np.vstack(feature)), feature_names
        else:
            feature = []
            feature_test = []
            for user in users:
                concepts, values = mkg.get_user_morality_profile([tweet.text for tweet in user.tweets])
                feature.append(values)

            feature_names = concepts

            for user in users_test:
                concepts, values = mkg.get_user_morality_profile([tweet.text for tweet in user.tweets])
                feature_test.append(values)

            feature_names = concepts

            return csr_matrix(np.vstack(feature)), csr_matrix(np.vstack(feature_test)), feature_names

    def get_user_hurtlex_profile_features(self, users, users_test=None):
        print("Calculating hurtlex profile feature...")

        hl = HurtLex(users[0].lang)

        if users_test is None:
            feature = []
            for user in users:
                concepts, values = hl.get_user_hurtlex_profile([tweet.text for tweet in user.tweets])
                feature.append(values)

            feature_names = concepts

            return csr_matrix(np.vstack(feature)), feature_names
        else:
            feature = []
            feature_test = []
            for user in users:
                concepts, values = hl.get_user_hurtlex_profile([tweet.text for tweet in user.tweets])
                feature.append(values)

            feature_names = concepts

            for user in users_test:
                concepts, values = hl.get_user_hurtlex_profile([tweet.text for tweet in user.tweets])
                feature_test.append(values)

            feature_names = concepts

            return csr_matrix(np.vstack(feature)), csr_matrix(np.vstack(feature_test)), feature_names

    def get_user_racial_slur_profile_features(self, users, users_test=None):
        print("Calculating Racial Slur profile feature...")

        racial_slur = RacialSlur(users[0].lang)

        if users_test is None:
            feature = []
            for user in users:
                concepts, values = racial_slur.get_user_racial_slur_profile([tweet.text for tweet in user.tweets])
                feature.append(values)

            feature_names = concepts

            return csr_matrix(np.vstack(feature)), feature_names
        else:
            feature = []
            feature_test = []
            for user in users:
                concepts, values = racial_slur.get_user_racial_slur_profile([tweet.text for tweet in user.tweets])
                feature.append(values)

            feature_names = concepts

            for user in users_test:
                concepts, values = racial_slur.get_user_racial_slur_profile([tweet.text for tweet in user.tweets])
                feature_test.append(values)

            feature_names = concepts

            return csr_matrix(np.vstack(feature)), csr_matrix(np.vstack(feature_test)), feature_names

    def get_user_no_swearing_profile_features(self, users, users_test=None):
        print("Calculating no swearing profile feature...")

        hl = NoSwearing(users[0].lang)

        if users_test is None:
            feature = []
            for user in users:
                concepts, values = hl.get_user_no_swearing_profile([tweet.text for tweet in user.tweets])
                feature.append(values)

            feature_names = concepts

            return csr_matrix(np.vstack(feature)), feature_names
        else:
            feature = []
            feature_test = []
            for user in users:
                concepts, values = hl.get_user_no_swearing_profile([tweet.text for tweet in user.tweets])
                feature.append(values)

            feature_names = concepts

            for user in users_test:
                concepts, values = hl.get_user_no_swearing_profile([tweet.text for tweet in user.tweets])
                feature_test.append(values)

            feature_names = concepts

            return csr_matrix(np.vstack(feature)), csr_matrix(np.vstack(feature_test)), feature_names
    def get_user_ner_features(self, users, users_test=None):
        print("Calculating ner profile feature...")

        hl = NerWikipedia(users[0].lang)

        if users_test is None:
            feature  = []
            i=0
            for user in users:
                #print(user)
                i+=1
                print(i,len(users))

                entities=hl.get_user_entities(user.id)
                concepts,values = hl.get_user_wikipedia_categories(entities)
                #print(values)
                feature.append(values)


            feature_names=concepts

            return csr_matrix(np.vstack(feature)), feature_names
        else:
            feature  = []
            feature_test  = []
            for user in users:
                #print(user)

                entities=hl.get_user_entities(user.id)
                concepts,values = hl.get_user_wikipedia_categories(entities)
                #print(values)
                feature.append(values)

            feature_names=concepts

            for user in users_test:
                #print(user)

                entities=hl.get_user_entities(user.id,True)
                concepts,values = hl.get_user_wikipedia_categories(entities)
                #print(values)
                feature_test.append(values)
            print(feature_test)

            feature_names=concepts

            return csr_matrix(np.vstack(feature)), csr_matrix(np.vstack(feature_test)), feature_names

    def get_user_communicative_style(self, users, users_test=None):
        print("Calculating Communication Behavior profile feature...")

        hl = CommunicationBehavior(users[0].lang)

        if users_test is None:
            feature  = []
            for user in users:
                concepts,values = hl.get_user_communicative_styles([tweet.text for tweet in user.tweets])
                #print(values)
                feature.append(values)


            feature_names=concepts

            return csr_matrix(np.vstack(feature)), feature_names
        else:
            feature  = []
            feature_test  = []
            for user in users:
                concepts, values = hl.get_user_communicative_styles([tweet.text for tweet in user.tweets])
                #print(values)
                feature.append(values)

            feature_names=concepts

            for user in users_test:
                concepts, values = hl.get_user_communicative_styles([tweet.text for tweet in user.tweets])
                #print(values)
                feature_test.append(values)

            feature_names=concepts

            return csr_matrix(np.vstack(feature)), csr_matrix(np.vstack(feature_test)), feature_names

    def get_bag_of_emoji_features(self, users, users_test=None):
        print("Calculating emoji profile feature...")
        emojiFeature = EmojiFeature(users[0].lang)
        tfidfVectorizer = CountVectorizer(ngram_range=(1, 2),
                                          analyzer="char",
                                          # stop_words="english",
                                          binary=True,
                                          max_features=500000)

        if users_test is None:
            feature = []
            for user in users:
                texts = emojiFeature.get_user_emoji_features([tweet.text for tweet in user.tweets])
                feature.append(texts)

            tfidfVectorizer = tfidfVectorizer.fit(feature)

            X = tfidfVectorizer.transform(feature)

            feature_names = tfidfVectorizer.get_feature_names()
            return X, feature_names
        else:
            feature = []
            feature_test = []
            for user in users:
                texts = emojiFeature.get_user_emoji_features([tweet.text for tweet in user.tweets])
                feature.append(texts)

            for user in users_test:
                texts = emojiFeature.get_user_emoji_features([tweet.text for tweet in user.tweets])
                feature_test.append(texts)

            tfidfVectorizer = tfidfVectorizer.fit(feature)

            X_train = tfidfVectorizer.transform(feature)
            X_test = tfidfVectorizer.transform(feature_test)

            feature_names = tfidfVectorizer.get_feature_names()
            print(feature_names)

            return X_train, X_test, feature_names

    def get_bio_profile_features(self, users, users_test=None):
        print("Calculating bio profile feature...")

        emojiFeature = EmojiFeature(users[0].lang)

        if users_test is None:
            feature = []
            for user in users:
                concepts, values = emojiFeature.get_bio_features([tweet.text for tweet in user.tweets])
                feature.append(values)

            feature_names = concepts

            return csr_matrix(np.vstack(feature)), feature_names
        else:
            feature = []
            feature_test = []
            for user in users:
                concepts, values = emojiFeature.get_bio_features([tweet.text for tweet in user.tweets])
                feature.append(values)

            feature_names = concepts

            for user in users_test:
                concepts, values = emojiFeature.get_bio_features([tweet.text for tweet in user.tweets])
                feature_test.append(values)

            feature_names = concepts

            return csr_matrix(np.vstack(feature)), csr_matrix(np.vstack(feature_test)), feature_names

    def get_hateeval_prediction_features(self, users, users_test=None):
        print("Calculating hateeval profile feature...")

        hateeval = HatEval(users[0].lang)

        if users_test is None:
            feature = []
            i=0

            for user in users:
                i += 1
                print("predicting user ",i, "/", len(users))
                concepts, values = hateeval.predict_user([tweet.text for tweet in user.tweets])
                feature.append(values)

            feature_names = concepts

            return csr_matrix(np.vstack(feature)), feature_names
        else:
            feature = []
            feature_test = []
            i = 0
            for user in users:
                i += 1
                print("predicting user ",i, "/", len(users))
                concepts, values = hateeval.predict_user([tweet.text for tweet in user.tweets])
                feature.append(values)

            feature_names = concepts
            i = 0
            for user in users_test:
                i += 1
                print("predicting user ",i, "/", len(users_test))
                concepts, values = hateeval.predict_user([tweet.text for tweet in user.tweets])
                feature_test.append(values)

            feature_names = concepts

            return csr_matrix(np.vstack(feature)), csr_matrix(np.vstack(feature_test)), feature_names
    def get_morality_prediction_features(self, users, users_test=None):
        print("Calculating morality profile feature...")

        morality = MoralAttitude(users[0].lang)

        if users_test is None:
            feature = []
            for user in users:
                concepts, values = morality.predict_user_morality([tweet.text for tweet in user.tweets])
                feature.append(values)

            feature_names = concepts

            return csr_matrix(np.vstack(feature)), feature_names
        else:
            feature = []
            feature_test = []
            i = 0
            for user in users:
                i += 1
                print(i, "/", len(users))
                concepts, values = morality.predict_user_morality([tweet.text for tweet in user.tweets])
                feature.append(values)

            feature_names = concepts
            i = 0
            for user in users_test:
                i += 1
                print(i, "/", len(users_test))
                concepts, values = morality.predict_user_morality([tweet.text for tweet in user.tweets])
                feature_test.append(values)

            feature_names = concepts

            return csr_matrix(np.vstack(feature)), csr_matrix(np.vstack(feature_test)), feature_names

    def get_atalay_prediction_features(self, users, users_test=None):
        print("Calculating no atalay profile feature...")

        hateeval = Atalay(users[0].lang)

        if users_test is None:
            feature = []
            i=0
            for user in users:
                i+=1
                print("predicting user",i ,"/", len(users))
                concepts, values = hateeval.predict_user([tweet.text for tweet in user.tweets])
                feature.append(values)

            feature_names = concepts

            return csr_matrix(np.vstack(feature)), feature_names
        else:
            feature = []
            feature_test = []
            for user in users:
                concepts, values = hateeval.predict_user([tweet.text for tweet in user.tweets])
                feature.append(values)

            feature_names = concepts

            for user in users_test:
                concepts, values = hateeval.predict_user([tweet.text for tweet in user.tweets])
                feature_test.append(values)

            feature_names = concepts

            return csr_matrix(np.vstack(feature)), csr_matrix(np.vstack(feature_test)), feature_names

    def get_fermi_prediction_features(self, users, users_test=None):
        print("Calculating fermi profile feature...")

        hateeval = Fermi(users[0].lang)

        if users_test is None:
            feature = []
            for user in users:
                concepts, values = hateeval.predict_user([tweet.text for tweet in user.tweets])
                feature.append(values)

            feature_names = concepts

            return csr_matrix(np.vstack(feature)), feature_names
        else:
            feature = []
            feature_test = []
            for user in users:
                concepts, values = hateeval.predict_user([tweet.text for tweet in user.tweets])
                feature.append(values)

            feature_names = concepts

            for user in users_test:
                concepts, values = hateeval.predict_user([tweet.text for tweet in user.tweets])
                feature_test.append(values)

            feature_names = concepts

            return csr_matrix(np.vstack(feature)), csr_matrix(np.vstack(feature_test)), feature_names

    def get_numhashtag_features(self, users, users_test=None):
        print("Calculating num hashtag feature...")
        # This method extracts a single column (feature_numhashtag)
        # len(tweets) rows of 1 column
        # sr_matrix(np.vstack(feature)) convert to an array of dimension len(tweets)X1

        if users_test is None:
            feature = []

            for user in users:
                user_feature = []
                for tweet in user.tweets:
                    numhashtag = len(re.findall(r"#(\w+)", tweet.text))
                    user_feature.append(numhashtag)
                feature.append([np.sum(user_feature),
                                np.mean(user_feature),
                                numpy.std(user_feature)
                                ])
                #print(feature)

            return csr_matrix(np.vstack(feature)), \
                   ["feature_numhashtag", "feature_avg_numhashtag", "feature_std_numhashtag"]

        else:
            feature = []
            feature_test = []

            for user in users:
                user_feature = []
                for tweet in user.tweets:
                    numhashtag = len(re.findall(r"#(\w+)", tweet.text))
                    user_feature.append(numhashtag)
                feature.append([np.sum(user_feature),
                                np.mean(user_feature),
                                numpy.std(user_feature)
                                ])

            for user in users_test:
                user_feature = []
                for tweet in user.tweets:
                    numhashtag = len(re.findall(r"#(\w+)", tweet.text))
                    user_feature.append(numhashtag)
                feature_test.append([np.sum(user_feature),
                                     np.mean(user_feature),
                                     numpy.std(user_feature)
                                     ])

            return csr_matrix(np.vstack(feature)), csr_matrix(np.vstack(feature_test)), \
                   ["feature_numhashtag", "feature_avg_numhashtag", "feature_std_numhashtag"]

    def get_nummention_features(self, users, users_test=None):
        print("Calculating num mention feature...")
        # This method extracts a single column (feature_numhashtag)
        # len(tweets) rows of 1 column
        # sr_matrix(np.vstack(feature)) convert to an array of dimension len(tweets)X1

        if users_test is None:
            feature = []

            for user in users:
                user_feature = []
                for tweet in user.tweets:
                    numhashtag = len(re.findall(r"@(\w+)", tweet.text))
                    user_feature.append(numhashtag)
                feature.append([np.sum(user_feature),
                                np.mean(user_feature),
                                numpy.std(user_feature)
                                ])

            return csr_matrix(np.vstack(feature)), \
                   ["feature_nummentinon", "feature_avg_nummentinon", "feature_std_nummentinon"]

        else:
            feature = []
            feature_test = []

            for user in users:
                user_feature = []
                for tweet in user.tweets:
                    numhashtag = len(re.findall(r"@(\w+)", tweet.text))
                    user_feature.append(numhashtag)
                feature.append([np.sum(user_feature),
                                np.mean(user_feature),
                                numpy.std(user_feature)
                                ])

            for user in users_test:
                user_feature = []
                for tweet in user.tweets:
                    numhashtag = len(re.findall(r"@(\w+)", tweet.text))
                    user_feature.append(numhashtag)
                feature_test.append([np.sum(user_feature),
                                     np.mean(user_feature),
                                     numpy.std(user_feature)
                                     ])

            return csr_matrix(np.vstack(feature)), csr_matrix(np.vstack(feature_test)), \
                   ["feature_nummentinon", "feature_avg_nummentinon", "feature_std_nummentinon"]

    def get_puntuaction_marks_features(self, users, users_test=None):
        print("Calculating puntuaction marks feature...")
        # This method extracts a single column (feature_numhashtag)
        # len(tweets) rows of 6 columns
        # sr_matrix(np.vstack(feature)) convert to an array of dimension len(tweets)X6

        if users_test is None:
            feature = []

            for user in users:
                user_feature = []
                for tweet in user.tweets:
                    user_feature.append([
                        len(re.findall(r"[!]", tweet.text)),
                        len(re.findall(r"[?]", tweet.text)),
                        len(re.findall(r"[.]", tweet.text)),
                        len(re.findall(r"[,]", tweet.text)),
                        len(re.findall(r"[;]", tweet.text)),
                        len(re.findall(r"[!?.,;]", tweet.text)),
                    ])
                feature.append(list(np.sum(user_feature, axis=0)) +
                               list(np.mean(user_feature, axis=0)) +
                               list(np.std(user_feature, axis=0)))

            return csr_matrix(feature), \
                   ["feature_exclamation",
                    "feature_question",
                    "feature_period",
                    "feature_comma",
                    "feature_semicolon",
                    "feature_overall",

                    "feature_avg_exclamation",
                    "feature_avg_question",
                    "feature_avg_period",
                    "feature_avg_comma",
                    "feature_avg_semicolon",
                    "feature_avg_overall",

                    "feature_std_exclamation",
                    "feature_std_question",
                    "feature_std_period",
                    "feature_std_comma",
                    "feature_std_semicolon",
                    "feature_std_overall",
                    ]


        else:
            feature = []
            feature_test = []

            for user in users:
                user_feature = []
                for tweet in user.tweets:
                    user_feature.append([
                        len(re.findall(r"[!]", tweet.text)),
                        len(re.findall(r"[?]", tweet.text)),
                        len(re.findall(r"[.]", tweet.text)),
                        len(re.findall(r"[,]", tweet.text)),
                        len(re.findall(r"[;]", tweet.text)),
                        len(re.findall(r"[!?.,;]", tweet.text)),
                    ])
                feature.append(list(np.sum(user_feature, axis=0)) +
                               list(np.mean(user_feature, axis=0)) +
                               list(np.std(user_feature, axis=0)))

            for user in users_test:
                user_feature = []
                for tweet in user.tweets:
                    user_feature.append([
                        len(re.findall(r"[!]", tweet.text)),
                        len(re.findall(r"[?]", tweet.text)),
                        len(re.findall(r"[.]", tweet.text)),
                        len(re.findall(r"[,]", tweet.text)),
                        len(re.findall(r"[;]", tweet.text)),
                        len(re.findall(r"[!?.,;]", tweet.text)),
                    ])
                feature_test.append(
                    list(np.sum(user_feature, axis=0)) +
                    list(np.mean(user_feature, axis=0)) +
                    list(np.std(user_feature, axis=0)))

            return csr_matrix(feature), csr_matrix(feature_test), \
                   ["feature_exclamation",
                    "feature_question",
                    "feature_period",
                    "feature_comma",
                    "feature_semicolon",
                    "feature_overall",

                    "feature_avg_exclamation",
                    "feature_avg_question",
                    "feature_avg_period",
                    "feature_avg_comma",
                    "feature_avg_semicolon",
                    "feature_avg_overall",

                    "feature_std_exclamation",
                    "feature_std_question",
                    "feature_std_period",
                    "feature_std_comma",
                    "feature_std_semicolon",
                    "feature_std_overall",
                    ]

    def get_length_features(self, users, users_test=None):
        print("Calculating length feature...")
        # This method extracts a single column (feature_numhashtag)
        # len(tweets) rows of 3 columns
        # sr_matrix(np.vstack(feature)) convert to an array of dimension len(tweets)X3

        if users_test is None:
            feature = []
            for user in users:
                user_feature = []
                for tweet in user.tweets:
                    user_feature.append([
                        len(tweet.text),
                        np.average([len(w) for w in tweet.text.split(" ")]),
                        len(tweet.text.split(" ")),
                    ])
                feature.append(
                    list(np.sum(user_feature, axis=0)) +
                    list(np.mean(user_feature, axis=0)) +
                    list(np.std(user_feature, axis=0)))
            #linear numpy.mean(accuracy), 0.54 [0.575, 0.525, 0.6, 0.55, 0.45]
            return csr_matrix(feature), \
                   ["feature_charlen",
                    "feature_avgwordleng",
                    "feature_numword"

                    "feature_avg_charlen",
                    "feature_avg_avgwordleng",
                    "feature_avg_numword",

                    "feature_std_charlen",
                    "feature_std_avgwordleng",
                    "feature_std_numword"
                    ]

        else:
            feature = []
            feature_test = []

            for user in users:
                user_feature = []
                for tweet in user.tweets:
                    user_feature.append([
                        len(tweet.text),
                        np.average([len(w) for w in tweet.text.split(" ")]),
                        len(tweet.text.split(" ")),
                    ])
                feature.append(
                    list(np.sum(user_feature, axis=0)) +
                    list(np.mean(user_feature, axis=0)) +
                    list(np.std(user_feature, axis=0)))

            for user in users_test:
                user_feature = []
                for tweet in user.tweets:
                    user_feature.append([
                        len(tweet.text),
                        np.average([len(w) for w in tweet.text.split(" ")]),
                        len(tweet.text.split(" ")),
                    ])
                feature_test.append(
                    list(np.sum(user_feature, axis=0)) +
                    list(np.mean(user_feature, axis=0)) +
                    list(np.std(user_feature, axis=0)))

            return csr_matrix(feature), csr_matrix(feature_test), \
                   ["feature_charlen",
                    "feature_avgwordleng",
                    "feature_numword"

                    "feature_avg_charlen",
                    "feature_avg_avgwordleng",
                    "feature_avg_numword",

                    "feature_std_charlen",
                    "feature_std_avgwordleng",
                    "feature_std_numword"
                    ]

    ##########################################################################
    ##########################################################################
    ##########################################################################
    ##########################################################################
    ##########################################################################
    ##########################################################################
    ##########################################################################
    ##########################################################################
    ##########################################################################
    ##########################################################################
    ##########################################################################
    ##########################################################################
    ##########################################################################
    ##########################################################################
    ##########################################################################

    def get_uppercase_features(self, users, users_test=None):
        print("Calculating uppercase feature...")
        # This method extracts a single column (feature_numhashtag)
        # len(tweets) rows of 3 columns
        # sr_matrix(np.vstack(feature)) convert to an array of dimension len(tweets)X3

        if users_test is None:
            feature = []
            for user in users:
                user_feature = []
                for tweet in user.tweets:
                    text = tweet.text.replace("#USER#", " ").replace("#URL#", " ").replace("#HASHTAG#", " ")
                    user_feature.append([
                        len(re.findall(r"[A-ZÀ-Ÿ]{2,}[^A-ZÀ-Ÿ]", text)),
                        len(re.findall(r"[A-ZÀ-Ÿ]{1,}", text))
                    ])
                feature.append(
                    list(np.sum(user_feature, axis=0)) +
                    list(np.mean(user_feature, axis=0)) +
                    list(np.std(user_feature, axis=0)))

            return csr_matrix(feature), \
                   ["feature_uppercase_words",
                    "feature_uppercase_chars",

                    "feature_avg_uppercase_words",
                    "feature_avg_uppercase_chars",

                    "feature_std_uppercase_words",
                    "feature_std_uppercase_chars",
                    ]

        else:
            feature = []
            feature_test = []

            for user in users:
                user_feature = []
                for tweet in user.tweets:
                    text = tweet.text.replace("#USER#", " ").replace("#URL#", " ").replace("#HASHTAG#", " ")
                    user_feature.append([
                        len(re.findall(r"[A-ZÀ-Ÿ]{2,}[^A-ZÀ-Ÿ]", text)),
                        len(re.findall(r"[A-ZÀ-Ÿ]{1,}", text))
                    ])
                feature.append(
                    list(np.sum(user_feature, axis=0)) +
                    list(np.mean(user_feature, axis=0)) +
                    list(np.std(user_feature, axis=0)))

            for user in users_test:
                user_feature = []
                for tweet in user.tweets:
                    text = tweet.text.replace("#USER#", " ").replace("#URL#", " ").replace("#HASHTAG#", " ")
                    user_feature.append([
                        len(re.findall(r"[A-ZÀ-Ÿ]{2,}[^A-ZÀ-Ÿ]", text)),
                        len(re.findall(r"[A-ZÀ-Ÿ]{1,}", text))
                    ])
                feature_test.append(
                    list(np.sum(user_feature, axis=0)) +
                    list(np.mean(user_feature, axis=0)) +
                    list(np.std(user_feature, axis=0)))

            return csr_matrix(feature), csr_matrix(feature_test), \
                   ["feature_uppercase_words",
                    "feature_uppercase_chars",

                    "feature_avg_uppercase_words",
                    "feature_avg_uppercase_chars",

                    "feature_std_uppercase_words",
                    "feature_std_uppercase_chars",
                    ]


# inizializer
def make_feature_manager():
    features_manager = Features_manager()

    return features_manager
