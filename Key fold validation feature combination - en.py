from itertools import combinations
import numpy
from sklearn.svm.classes import SVC
import Features_manager
import Database_manager
from sklearn.metrics.classification import precision_recall_fscore_support, accuracy_score

import numpy
from sklearn.svm import SVC
import Features_manager
import Database_manager
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
from sklearn.model_selection import KFold
import csv

language="en"
file=open("reports/"+language+"_1_2_3_feature_combination.csv","r")
features_combination_done=[]
max_accuracy=0
max_feature=[]
content = file.readlines()
i=0
while i < len(content):
    row=content[i]
    if "]" not in content[i]:
        row=content[i]+content[i+1]
        i+=1
    this_features=row.split(",")[0].replace("'","").replace("]","").replace("[","").replace("\n","").split(" ")
    accuracy=float(row.split(",")[-2])
    if max_accuracy<accuracy:
        max_accuracy=accuracy
        max_feature=this_features
    i+=1
    #print(this_features)
    features_combination_done.append(this_features)

print("max_accuracy")
print("max_feature")
print(max_accuracy)
print(max_feature)

# initialize database_manager
database_manager = Database_manager.make_database_manager("pan21-author-profiling-training-2021-03-14/*/")
# initialize feature_manager
feature_manager = Features_manager.make_feature_manager()
# recover tweets

users = numpy.array(database_manager.return_users_training(language))
labels = numpy.array(feature_manager.get_label(users))
print("users len: ",len(users), "language",language)
# recover keyword list corresponding to available features
feature_types = feature_manager.get_availablefeaturetypes()


feature_types=numpy.array([
             "ngrams",
             "user_morality_profile",
             "hurtlex",
             "no_swearing",
             "racial_slur",
             "emoji_profile",
             "bio_profile",
             "hateeval_prediction",
             "morality_prediction_en",
             "fermi_en",
             "numhashtag",
             "nummention",
             "ner",
             "puntuactionmarks",
             "length",
             "uppercase",
             "communicative_styles",
    ])

# create the feature space with all available features
X,feature_names,feature_type_indexes=feature_manager.create_feature_space(users,feature_types)

"""
https://en.wikipedia.org/wiki/Combination
"""
print("feature space dimension X:", X.shape)

N = len(feature_types)

for K in range(1, N+1):
    for subset in combinations(range(0, N), K):
        go=True
        for feature_done in features_combination_done:
            if list(feature_types[list(subset)])==feature_done:
                go=False
                continue
        if not go:
            print("combination already done: ",feature_types[list(subset)])
            continue
        print(feature_types[list(subset)])
        feature_index_filtered=numpy.array([list(feature_types).index(f) for f in feature_types[list(subset)]])
        feature_index_filtered=numpy.concatenate(feature_type_indexes[list(feature_index_filtered)])
        X_filter=X[:,feature_index_filtered]
        print("X_filter.shape",X_filter.shape)
        accuracies=[]
        fmacros=[]
        for random_state in [1,2,3]:
            kf = KFold(n_splits=5,random_state=random_state,shuffle=True)
            for index_train, index_test in kf.split(X):
                # extract the column of the features considered in the current combination
                # the feature space is reduced
                print("feature space dimension X for ",feature_types[list(subset)],":", X_filter.shape)

                clf= SVC(kernel='linear')

                clf.fit(X_filter[index_train],labels[index_train])
                test_predict = clf.predict(X_filter[index_test])

                prec, recall, f, support = precision_recall_fscore_support(
                    labels[index_test],
                    test_predict,
                    beta=1)

                accuracy = accuracy_score(
                    test_predict,
                    labels[index_test]
                )
                accuracies.append(accuracy)
                fmacros.append((f[0]+f[1])/2)
        print("numpy.mean(accuracy),",numpy.mean(accuracies),accuracies)
        print("numpy.mean(fmacros),",numpy.mean(fmacros),fmacros)
        if(max_accuracy<numpy.mean(accuracies)):
            max_accuracy=numpy.mean(accuracies)
            max_feature=feature_types[list(subset)]
        print("BEST RESULT UNTIL NOW")
        print(max_feature)
        print(max_accuracy)

        file=open("reports/"+language+"_1_2_3_feature_combination.csv","a")
        file.write(str(feature_types[list(subset)])+","+
                   str(X_filter.shape)+","+
                   str(numpy.mean(accuracies))+","+
                   str(numpy.mean(fmacros))+"\n")
        file.close()
