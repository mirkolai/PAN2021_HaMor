import numpy
from sklearn.svm import SVC
import Features_manager
import Database_manager
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
from sklearn.model_selection import KFold
import os
import pickle
# initialize database_manager
database_manager = Database_manager.make_database_manager("pan21-author-profiling-training-2021-03-14/*/")
# initialize feature_manager
feature_manager = Features_manager.make_feature_manager()
# recover tweets
language="en"
users = numpy.array(database_manager.return_users_training(language))
labels = numpy.array(feature_manager.get_label(users))
print("users len: ",len(users), "language",language)
# recover keyword list corresponding to available features
feature_types = feature_manager.get_availablefeaturetypes()
#########################
##feature disponibili per l'inglese
########################
feature_types=[  "ngrams",
             "user_morality_profile",
             "hurtlex",
             "no_swearing",
             "racial_slur",
             "emoji_profile",
             "bio_profile",
             "hateeval_prediction",
             "morality_prediction_en",
             #"atalay_sp",
             "fermi_en",
             "numhashtag",
             "nummention",
             "ner",
             "puntuactionmarks",
             "length",
             "uppercase",
             "communicative_styles"]
# create the feature space with all available feature
if os.path.isfile("cache/"+language+"_features.pickle"):
        infile = open("cache/"+language+"_features.pickle",'rb')
        features_picke = pickle.load(infile)
        infile.close()
        X=features_picke['X']
        feature_names=features_picke['feature_names']
        feature_type_indexes=features_picke['feature_type_indexes']
else:
    print("loading from cache")
    X,feature_names,feature_type_indexes=feature_manager.create_feature_space(users,feature_types)
    features_picke={}
    features_picke['X']=X
    features_picke['feature_names']=feature_names
    features_picke['feature_type_indexes']=feature_type_indexes
    output=open("cache/"+language+"_features.pickle", "wb")
    pickle.dump(features_picke,output)
    output.close()
#########################
##feature scelte da me
########################
filtered_feature_types=\
    [        #"ngrams",#0.633
             "user_morality_profile",#0.659
             #"hurtlex", #0.683
             #"no_swearing",#0.666
             "racial_slur", #0.655
             #"emoji_profile", #0.668
             #"bio_profile",#0.666
             "hateeval_prediction", #0.653
             #"morality_prediction_en",#0.669
             "fermi_en",#0.669
             #"numhashtag", #0.6683
             #"nummention", #0.6616
             "ner", #0.654
             #"puntuactionmarks", #0.661
             #"length", #0.656
             #"uppercase", #0.639
             #"communicative_styles" #0.641
     ]#0.643

#0.72 con fermi marco - io 0.729
#     senza marco - io 0.724
#filtered_feature_types=['user_morality_profile', 'racial_slur', 'hateeval_prediction', 'ner']#0.74166
#filtered_feature_types=[numhashtag, nummention, 'user_morality_profile', 'racial_slur', 'hateeval_prediction', 'ner']#0.74166
feature_index_filtered=numpy.array([list(feature_types).index(f) for f in filtered_feature_types])#0.746
feature_index_filtered=numpy.concatenate(feature_type_indexes[list(feature_index_filtered)])
X=X[:,feature_index_filtered]
print("features:", filtered_feature_types)
print("feature space dimension:", X.shape)

accuracies=[]
fmacros=[]
#for random_state in [1,2,3]:
for random_state in range(0,100):
    kf = KFold(n_splits=5,random_state=random_state,shuffle=True)
    for index_train, index_test in kf.split(X):

        clf = SVC(kernel="linear",cache_size=500)

        clf.fit(X[index_train],labels[index_train])
        test_predict = clf.predict(X[index_test])

        #golden=numpy.concatenate((golden,labels[index_test]), axis=0)
        #print(("".join(labels[index_train])).count("1"),("".join(labels[index_train])).count("0"))
        #print(("".join(labels[index_test])).count("1"),("".join(labels[index_test])).count("0"))
        #predict=numpy.concatenate((predict,test_predict), axis=0)

        #for i in range(0,len(test_predict)):
        #    if(test_predict[i]!=labels[index_test][i]):
        #        print(users[i].id,test_predict[i],labels[index_test][i])

        prec, recall, f, support = precision_recall_fscore_support(
        labels[index_test],
        test_predict,
        beta=1)
        fmacros.append((f[0]+f[1])/2)
        accuracy = accuracy_score(
        labels[index_test],
        test_predict,
        )
        accuracies.append(accuracy)

        print("prec", "recall", "f", "support" )
        print(prec, recall, f, support )
        print("accuracy")
        print(accuracy)
print("numpy.mean(accuracy),",numpy.mean(accuracies),numpy.std(accuracies),accuracies)
print("numpy.mean(fmacros),",numpy.mean(fmacros),numpy.std(fmacros),fmacros)
