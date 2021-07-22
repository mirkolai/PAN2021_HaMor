import numpy
from sklearn.svm.classes import SVC
import Features_manager
import Database_manager
import string
from xml.sax.saxutils import escape

# initialize database_manager
database_manager = Database_manager.make_database_manager("pan21-author-profiling-training-2021-03-14/*/","pan21-author-profiling-test-without-gold/*/")
# initialize feature_manager
feature_manager = Features_manager.make_feature_manager()
# recover tweets
language="es"
users = numpy.array(database_manager.return_users_training(language))
labels = numpy.array(feature_manager.get_label(users))
users_test = numpy.array(database_manager.return_users_test(language))

print("users len: ",len(users),len(labels), "language",language)
print("users test len: ",len(users_test), "language",language)
# recover keyword list corresponding to available features
feature_types = feature_manager.get_availablefeaturetypes()

#feature_type=feature_manager.get_availablefeaturetypes()

feature_type=[
             "ner",
             "ngrams",
             "user_morality_profile",
             "hurtlex",
             "no_swearing",
             "emoji_profile",
             "atalay_sp",

]

X,X_test,feature_name,feature_index=feature_manager.create_feature_space(users,feature_type,users_test)

print(feature_name)
print("feature space dimension X:", X.shape)
print("feature space dimension X_test:", X_test.shape)

clf = SVC(kernel="linear")

clf.fit(X,labels)
test_predict = clf.predict(X_test)
"""
    <author id="author-id"
        lang="en|es"
        type="0|1"
    />
"""

for i in range(0, len(test_predict)):
    author_id=users_test[i].id
    language=users_test[i].lang
    label=test_predict[i]
    result='<author id="'+author_id+'" \nlang="'+language+'"\n type="'+label+'"\n />'
    file=open("prediction/"+language+"/"+author_id+".xml","w")
    file.write(result)
    file.close()

"""
['ner_mean' 'targ_tot' 'ner_std' ... '🪕' '🪕 ' 'hs_yes']
feature space dimension X: (200, 505529)
feature space dimension X_test: (100, 505529)
"""

