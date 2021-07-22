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
language="en"
users = numpy.array(database_manager.return_users_training(language))
labels = numpy.array(feature_manager.get_label(users))
users_test = numpy.array(database_manager.return_users_test(language))

print("users len: ",len(users),len(labels), "language",language)
print("users test len: ",len(users_test), "language",language)
# recover keyword list corresponding to available features
feature_type = feature_manager.get_availablefeaturetypes()

feature_type=\
    [
             "ner",
             "user_morality_profile",
             "racial_slur",
             "hateeval_prediction",
             "fermi_en",
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
['ner_mean' 'targ_tot' 'ner_std' 'ent_tot' 'ner_prop' 'authority_mean'
 'loyaltyvirtue_mean' 'authorityvirtue_mean' 'loyaltyvice_mean'
 'authorityvice_mean' 'loyalty_mean' 'authority_tot' 'loyaltyvirtue_tot'
 'authorityvirtue_tot' 'loyaltyvice_tot' 'authorityvice_tot' 'loyalty_tot'
 'authority_std' 'loyaltyvirtue_std' 'authorityvirtue_std'
 'loyaltyvice_std' 'authorityvice_std' 'loyalty_std' 'jews_mean'
 'arabs_mean' 'blacks_mean' 'chinese_mean' 'hispanics_mean'
 'mexicans_mean' 'asians_mean' 'mixed races_mean' 'muslims_mean'
 'jews_tot' 'arabs_tot' 'blacks_tot' 'chinese_tot' 'hispanics_tot'
 'mexicans_tot' 'asians_tot' 'mixed races_tot' 'muslims_tot' 'jews_std'
 'arabs_std' 'blacks_std' 'chinese_std' 'hispanics_std' 'mexicans_std'
 'asians_std' 'mixed races_std' 'muslims_std' 'hs_yes' 'hs_yes']
feature space dimension X: (200, 52)
feature space dimension X_test: (100, 52)"""
