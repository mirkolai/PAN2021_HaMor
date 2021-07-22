from Tweet import make_tweet
from User import  make_user
from bs4 import BeautifulSoup
import glob


class Database_manager(object):

    users=[]
    users_test=[]

    def __init__(self,path,path_test=None):
        print(path+"*.xml")
        print("reading data...")
        file_names=glob.glob(path+"*.xml")

        for file_name in file_names:
            xml_file = open(file_name, 'r')
            data = xml_file.read()
            soup = BeautifulSoup(data, "xml")


            author = soup.find('author')
            lang=author['lang']
            label=author['class']
            author_id=file_name.split("/")[-1].replace(".xml","")
            documents = author.find_all('document')
            i=0
            tweets=[]
            for document in documents:
                tweet=make_tweet(i,document.text)
                tweets.append(tweet)
                i+=1
            user=make_user(author_id,tweets,label,lang,"train")
            self.users.append(user)
        print("data imported.")
        if path_test is not None:
            print(path_test+"*.xml")
            print("reading data test...")
            file_names=glob.glob(path_test+"*.xml")

            for file_name in file_names:
                xml_file = open(file_name, 'r')
                data = xml_file.read()
                soup = BeautifulSoup(data, "xml")


                author = soup.find('author')
                lang=author['lang']
                label=None #author['class']
                author_id=file_name.split("/")[-1].replace(".xml","")
                documents = author.find_all('document')
                i=0
                tweets=[]
                for document in documents:
                    tweet=make_tweet(i,document.text)
                    tweets.append(tweet)
                    i+=1
                user=make_user(author_id,tweets,label,lang,"test")
                self.users_test.append(user)
            print("data test imported.")
    def return_users(self):

        return self.users

    def return_users_training(self, lang):

        return [user  for user in self.users if user.type=="train" and user.lang==lang]


    def return_users_test(self, lang) :

        return [user  for user in self.users_test if user.type=="test" and  user.lang==lang]



def make_database_manager(path,path_test=None):
    database_manager = Database_manager(path,path_test)

    return database_manager


if __name__ == "__main__":
    database_manager = make_database_manager("pan21-author-profiling-training-2021-03-14/*/")
    users=database_manager.return_users_training("en")
    print(len(users))
    database_manager = make_database_manager("pan21-author-profiling-training-2021-03-14/*/","pan21-author-profiling-test-without-gold/*/")
    users=database_manager.return_users_test("en")
    users_test=database_manager.return_users_test("en")
    print(len(users))
    print(len(users_test))

