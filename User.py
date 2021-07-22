
class User(object):

    id=None
    tweets=[]
    label=None
    lang=None
    type=None

    def __init__(self, id, tweets, label,lang,type):

        self.id=id
        self.tweets=tweets
        self.label=label
        self.lang=lang
        self.type=type


def make_user(id, tweets, label, lang, type):
    """
        Return a Tweet object.
    """
    user = User(id, tweets, label, lang, type)

    return user



