import warnings
from asl_data import SinglesData


def recognize(models: dict, test_set: SinglesData):
    """ Recognize test word sequences from word models set

   :param models: dict of trained models
       {'SOMEWORD': GaussianHMM model object, 'SOMEOTHERWORD': GaussianHMM model object, ...}
   :param test_set: SinglesData object
   :return: (list, list)  as probabilities, guesses
       both lists are ordered by the test set word_id
       probabilities is a list of dictionaries where each key a word and value is Log Liklihood
           [{SOMEWORD': LogLvalue, 'SOMEOTHERWORD' LogLvalue, ... },
            {SOMEWORD': LogLvalue, 'SOMEOTHERWORD' LogLvalue, ... },
            ]
       guesses is a list of the best guess words ordered by the test set word_id
           ['WORDGUESS0', 'WORDGUESS1', 'WORDGUESS2',...]
   """
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    probabilities = []
    guesses = []
    # DONE implement the recognizer
    #
    # Recognizer Implementation
    # https://discussions.udacity.com/t/recognizer-implementation/234793

    for (X, lengths) in test_set.get_all_Xlengths().values():
        probability = {}
        select_logL = float("-inf")
        guess = None

        for word, model in models.items():
            try:
                logL = model.score(X, lengths)
                probability[word] = logL
            except:
                probability[word] = float("-inf")
                # continue

            if logL > select_logL:
                select_logL = logL
                guess = word

        probabilities.append(probability)
        guesses.append(guess)

    return probabilities, guesses
