import pandas as pd
import os
import sys
import nltk
from collections import defaultdict
import numpy as np
from pickle import dump, load
from nltk.corpus import brown
import re
from sklearn.model_selection import train_test_split
from sklearn_crfsuite import CRF
from sklearn_crfsuite import metrics
import string

dir_path = os.path.dirname(os.path.realpath(__file__))

class POS_TAG:
    train_data_path = dir_path + "/sentence_train.csv"
    sentence_train  = pd.read_csv(train_data_path)

    tag_dictionary = defaultdict(list) # Dictionary for grouping words in a sentense
    pos_tags_path = dir_path + "/pos_tag_train.csv"
    pos_tags_train = pd.read_csv(pos_tags_path)

    # Make the sentence_id and token_id as indices for getting the tags efficiently
    sentence_train = sentence_train.set_index(['sentence_id', 'token_id'])
    pos_tags_train  = pos_tags_train.set_index(['sentence_id', 'token_id'])

    tag_list = []
    train_size = 0
    word_groups = defaultdict(list)
    sent_word_lists = []


    # Feature Engineering ideas adapted from https://medium.com/analytics-vidhya/pos-tagging-using-conditional-random-fields-92077e5eaa31
    # and https://sklearn-crfsuite.readthedocs.io/en/latest/tutorial.html#let-s-check-what-classifier-learned
    # and https://github.com/abcdw/crf-pos-tagger
    @staticmethod
    def word2features(sentence,index):
    ### sentence is of the form [w1,w2,w3,..], index is the position of the word in the sentence
        features =  {
            'bias': 1.0,
            'word.lower()': sentence[index].lower(),
            'is_first_capital':int(sentence[index][0].isupper()),
            'is_complete_capital': int(sentence[index].upper()==sentence[index]),
            'prev_word':'' if index==0 else sentence[index-1],
            'next_word':'' if index==len(sentence)-1 else sentence[index+1],
            'is_numeric':int(sentence[index].isdigit()),
            'prefix_2': sentence[index][:2],
            'prefix_3':sentence[index][:3],
            'prefix_4':sentence[index][:4],
            'suffix_1':sentence[index][-1],
            'suffix_2':sentence[index][-2:],
            'suffix_3':sentence[index][-3:],
            'suffix_4':sentence[index][-4:],
            'word_has_hyphen': 1 if '-' in sentence[index] else 0  
            }
        return features

    # This function is used to create the tag list for the sentences
    # The words for a sentence are grouped using a dictionary
    # The word-tag list are used for training the taggers available in nltk
    @staticmethod
    def create_tag_list():
        print("Creating tag list for training and testing")
        words = POS_TAG.sentence_train['before']
        pos = POS_TAG.pos_tags_train['pos']
        i = 0
        for s_id,t_id in POS_TAG.pos_tags_train.index:
            w = words[(s_id, t_id)]
            t = pos[(s_id, t_id)]
            if type(w) == str and type(t)== str:
                POS_TAG.tag_dictionary[s_id].append((w, t))
                # t = ""
                # try:
                #     t = pos[(s_id, t_id)]
                # except:
                #     if w.isdigit():
                #         t = 'CARDINAL'
                #     else:
                #         t = '.'
                #POS_TAG.tag_dictionary[s_id].append((w, t))
            # #     POS_TAG.word_groups[s_id].append(w)
            # i = i + 1
            # if i==3000000:
            #     break
        POS_TAG.tag_list = list(POS_TAG.tag_dictionary.values())
        #POS_TAG.sent_word_lists = list(POS_TAG.word_groups.values())
        POS_TAG.train_size = int(len(POS_TAG.tag_list) * .95)

    # This training method is used to train a part of speech tagging model using
    # methods in nltk
    @staticmethod
    def train_pos():
        print("Training pos tagger using N-gram tagger models")
        train_data = POS_TAG.tag_list[:POS_TAG.train_size]
        t0 = nltk.DefaultTagger('NN')
        t1 = nltk.UnigramTagger(train_data, backoff=t0)
        t2 = nltk.BigramTagger(train_data, backoff=t1)
        t3 = nltk.TrigramTagger(train_data, backoff=t2)
        output = open(dir_path +'/input_data/tgt.pkl', 'wb')
        dump(t3, output, -1)
        output.close()

    # This method is used to prepare the data for training a part of speech tagging model
    # using Conditional Random Fields
    # This adapted from https://medium.com/analytics-vidhya/pos-tagging-using-conditional-random-fields-92077e5eaa31
    # Extra checks added to replace nan values for words and tags
    @staticmethod
    def prepareData(tagged_sentences):
        print("Preparing data for training CRF")
        X,y=[],[]
        for sentences in tagged_sentences:
            words = []
            tags = []
            for word,tag in sentences:
                if type(tag) != str:
                    tags.append('NN')
                else:
                    tags.append(tag)
                    
                if type(word) != str:
                    words.append('Noun')
                else:
                    words.append(word)
            X.append([POS_TAG.word2features(words, index) for index in range(len(sentences))])
            y.append(tags)
        return X,y

    # Train a part of speech tagging model using 90% of the data, 10% is used for testing
    # Using Conditional Random Fields
    @staticmethod
    def train_pos2():
        print("Training POS using CRF")
        train_set, test_set = train_test_split(POS_TAG.tag_list,test_size=0.1, random_state=1000)
        X_train,y_train=POS_TAG.prepareData(train_set)
        crf = CRF(
            algorithm='lbfgs',
            c1=0.1,
            c2=0.1,
            max_iterations=40,
            all_possible_transitions=True,
            verbose= True
        )
        crf.fit(X_train, y_train)

        # Store the model into 
        output = open(dir_path +'/input_data/crf.pkl', 'wb')
        dump(crf, output)
        output.close()

    # Generate a test file taking sentences as a 2D array
    # Perfect for words and puntuations
    # Does not detected numbers and letters
    @staticmethod
    def generate_test_files(sentences):
        print("Create a test file")
        sentenses_array = []
        puncts = string.punctuation+" `` ''"

        f=open(dir_path+'/in_file_test_auto_gen.txt','w')
        header='sentence_id,token_id,class,before,after\n'
        f.write(header)
        for j in range(len(sentences)):
            sentence = sentences[j]
            sent_array = []
            for i in range(len(sentence)):
                if sentence[i] not in puncts:
                    word_details = str(j+1) +","+ str(i+1) + "," + '"PLAIN",'+'"'+sentence[i]+'",' + '"'+sentence[i]+'"\n'
                    f.write(word_details)
                else:
                    word_details = str(j+1) +","+ str(i+1) + "," + '"PUNCT",'+'"'+sentence[i]+'",' + '"'+sentence[i]+'"\n'
                    f.write(word_details)
            f.write("\n")
        f.close()
        

    @staticmethod
    def test_pos_tagger():
        print("Test POS tagger based on N-Gram")
        input = open(dir_path +'/input_data/tgt.pkl', 'rb')
        tagger = load(input)
        input.close()
        test_data = POS_TAG.tag_list[POS_TAG.train_size:]
        print("Testing on last 10% of data")
        print(tagger.evaluate(test_data))

        print("Testing on brown corpus")
        brown_tagged_sents = brown.tagged_sents(categories='news')
        print(tagger.evaluate(brown_tagged_sents[:1000]))
    
    @staticmethod
    def test_pos_tagger_2(X_test, y_test):
        print("Test POST tagger based on CRF")
        input = open(dir_path +'/input_data/crf.pkl', 'rb')
        crf = load(input)
        input.close()
        y_pred=crf.predict(X_test)
        labels = list(crf.classes_)
        print(metrics.flat_f1_score(y_test, y_pred,average='weighted',labels=labels))

    @staticmethod
    def predict_with_pos_tagger_2(X_test):
        input = open(dir_path +'/input_data/crf.pkl', 'rb')
        crf = load(input)
        input.close()
        y_pred=crf.predict(X_test)
        return y_pred

    @staticmethod
    def load_and_get_test_data():
        input_file_name = sys.argv[1]
        input_data  = pd.read_csv(dir_path + '/' + input_file_name)
        input_data = input_data[input_data['class']=='PLAIN']
        input_data  = input_data.set_index(['sentence_id', 'token_id'])
        plain_words = input_data['before']
        sentences = []
        index = input_data.index
        for s_id in np.unique(index.get_level_values(0)):# instead of set(index.get_level_values(0))
            sentences.append(list(plain_words.loc[s_id]))
        print(sentences[-1])
        return sentences, index
    
    @staticmethod
    def predict_and_write():
        print('<======== Predictiing and writing to '+sys.argv[2]+'  ========>')
        sentences, index = POS_TAG.load_and_get_test_data()
        X = []
        for sentence in sentences:
            words = []
            for word in sentence:
                if type(word) != str:
                    words.append('Noun')
                else:
                    words.append(word)
            X.append([POS_TAG.word2features(words, i) for i in range(len(sentence))])
        predictions = POS_TAG.predict_with_pos_tagger_2(X)
        f=open(dir_path + '/' +sys.argv[2],'w')
        header='sentence_id,token_id,class,before,after\n'
        f.write(header)
        i = 0
        print(predictions[-1])
        for tag_list in predictions:
            for tag in tag_list:
                f.write(str(index[i][0])+","+str(index[i][1]) +","+ tag+"\n")
                i +=1
            f.write("\n")
        f.close()
        #print("Length: "+str(i))

    @staticmethod
    def tests_crf():
        print('<========Testing on original dataset test sample========>')

        train_set, test_set = train_test_split(POS_TAG.tag_list,test_size=0.1, random_state=1000)
        X_test,y_test= POS_TAG.prepareData(test_set)
        POS_TAG.test_pos_tagger_2(X_test, y_test)

        print('<================Testing on brown corpus====================>')
        brown_tagged_sents = brown.tagged_sents(categories='news')
        train_set, test_set = train_test_split(brown_tagged_sents,test_size=0.5, random_state=1000)
        X_test,y_test= POS_TAG.prepareData(test_set)
        POS_TAG.test_pos_tagger_2(X_test, y_test)
        

    

# Comment these out for training, 
# model has been trained already and 
# saved in input_data folder, 
# for test they are loaded and used for predictions

# Comment this out for testing N-Gram combination tagger

# POS_TAG.create_tag_list()
# # # #POS_TAG.train_pos()  # THIS WILL TRAIN THE MODEL AGAIN, trained model is already saved, so no need
# POS_TAG.test_pos_tagger()

# Comment this out to test for CRF 
# #POS_TAG.train_pos2() # THIS WILL TRAIN THE MODEL AGAIN
# POS_TAG.create_tag_list()
# POS_TAG.tests_crf()
POS_TAG.predict_and_write()
#POS_TAG.load_and_get_test_data()

#<============ Generating a test file from sentences  as follows ===============> #
# sentences = [['The', 'Fulton', 'County', 'Grand', 'Jury', 'said', 'Friday', 'an', 'investigation', 'of', "Atlanta's", 'recent', 'primary', 'election', 'produced', 'no', 'evidence', 'that', 'any', 'irregularities', 'took', 'place'], ['The', 'jury', 'further', 'said', 'in', 'term-end', 'presentments', 'that', 'the', 'City', 'Executive', 'Committee', 'which', 'had', 'over-all', 'charge', 'of', 'the', 'election', 'deserves', 'the', 'praise', 'and', 'thanks', 'of', 'the', 'City', 'of', 'Atlanta', 'for', 'the', 'manner', 'in', 'which', 'the', 'election', 'was', 'conducted'], ['The', 'September-October', 'term', 'jury', 'had', 'been', 'charged', 'by', 'Fulton', 'Superior', 'Court', 'Judge', 'Durwood', 'Pye', 'to', 'investigate', 'reports', 'of', 'possible', 'irregularities', 'in', 'the', 'hard-fought', 'primary', 'which', 'was', 'won', 'by', 'Mayor-nominate', 'Ivan', 'Allen', 'Jr.'], ['Only', 'a', 'relative', 'handful', 'of', 'such', 'reports', 'was', 'received', 'the', 'jury', 'said', 'considering', 'the', 'widespread', 'interest', 'in', 'the', 'election', 'the', 'number', 'of', 'voters', 'and', 'the', 'size', 'of', 'this', 'city'], ['The', 'jury', 'said', 'it', 'did', 'find', 'that', 'many', 'of', "Georgia's", 'registration', 'and', 'election', 'laws', 'are', 'outmoded', 'or', 'inadequate', 'and', 'often', 'ambiguous'], ['It', 'recommended', 'that', 'Fulton', 'legislators', 'act', 'to', 'have', 'these', 'laws', 'studied', 'and', 'revised', 'to', 'the', 'end', 'of', 'modernizing', 'and', 'improving', 'them'], ['The', 'grand', 'jury', 'commented', 'on', 'a', 'number', 'of', 'other', 'topics', 'among', 'them', 'the', 'Atlanta', 'and', 'Fulton', 'County', 'purchasing', 'departments', 'which', 'it', 'said', 'are', 'well', 'operated', 'and', 'follow', 'generally', 'accepted', 'practices', 'which', 'inure', 'to', 'the', 'best', 'interest', 'of', 'both', 'governments'], ['Merger', 'proposed'], ['However', 'the', 'jury', 'said', 'it', 'believes', 'these', 'two', 'offices', 'should', 
# 'be', 'combined', 'to', 'achieve', 'greater', 'efficiency', 'and', 'reduce', 'the', 'cost', 'of', 'administration'], ['The', 'City', 'Purchasing', 'Department', 'the', 'jury', 'said', 'is', 'lacking', 'in', 'experienced', 'clerical', 'personnel', 'as', 'a', 'result', 'of', 'city', 'personnel', 'policies'], ['It', 'urged', 'that', 
# 'the', 'city', 'take', 'steps', 'to', 'remedy', 'this', 'problem'], ['Implementation', 'of', "Georgia's", 'automobile', 'title', 'law', 'was', 'also', 'recommended', 
# 'by', 'the', 'outgoing', 'jury'], ['It', 'urged', 'that', 'the', 'next', 'Legislature', 'provide', 'enabling', 'funds', 'and', 're-set', 'the', 'effective', 'date', 'so', 'that', 'an', 'orderly', 'implementation', 'of', 'the', 'law', 'may', 'be', 'effected'], ['The', 'grand', 'jury', 'took', 'a', 'swipe', 'at', 'the', 'State', 'Welfare', "Department's", 'handling', 'of', 'federal', 'funds', 'granted', 'for', 'child', 'welfare', 'services', 'in', 'foster', 'homes'], ['This', 'is', 'one', 'of', 
# 'the', 'major', 'items', 'in', 'the', 'Fulton', 'County', 'general', 'assistance', 'program', 'the', 'jury', 'said', 'but', 'the', 'State', 'Welfare', 'Department', 'has', 'seen', 'fit', 'to', 'distribute', 'these', 'funds', 'through', 'the', 'welfare', 'departments', 'of', 'all', 'the', 'counties', 'in', 'the', 'state', 'with', 'the', 'exception', 'of', 'Fulton', 'County', 'which', 'receives', 'none', 'of', 'this', 'money'], ['The', 'jurors', 'said', 'they', 'realize', 'a', 'proportionate', 'distribution', 'of', 'these', 'funds', 'might', 'disable', 'this', 'program', 'in', 'our', 'less', 'populous', 'counties'], ['Nevertheless', 'we', 'feel', 'that', 'in', 'the', 'future', 'Fulton', 'County', 'should', 'receive', 'some', 'portion', 'of', 'these', 'available', 'funds', 'the', 'jurors', 'said'], ['Failure', 'to', 'do', 'this', 'will', 'continue', 'to', 'place', 'a', 'disproportionate', 'burden', 'on', 'Fulton', 'taxpayers'], ['The', 'jury', 'also', 'commented', 'on', 'the', 'Fulton', "ordinary's", 'court', 'which', 'has', 'been', 'under', 'fire', 'for', 'its', 'practices', 'in', 'the', 'appointment', 'of', 'appraisers', 'guardians', 'and', 'administrators', 'and', 'the', 'awarding', 'of', 'fees', 'and', 'compensation'], ['Wards', 'protected'], ['The', 'jury', 'said', 'it', 'found', 'the', 'court', 'has', 'incorporated', 'into', 'its', 'operating', 'procedures', 'the', 'recommendations', 'of', 'two', 'previous', 'grand', 'juries', 'the', 'Atlanta', 'Bar', 'Association', 
# 'and', 'an', 'interim', 'citizens', 'committee'], ['These', 'actions', 'should', 'serve', 'to', 'protect', 'in', 'fact', 'and', 'in', 'effect', 'the', "court's", 'wards', 'from', 'undue', 'costs', 'and', 'its', 'appointed', 'and', 'elected', 'servants', 'from', 'unmeritorious', 'criticisms', 'the', 'jury', 'said'], ['Regarding', "Atlanta's", 'new', 'multi-million-dollar', 'airport', 'the', 'jury', 'recommended', 'that', 'when', 'the', 'new', 'management', 'takes', 'charge', 'Jan.', 'the', 'airport', 'be', 'operated', 'in', 'a', 'manner', 'that', 'will', 'eliminate', 'political', 'influences'], ['The', 'jury', 'did', 'not', 'elaborate', 'but', 'it', 'added', 'that', 'there', 'should', 'be', 'periodic', 'surveillance', 'of', 'the', 'pricing', 'practices', 'of', 'the', 'concessionaires', 'for', 'the', 'purpose', 'of', 'keeping', 'the', 'prices', 'reasonable'], ['Ask', 'jail', 'deputies'], ['On', 'other', 'matters', 'the', 'jury', 'recommended', 'that'], ['Four', 'additional', 'deputies', 'be', 'employed', 'at', 'the', 'Fulton', 'County', 'Jail', 'and', 'a', 'doctor', 'medical', 'intern', 'or', 'extern', 'be', 'employed', 'for', 'night', 'and', 'weekend', 'duty', 'at', 'the', 'jail'], ['Fulton', 'legislators', 'work', 'with', 'city', 'officials', 'to', 'pass', 'enabling', 'legislation', 'that', 'will', 'permit', 'the', 'establishment', 'of', 'a', 'fair', 'and', 'equitable', 'pension', 'plan', 'for', 'city', 'employes'], ['The', 'jury', 'praised', 'the', 'administration', 'and', 'operation', 'of', 'the', 'Atlanta', 'Police', 'Department', 'the', 'Fulton', 'Tax', "Commissioner's", 'Office', 'the', 'Bellwood', 'and', 'Alpharetta', 'prison', 'farms', 'Grady', 'Hospital', 'and', 'the', 'Fulton', 'Health', 'Department'], ['Mayor', 'William', 'Hartsfield', 'filed', 'suit', 'for', 'divorce', 'from', 'his', 
# 'wife', 'Pearl', 'Williams', 'Hartsfield', 'in', 'Fulton', 'Superior', 'Court', 'Friday'], ['His', 'petition', 'charged', 'mental', 'cruelty'], ['The', 'couple', 'was', 'married', 'Aug.', '2'], ['They', 'have', 'a', 'son', 'William', 'Berry', 'Jr.', 'and', 'a', 'daughter', 'Mrs.', 'Cheshire', 'of', 'Griffin'], ['Attorneys', 'for', 'the', 'mayor', 'said', 'that', 'an', 'amicable', 'property', 'settlement', 'has', 'been', 'agreed', 'upon'], ['The', 'petition', 'listed', 'the', "mayor's", 'occupation', 'as', 'attorney', 'and', 'his', 'age', 'as'], ['It', 'listed', 'his', "wife's", 'age', 'as', 'and', 'place', 'of', 'birth', 'as', 'Opelika', 'Ala.'], ['The', 'petition', 'said', 'that', 'the', 'couple', 'has', 'not', 'lived', 'together', 'as', 'man', 'and', 'wife', 'for', 'more', 'than', 'a', 'year'], ['The', 'Hartsfield', 
# 'home', 'is', 'at', 'E.', 'Pelham'], ['Henry', 'Bowden', 'was', 'listed', 'on', 'the', 'petition', 'as', 'the', "mayor's", 'attorney'], ['Hartsfield', 'has', 'been', 
# 'mayor', 'of', 'Atlanta', 'with', 'exception', 'of', 'one', 'brief', 'interlude', 'since'], ['His', 'political', 'career', 'goes', 'back', 'to', 'his', 'election', 'to', 'city', 'council', 'in', '1923'], ['The', "mayor's", 'present', 'term', 'of', 'office', 'expires', 'Jan.'], ['He', 'will', 'be', 'succeeded', 'by', 'Ivan', 'Allen', 'who', 'became', 'a', 'candidate', 'in', 'the', 'Sept.', 'primary', 'after', 'Mayor', 'Hartsfield', 'announced', 'that', 'he', 'would', 'not', 'run', 'for', 'reelection'], ['Georgia', 'Republicans', 'are', 'getting', 'strong', 'encouragement', 'to', 'enter', 'a', 'candidate', 'in', 'the', '1962', "governor's", 'race', 'a', 'top', 'official', 'said', 'Wednesday'], ['Robert', 'Snodgrass', 'state', 'GOP', 'chairman', 'said', 'a', 'meeting', 'held', 'Tuesday', 'night', 'in', 'Blue', 'Ridge', 'brought', 'enthusiastic', 'responses', 'from', 'the', 'audience'], ['State', 'Party', 'Chairman', 'James', 'Dorsey', 'added', 'that', 'enthusiasm', 'was', 'picking', 'up', 'for', 'a', 'state', 'rally', 'to', 'be', 'held', 'Sept.', 'in', 'Savannah', 'at', 'which', 'newly', 'elected', 'Texas', 'Sen.', 'John', 'Tower', 'will', 'be', 'the', 'featured', 'speaker'], ['In', 'the', 'Blue', 'Ridge', 'meeting', 'the', 'audience', 'was', 'warned', 'that', 'entering', 'a', 'candidate', 'for', 'governor', 'would', 'force', 'it', 'to', 'take', 'petitions', 'out', 'into', 'voting', 'precincts', 'to', 'obtain', 'the', 'signatures', 'of', 'registered', 'voters'], ['Despite', 'the', 'warning', 'there', 'was', 'a', 'unanimous', 'vote', 'to', 'enter', 'a', 'candidate', 'according', 'to', 'Republicans', 'who', 'attended'], ['When', 'the', 'crowd', 'was', 'asked', 'whether', 'it', 'wanted', 'to', 'wait', 'one', 'more', 'term', 'to', 'make', 'the', 'race', 'it', 'voted', 'no', '--', 'and', 'there', 'were', 'no', 'dissents']]
# POS_TAG.generate_test_files(sentences)
#<===============================================================================>


