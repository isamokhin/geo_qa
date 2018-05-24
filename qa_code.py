import random
import re
import pickle
import string
import pymorphy2
morph = pymorphy2.MorphAnalyzer(lang='uk')
from models.ukr_stemmer3 import UkrainianStemmer
from tokenize_uk import tokenize_words, tokenize_sents
from models.tagger import PerceptronTagger # POS tagger
from models.qa_perceptron import AveragedPerceptron # model for parsing question
from sklearn.externals import joblib
from difflib import get_close_matches
import nltk
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline, FeatureUnion

def lemmatize_phrase(phrase):
    """
    Also we can stem instead of lemmatizing...
    """
    words = fix_hyphens(tokenize_words(phrase))
    if len(words) == 1:
        wparsed = morph.parse(phrase)[0]
        if not wparsed:
            return phrase
        return wparsed.normal_form
    else:
        new_phrase = ''
        for w in words:
            new_phrase += morph.parse(w)[0].normal_form + ' '
        return new_phrase.strip()

def lemmatize_ent(ent):
    wparsed = morph.parse(ent)[0]
    if not wparsed:
        return ent
    if ent.istitle():
        return wparsed.normal_form.title()
    else:
        return wparsed.normal_form
    
def fix_hyphens(sent):
    """
    sent is tokenized with tokenize_uk
    """
    new_sent = []
    i = 0
    while i < len(sent):
        w = sent[i]
        if w == '—' or w == '-':
            new_sent.pop()
            new_word = sent[i-1]+'-'+sent[i+1]
            new_sent.append(new_word)
            i += 1
        else:
            new_sent.append(w)
        i += 1
    return new_sent

def gender_agree(w_parsed):
    """
    Inflect noun phrase with adjective the right way
    """
    gender = w_parsed.tag.gender
    if not gender:
        return w_parsed.normal_form
    w = w_parsed.inflect({gender, 'nomn'}).word
    return w
    
def get_matches(ent, all_ents):
    matches = get_close_matches(ent, all_ents)
    if not matches:
        for entry in all_ents:
            if ent.lower() in entry.lower():
                return entry
    return matches[0]

def deparentize(k):
    res = re.sub(r'\(.*\)', '', k)
    res = re.sub(r'\[.*\]', '', res).strip()
    return res

def ent_phrase(ner_recognized):
    """
    ner_recognized is a list of tokens and labels.
    """
    ent_phrases = []
    current_phrase = ''
    for token, label in ner_recognized:
        if label == 'LOC':
            current_phrase += lemmatize_ent(token) + ' '
        elif (len(current_phrase) > 0) and label != 'LOC':
            ent_phrases.append(current_phrase.strip())
            current_phrase = ''
        else:
            continue
    ent_phrases.append(current_phrase.strip())
    if not [e for e in ent_phrases if e != '']:
        return None
    return ent_phrases[0]

class QuestionParser():
    
    def __init__(self, model = 'perceptron'):
        self.model_name = model
        self.pos_tagger = PerceptronTagger()
        try:
            with open('data/obj_dict.pkl', 'rb') as od:
                self.obj_dict = pickle.load(od)
            with open('data/units.pkl', 'rb') as ud:
                self.unit_dict = pickle.load(ud)
            self.ner_model = joblib.load('models/NER_model.pkl')
            if model == 'perceptron':
                self.qa_model = AveragedPerceptron()
                self.load_perc('models/qa_model.pkl')
            elif model == 'logistic':
                self.qa_model = joblib.load('models/qa_skl_model.pkl')
            elif model == None:
                print('Please train the model for QA.')
        except:
            print('Будь ласка, переконайтесь, що в директорії models є всі потрібні файли.')
            print('Без них програма не працюватиме.')
        self.lem_dict = [morph.parse(ent.split()[0])[0].normal_form 
                         for ent in self.obj_dict.keys()]
        self.disamb_dict = self.build_disamb_dict()
        self.NO_ANSWER = "Відповідь не знайшлась. Можливо, бракує даних або програма не знайшла географічний об'єкт."
    
    def load_perc(self, loc):
        weights, classes = pickle.load(open(loc, 'rb'))
        self.qa_model.weights = weights
        self.qa_model.classes = classes
        return None
    
    def build_disamb_dict(self):
        from collections import Counter
        depar_keys = [deparentize(k) for k in self.obj_dict]
        disamb_dict = dict()
        duplicates = [item for item, count in Counter(depar_keys).items() if count > 1]
        for k in depar_keys:
            disamb_dict[k] = []
        for k in self.obj_dict:
            disamb_dict[deparentize(k)].append(k)
        return disamb_dict
    
    def get_entity_pymorphy(self, q_text):
        """
        Look for (capitalized) entities in q_text.
        For this specific application pymorphy2 tagging is enough.
        """
        forbidden = ['ВВП', 'HDI', 'ISO', 'ООН', 'UN', 'UTC', 
                     'Utc-Поправка', 'Utc-Поправка']
        words = fix_hyphens(tokenize_words(q_text))
        phrase = []
        for i, w in enumerate(words[1:]):
            if w in forbidden:
                continue
            if w[0] == w[0].upper():
                w_parsed = morph.parse(w.strip(' ?'))[0]
                w_lemma = w_parsed.normal_form
                if w_lemma in self.lem_dict:
                    if 'ADJF' in w_parsed.tag:
                        phrase.append(gender_agree(w_parsed).title())
                        phrase.append(morph.parse
                                      (words[i+2].strip(' ?'))[0].normal_form)
                        return ' '.join(phrase).title()
                    elif 'NOUN' in w_parsed.tag:
                        return w_lemma.title()
                    elif 'UNKN' in w_parsed.tag:
                        return w_lemma.title()
                matches = get_close_matches(w_lemma.title(), list(self.disamb_dict.keys()))
                if matches:
                    return matches[0]
                else:
                    continue
        return None
    
    def _get_ner_features(self, word, prev_word, next_word):
        features = {
            'word': word,
            'word_stem': UkrainianStemmer(word).stem_word(),
            'prev_word': prev_word,
            'next_word': next_word,
            'prev_stem': UkrainianStemmer(prev_word).stem_word(),
            'next_stem': UkrainianStemmer(next_word).stem_word(),
            'is_uppercase': word.title() == word,
            'is_after_punct': prev_word in string.punctuation,
            'is_after_uppercase': prev_word.title() == prev_word,
            'pos': self.pos_tagger.tag(' '.join([prev_word, word, next_word]))[1][1]
        }
        return features
    
    def ner_recognize(self, sent):
        sent = sent.strip(string.punctuation)
        tokens = fix_hyphens(tokenize_words(sent))
        feats = []
        for (i, t) in enumerate(tokens):
            if i == 0:
                prev_word = '.'
            else:
                prev_word = tokens[i-1]
            if i == len(tokens)-1:
                next_word = '.'
            else:
                next_word = tokens[i+1]
            feats.append(self._get_ner_features(t, prev_word, next_word))
        labels = self.ner_model.predict(feats)
        first_res = list(zip(tokens, labels))
        res = []
        for token, label in first_res:
            if token in ['море', "моря", "озеро", "озера", "океан", "океану"]:
                res.append((token, 'LOC'))
            else:
                res.append((token, label))
        return res
    
    def get_entity(self, q):
        all_ents = self.disamb_dict.keys()
        ner_recognized = self.ner_recognize(q)
        to_match = ent_phrase(ner_recognized)
        if not to_match:
            return self.get_entity_pymorphy(q)
        match = get_matches(to_match, all_ents)
        if not match:
            return None
        return match
    
    def parse_question(self, q):
        ent = self.get_entity(q)
        if not ent:
            #print("Не вдалось знайти географічний об'єкт!")
            return None, None
        lem_sent = lemmatize_phrase(q)
        lem_ent = lemmatize_phrase(ent)
        new_sent = lem_sent.replace(lem_ent, '').replace('  ', ' ')
        new_sent = new_sent.replace('який', '')
        return ent, new_sent.strip()
    
    def get_features(self, q):
        try:
            ent, sent = self.parse_question(q)
        except:
            return None
        if not ent:
            return None
        if self.model_name == 'perceptron':
            return self.get_features_perc(ent, sent)
        elif self.model_name == 'logistic':
            return self.get_features_sklearn(ent, sent)
    
    def get_features_perc(self, ent, sent):
        """
        Given question, get features from it.
        """
        features = {}
        words = fix_hyphens(tokenize_words(sent))
        for i, w in enumerate(words):
            features['word_{i}={w}'.format(i=i, w=w)] = 1
        features['words'] = [('w={w}'.format(w=w), 1) for w in words]
        bigrams = ['_'.join(b) for b in nltk.bigrams(words)]
        features['bigrams'] = [('bg={bg}'.format(bg=bg), 1) for bg in bigrams]
        n = 3
        char_trigrams = [sent[i:i+n] for i in range(len(sent)-n+1)]
        features['trigrams'] = [('t={t}'.format(t=t), 1) for t in char_trigrams]
        return ent, features
    
    def get_features_sklearn(self, ent, sent):
        features = dict()
        words = fix_hyphens(tokenize_words(sent))
        bigrams = ['_'.join(b) for b in nltk.bigrams(words)]
        n = 3
        char_trigrams = [sent[i:i+n] for i in range(len(sent)-n+1)]
        for w in words:
            features[w] = 1
        for b in bigrams:
            features[b] = 1
        for c in char_trigrams:
            features[c] = 1
        return ent, features
    
    def train(self, train_df, n_iter=5):
        if self.model_name == 'perceptron':
            self.train_perc(train_df, n_iter)
        elif self.model_name == 'logistic':
            self.train_sklearn(train_df)
    
    def train_sklearn(self, train_df):
        features = []
        labels = []
        for i, row in train_df.iterrows():
            q = row['Q']
            k = row['K']
            try:
                ent, feats = self.get_features(q)
            except:
                continue
            labels.append(k)
            features.append(feats)
        model = Pipeline([
                    ('vec', DictVectorizer()),
                    ('clf', LogisticRegression(penalty='l1'))
        ])
        model.fit(features, labels)
        joblib.dump(model, 'models/qa_skl_model.pkl')
        self.qa_model = model
    
    def train_perc(self, train_df, n_iter=5):
        """
        train_df contains columns Q and A
        """
        allkeys = []
        for c in obj_dict:
            for k in obj_dict[c]:
                allkeys.append(k)
        allkeys = set(allkeys)
        self.qa_model.classes = allkeys
        for iteration in range(n_iter):
            print('Training iteration number', iteration+1)
            train_df = train_df.sample(len(train_df))
            for i, row in train_df.iterrows():
                q = row['Q']
                k = row['K']
                true_keys = []
                try:
                    ent, feats = self.get_features(q)
                except:
                    continue
                guess = self.qa_model.predict(feats)
                self.qa_model.update(k, guess, feats)
        self.qa_model.average_weights()
        self.qa_model.save('models/qa_model.pkl')
    
    def provide_gen_case(self, ent):
        if 'Назва в родовому відмінку' in self.obj_dict[ent].keys():
            if not 'нема інформації' in self.obj_dict[ent]['Назва в родовому відмінку']:
                return self.obj_dict[ent]['Назва в родовому відмінку']
        ent = deparentize(ent)
        if len(ent.split()) == 1:
            w_parsed = morph.parse(ent)[0]
            return w_parsed.inflect({'gent'}).word.title()
        else:
            res = ''
            for w in ent.split():
                w_parsed = morph.parse(w)[0]
                gender = w_parsed.tag.gender
                if not gender:
                    res += w
                elif w.startswith('мор'):
                    res += 'моря'
                else:
                    res += w_parsed.inflect({gender, 'gent'}).word + ' '
            res = res[0].upper() + res[1:]
            return res.strip()
    
    def answer_text(self, answers):
        answer_template = '{pred} {ent} - {a} {units}'
        answer_texts = []
        for ent, cl, a in answers:
            a = str(a)
            detail = ''
            if len(answers) > 1:
                if 'уточнення' in self.obj_dict[ent]:
                    detail += '{0}, '.format(self.obj_dict[ent]['уточнення'])
                if '[' in ent:
                    detail += re.search(r'\[(.*)\]', ent).group(1)
                detail = '('+detail.strip(' ,')+')'
            gen_name = self.provide_gen_case(ent) + ' ' + detail
            gen_name = gen_name.strip()
            units = self.unit_dict.get(cl)
            if a == '' or ('нема інформації' in a):
                units = ''
            if not units:
                units = ''
            res = answer_template.format(pred=cl, 
                                         ent=gen_name,
                                         a=a,
                                         units=units).strip()
            answer_texts.append(res)
        if len(answer_texts) > 1:
            a_text = '\n'.join(answer_texts)
        else:
            a_text = answer_texts[0]
        return a_text
    
    def find_answers(self, ent, pred_classes):
        """
        """
        answers = []
        candidates = self.disamb_dict[ent]
        for c in candidates:
            for cl in pred_classes:
                a = self.obj_dict[c].get(cl)
                if a:
                    answers.append((c, cl, a))
                    break
        return answers
    
    def answer_the_question(self, q):
        q = q.replace('ґ', "г")
        try:
            ent, feats = self.get_features(q)
        except:
            return self.NO_ANSWER
        if not ent:
            return self.NO_ANSWER
        if self.model_name == 'perceptron':
            pred_classes = self.qa_model.get_scored_classes(feats)[:3]
        elif self.model_name == 'logistic':
            pred_probs = self.qa_model.predict_proba([feats])[0]
            prob_per_class = dict(zip(self.qa_model.classes_, pred_probs))
            cl_by_prob = list(map(lambda x: x[0], 
                                  sorted(zip(self.qa_model.classes_, prob_per_class), 
                                         key=lambda x: x[1], reverse=True)))
            pred_classes = cl_by_prob[:3]
        answers = self.find_answers(ent, pred_classes)
        if not answers:
            return self.NO_ANSWER
        return self.answer_text(answers)
