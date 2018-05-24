import sys
import pandas as pd
import re
import string
import pymorphy2
from nltk import bigrams
import rdflib
from itertools import groupby
import gzip
morph = pymorphy2.MorphAnalyzer(lang='uk')

class Question():
    
    """
    A class that gathers all our functions
    for parsing and answering the question
    """
    def __init__(self, q_text):
        self.q_text = q_text
    
    def __repr__(self):
        return 'Question: {q}'.format(q=self.q_text)
    
    def __str__(self):
        return 'Question: {q}'.format(q=self.q_text)
    
    PATTERNS = {
    "яка столиця": ["capital", "столиця"], 
    "яка форма правління": ["governmentType"], 
    "яка валюта": ["currency", "валюта"], 
    "яка площа": ["area", "площа"], 
    "яке населення": ["population", "populationEstimate", "населення"],
    "скільки людей": ["population", "populationEstimate", "населення"], 
    "де знаходиться": ["GET_COORDINATES"],
    "де розташовується": ["GET_COORDINATES"],
    "яка столиця": ["capital", "столиця"], 
    "який гімн": ["nationalAnthem"],
    "офіційні мови": ["officialLanguages"], 
    "державна мова": ["officialLanguages"],  
    "державні мови": ["officialLanguages"], 
    "офіційна мова": ["officialLanguages"], 
    "якими мовами говорять": ["officialLanguages"], 
    "якими мовами розмовляють": ["officialLanguages"], 
    "найбільше місто": ["largestCity"], 
    "який президент": ["leaderName1"], 
    "хто президент": ["leaderName1"],  
    "хто голова держави": ["leaderName1"], 
    "яка густота населення": ["populationDensity", "густота"], 
    "ВВП на душу": ["gdpPppPerCapita", "gdpNominalPerCapita"], 
    "який ВВП": ["gdpPpp", "gdpNominal"],
    "ВВП": ["gdpPpp", "gdpNominal"],
    "індекс розвитку": ["hdi"], 
    "код валюти": ["currencyCode", "кодВалюти"], 
    "домен": ["cctld"],
    "телефонний код": ["callingCode", "кодКраїни"], 
    "який код": ["callingCode", "кодКраїни"], 
    "коли засновано": ["establishedDate1", "засноване"], 
    "який часовий пояс": ["timeZone", "utcOffset"], 
    "у якій країні": ["країна", "country"], 
    "у якому регіоні": ["регіон", "region"], 
    "девіз": ["nationalMotto", "девіз"], 
    "яке населення агломерації": ["агломерація"], 
    "яка площа міста": ["area", "площа"], 
    "яка висота над рівнем моря": ["висотаНадРівнемМоря", "elevationM"], 
    "який поділ міста": ["поділМіста"], 
    "яка довжина": ["length", "totalLength", "partLength", "довжина"], 
    "глибина": ["depth", "maxDepth", "глибина", "найбільшаГлибина"], 
    "яка ширина": ["width", "ширина"], 
    "довжина берегу": ["довжинаБереговоїЛінії"], 
    "довжина берегової лінії": ["довжинаБереговоїЛінії"], 
    "об'єм": ["volume"], 
    "який регіон": ["регіон", "region"], 
    "яке розташування": ["location", "розташування"], 
    "яка гірська система": ["range"], 
    "яка висота": ["elevation", "elevationM"], 
    "яка площа басейну": ["площаБасейну", "areaWaterKm"], 
    "яке гирло": ["гирло"], 
    "куди впадає": ["басейн", "гирло"], 
    "який тип озера": ["тип"], 
    "який витік": ["витік"], 
    "звідки витікає": ["витік", "витікКоорд"], 
    "які прирічкові країни": ["прирічковіКраїни"], 
    "серередньорічний стік": ["стік"],
    "який стік": ["стік"],
    "назва країни": ["commonName"],
    }
    
    UNITS = {
    "area": "км²",
    "площа": "км²",
    "population_estimate": "людей",
    "GDP_PPP": "доларів",
    "GDP_PPP_per_capita": "доларів",
    "population_density": "людей на км²",
    "length": "кілометрів",
    "довжина": "кілометрів",
    "depth": "метрів",
    "width": "кілометрів",
    "volume": "кубічних метрів",
    "areaWaterKm": "км²",
    "площаБасейну": "км²"
    }
    
    EN_TO_UK = [('capital', 'столиця'),
    ('area', 'площа'),
    ('national_anthem', 'гімн'),
    ('national_motto', 'девіз'),
    ('largest_city', 'найбільше місто'),
    ('common_name', 'назва'),
    ('official_languages', 'офіційні мови'),
    ('officialLanguages', 'офіційні мови'),
    ('population_estimate', 'населення'),
    ('population_density', 'густота населення'),
    ('population', 'населення'),
    ('GDP_PPP', 'ВВП'),
    ('GDP_PPP_per_capita', 'ВВП на душу населення'),
    ('HDI', 'індекс людського розвитку'),
    ('government_type', 'форма правління'),
    ('established_date1', 'дата заснування'),
    ('currency', 'валюта'),
    ('currency_code', 'код валюти'),
    ('leader_name1', 'голова держави'),
    ('time_zone', 'часовий пояс'),
    ('cctld', 'домен'),
    ('calling_code', 'телефонний код'),
    ('elevationM', 'висота'),
    ('elevation', 'висота'),
    ('length', 'довжина'),
    ('partLength', 'довжина'),
    ('totalLength', 'повна довжина'),
    ('depth', 'глибина'),
    ('maxDepth', 'максимальна глибина'),
    ('width', 'ширина'),
    ('volume', "об'єм"),
    ('location', 'розташування'),
    ('region', 'регіон'),
    ('areaWaterKm', 'площа басейну'),
    ('range', 'гірська система')]
    EN_TO_UK = {k:v for k,v in EN_TO_UK}
    
    def gender_agree(self, w_parsed):
        """
        Inflect noun phrase with adjective the right way
        """
        gender = w_parsed.tag.gender
        if not gender:
            return w_parsed.normal_form
        w = w_parsed.inflect({gender, 'nomn'}).word
        return w
    
    def get_entity(self, q_text):
        """
        Look for (capitalized) entities in q_text
        """
        words = q_text.split()
        phrase = []
        for i, w in enumerate(words[1:]):
            if w[0] == w[0].upper():
                w_parsed = morph.parse(w.strip(' ?'))[0]
                if 'ADJF' in w_parsed.tag:
                    phrase.append(self.gender_agree(w_parsed).title())
                    phrase.append(morph.parse
                                  (words[i+2].strip(' ?'))[0].normal_form)
                    return ' '.join(phrase)
                elif 'NOUN' in w_parsed.tag:
                    return w_parsed.normal_form.title()
                elif 'UNKN' in w_parsed.tag:
                    return w_parsed.normal_form.title()
                else:
                    continue
        return None
    
    def pattern_match(self, q_text):
        """
        Match a question against common patterns
        """
        answer_list = []
        for k in self.PATTERNS.keys():
            if k in q_text.lower():
                if "GET_COORDINATES" in self.PATTERNS[k]:
                    if not self.get_entity(q_text):
                        continue
                    else:
                        answer_list.append(('GET_COORDINATES', self.get_entity(q_text)))
                else:
                    if not self.get_entity(q_text):
                        continue
                    for prop in self.PATTERNS[k]:
                        ent = self.get_entity(q_text)
                        answer_list.append((prop, ent))
        return answer_list
    
    def map_question(self, q_parsed):
        """
        Take list of tuples q_parsed and transform all that is relevant
        into properties or objects for subsequent query
        """
        focus_list = [e[0] for e in q_parsed if e[1]=='focus']
        verb_list = [e[0] for e in q_parsed if e[1]=='verb']
        entity_list = [e[0] for e in q_parsed if e[1]=='entity']
        if len(focus_list) == 1:
            focus_key_list = focus_list
        else:
            focus_key_list = []
            focus_key_list.append("".join(focus_key_list))
            focus_key_list += focus_list
        if len(verb_list) == 1:
            pass
        if len(entity_list) == 1:
            ent = entity_list[0].title()
        else:
            return None
        res = []
        for f in focus_key_list:
            if not f:
                continue
            res.append((f, ent))
        return res
    
    def parse_q(self, q_text):
        """
        A function to parse a question text using bigrams,
        looking for focus of the question and entities whose
        properties are the focus
        """
        matched = self.pattern_match(q_text)
        if matched:
            return matched
        q_words = ["що", "коли", "скільки", "де", "хто"]
        non_nomn = ['gent', 'loct', 'datv', 'accs']
        words = q_text.strip('.,?" -').split()
        bgrams = bigrams(words)
        parsed = []
        phrase_list = []
        for g in bgrams:
            w1, w2 = g
            w1_parsed = morph.parse(w1)[0]
            w2_parsed = morph.parse(w2)[0]
            w1_tag = w1_parsed.tag
            w1_lemma = w1_parsed.normal_form
            w2_tag = w2_parsed.tag
            w2_lemma = w2_parsed.normal_form
            if w1_lemma == 'який':
                parsed.append((w1_lemma, 'which'))
                if 'NOUN' in w2_tag:
                    parsed.append((w2_lemma, 'focus'))
            elif w1_lemma in q_words:
                parsed.append((w1_lemma, 'q_word'))
                if 'VERB' in w2_tag:
                    parsed.append((w2_lemma, 'verb'))
                elif 'NOUN' in w2_tag:
                    parsed.append((w2_lemma, 'focus'))
            elif 'NOUN' in w1_tag:
                parsed.append((w1_lemma, 'focus'))
                if ('NOUN' in w2_tag and (any(vidm in w2_tag for vidm in non_nomn)
                                         or w2[0].upper() == w2[0])):
                    parsed.append((w2_lemma, 'entity'))
                elif 'NOUN' in w2_tag and 'nomn' in w2_tag:
                    parsed.append((w2_lemma, 'focus'))
                elif 'VERB' in w2_tag:
                    parsed.append((w2_lemma, 'verb'))
                elif 'ADJF' in w2_tag or 'COMP' in w2_tag:
                    phrase_list.append(self.gender_agree(w2_parsed))
            elif 'PREP' in w1_tag:
                if 'NOUN' in w2_tag:
                    parsed.append((w2_lemma, 'entity'))
            elif 'ADJF' in w1_tag or 'COMP' in w1_tag:
                if 'NOUN' in w2_tag:
                    if phrase_list:
                        phrase = ' '.join(phrase_list)
                        phrase_list = []
                        parsed.append((phrase + ' ' + w2_lemma, 'entity'))
                    elif w2[0] == w2[0].upper():
                        parsed.append((self.gender_agree(w1_parsed) + ' ' + w2_lemma, 'entity'))
                    else:
                        if w1[0] == w1[0].upper():
                            parsed.append((self.gender_agree(w1_parsed) + ' ' + w2_lemma, 'entity'))
                        else:
                            parsed.append((self.gender_agree(w1_parsed) + ' ' + w2_lemma, 'focus'))
                elif 'ADJF' in w2_tag or 'COMP' in w2_tag:
                    phrase_list.append(self.gender_agree(w1_parsed))
                    phrase_list.append(self.gender_agree(w2_parsed))
            elif 'VERB' in w1_tag and 'NOUN' in w2_tag:
                parsed.append((w1_lemma, 'verb'))
                if w2[0] == w2[0].upper():
                    parsed.append((w2_lemma, 'entity'))
                else:
                    parsed.append((w2_lemma, 'focus'))
            else:
                if 'NOUN' in w2_tag and (w2[0] == w2[0].upper()):
                    parsed.append((w2_lemma, 'entity'))
                continue
        parsed = [p[0] for p in groupby(parsed)]
        return self.map_question(parsed)
    
    def run_query(self, g, prop, ent):
        """
        Run a single query and get an answer
        """
        template = """
        PREFIX prop: <http://uk.dbpedia.org/property/>
        PREFIX resource: <http://uk.dbpedia.org/resource/>
        SELECT DISTINCT ?prop ?obj
           WHERE {{
               resource:{entity} prop:{prop} ?obj
           }}"""
        if len(ent.split()) > 1:
            ent = ''.join([w.title() for w in ent.split()])
        if len(prop.split()) > 1:
            prop = ''.join(prop.split())
        q = template.format(entity=ent, prop=prop)
        qres = g.query(q)
        q_answers = [str(r[1]).replace('http://uk.dbpedia.org/resource/', '') for r in qres]
        if q_answers:
            return q_answers
        return None
    
    def get_coordinates(self, g, ent):
        """
        A special query for getting the coordinates
        """
        latd_list = ['latDeg', 'latd', 'широта']
        lond_list = ['lonDeg', 'lond', 'довгота']
        coord = []
        for prop in latd_list:
            latd_a = self.run_query(g, prop, ent)
            if latd_a:
                coord.append(latd_a[0])
                break
        for prop in lond_list:
            lond_a = self.run_query(g, prop, ent)
            if lond_a:
                coord.append(lond_a[0])
                break
        if len(coord) == 2:
            return coord
        else:
            return None
        
    def run_queries(self, g, prop_list):
        """
        Run query with focus and entity taken from list
        on a rdflib graph g
        """
        answers = []
        if not prop_list:
            return None
        for prop, entity in prop_list:
            if prop == 'GET_COORDINATES':
                coord = self.get_coordinates(g, entity)
                if coord:
                    answers = [
                        ('latd', entity, coord[0]),
                        ('lond', entity, coord[1])
                    ]
                    return answers
            q_answers = self.run_query(g, prop, entity)
            if not q_answers:
                continue
            for a in q_answers:
                answers.append((prop, entity, a))
        return answers
    
    def make_coord_answer(self, answer_list):
        """
        Construct an answer, but for coordinates.
        """
        template = 'Координати {ent} - {lat} широти і {lon} довготи'
        ent = answer_list[0][1]
        ent = morph.parse(ent)[0].inflect({'gent'}).word.title()
        lat = answer_list[0][2]
        lon = answer_list[1][2]
        return template.format(ent=ent, lat=lat, lon=lon)
    
    def construct_answer(self, answer_tuple, en_to_uk, units_dict):
        """
        Construct a Ukrainian-language answer to the question
        """
        template = "{focus} {entity} - {a}{units}"
        en_focus, entity, a = answer_tuple
        if en_focus not in en_to_uk:
            focus = en_focus
        else:
            focus = en_to_uk[en_focus]
        focus = focus[0].upper() + focus[1:]
        entity = morph.parse(entity)[0].inflect({'gent'}).word.title()
        if en_focus in units_dict:
            units = ' ' + units_dict[en_focus]
        else:
            units = ''
        answer = template.format(focus=focus, entity=entity, 
                                 a=a.replace('_', ' '), units=units)
        return answer
    
    def provide_answers(self, answer_list, en_to_uk, units_dict):
        """
        Return a list of answers in Ukranian language,
        if there are any
        """
        if not answer_list:
            return None
        if (len(answer_list) == 2) and (answer_list[0][0]=='latd'):
            return [self.make_coord_answer(answer_list)]
        res = [self.construct_answer(a, 
                en_to_uk, units_dict)
                for a in answer_list]
        return res
    
    def show_an_answer(self, answers):
        """
        The text that is showed to the user
        """
        text = ''
        if answers:
            first = answers[0]
            text += 'Відповідь на запитання:\n'
            text += first+'\n'
            if len(answers) > 1:
                text += '\nТакож можливі такі відповіді:\n'
                for a in answers[1:]:
                    text += a+'\n'
        else:
            text += 'Вибачте, відповідь на запитання не знайшлась!\n'
        return text
    
    def process_a_question(self, g):
        """
        A pipeline to process the question
        and return the answer
        """
        parsed_q = self.parse_q(self.q_text)
        queried_answers = self.run_queries(g, parsed_q)
        answers = self.provide_answers(queried_answers, self.EN_TO_UK, self.UNITS)
        result = self.show_an_answer(answers)
        return result
    
def load_KB(fname):
    """
    Loading the knowledge base
    """
    g = rdflib.Graph()

    with gzip.open(fname, 'r') as f:
        print('Зачекайте, будь ласка, триває завантаження бази знань!')
        g.parse(f, format='n3')
        print('\n...Базу завантажено!')
    
    return g


if __name__ == '__main__':
    G = load_KB('geoproperties_uk.ttl.gz')
    sys.stdout.write('Будь ласка, ставте географічні запитання.\n')
    sys.stdout.write('(щоб завершити роботу, введіть exit)\n\n')
    while True:
        q_text = input()
        if q_text == 'exit':
            sys.stdout.write('Завершення роботи.\n')
            break
        q = Question(q_text)
        sys.stdout.write(q.process_a_question(G))
        