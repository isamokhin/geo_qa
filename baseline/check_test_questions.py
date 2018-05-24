from geo_qa_project import Question, load_KB

with open('test_questions.txt', 'r') as f:
    tq = f.read().split('\n')
    
G = load_KB('geoproperties_uk.ttl.gz')

success = 0
for q_text in tq:
    q = Question(q_text)
    print(q.process_a_question(G))
    if "відповідь на запитання не знайшлась" in q.process_a_question(G):
        continue
    else:
        success += 1
        
print('Правильних відповідей', success)
print('Точність системи - ', success/len(tq))
    
