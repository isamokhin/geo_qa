import sys
from qa_code import QuestionParser

if __name__ == '__main__':
    qp1 = QuestionParser()
    qp2 = QuestionParser(model='logistic')
    sys.stdout.write("Ця програма може відповісти на деякі запитання про різноманітні географічні факти.\n")
    sys.stdout.write('Будь ласка, ставте запитання!\n')
    sys.stdout.write("(щоб завершити роботу, введіть 'exit' або 'геть')\n\n")
    while True:
        q_text = input()
        if q_text.strip() == 'exit' or q_text.strip() == 'геть':
            sys.stdout.write('Завершення роботи.\n')
            break
        answer = qp1.answer_the_question(q_text)
        if not answer or 'не знайшлась' in answer:
            answer = qp2.answer_the_question(q_text)
        sys.stdout.write(answer + '\n\n')