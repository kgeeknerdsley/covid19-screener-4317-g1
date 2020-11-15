from test_audio import speech_recog, record_audio
import time

question1 = ('Are you experiencing any of these symptoms?' + '\n' +
                '-Bluish lips or face' + '\n' +
                '-Severe and constant pain or pressure on chest' + '\n' +
                '-Extreme difficulty breathing' + '\n' +
                '-Disorientation' + '\n' +
                '-Signs of low blood pressure'
)
question2 = 'Are you currently feeling sick?'
question3 = ('In the last two weeks, did you have close contact with someone ' + '\n' +
                'with symptoms of COVID-19 or diagnosed with COVID-19?'
)

questions = [question1, question2, question3]
count = 0

print("Welcome to the Coronavirus Self-Checker!\n")
time.sleep(2)

for question in questions:
    print(question)
    time.sleep(5)
    print("Answer question in :")
    for second in range(3, 0,-1):
        time.sleep(1)
        print(second)
    record_audio()
    word = speech_recog()
    print("The system recognized the word to be: " + word)
    while (word == "unknown"):
        print("Please repeat your answer (yes/no) in: ")
        for second in range(3, 0,-1):
            time.sleep(1)
            print(second)
        record_audio()
        word = speech_recog()
        print("The system recognized the word to be: " + word)
    if (word == 'yes'):
        print("\nPlease go home and keep others safe.\nYou may be eligible for a COVID-19 Testing.")
        break
    count = count + 1
    if (count == 3):
        print("\nYou may come inside.")