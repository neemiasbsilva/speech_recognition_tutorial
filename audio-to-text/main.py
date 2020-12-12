import speech_recognition as sr

r = sr.Recognizer()

with sr.Microphone() as source:
    print("Speech: ")
    audio = r.listen(source)
    try:
        text = r.recognize_google(audio)
        print("You said: {}".format(text))
    except:
        print("Sorry could not recognize your voice")
