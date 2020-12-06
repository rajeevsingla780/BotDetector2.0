import pyttsx3 
  
def fun(t):
    engine = pyttsx3.init() 
    
    # testing 
    engine.setProperty('rate',120)
    engine.setProperty('volume',0.9)
    engine.say(str(t))
   
    engine.runAndWait()
