from flask import Flask, Response,render_template,redirect,request,url_for,flash
import cv2
import time
import random
import os
from flask_sqlalchemy import SQLAlchemy
import main

from detect import Camera 
app = Flask(__name__)

app.config['SQLALCHEMY_DATABASE_URI'] = 'mysql://root:@localhost/my_data'

db = SQLAlchemy(app)
# class Information(db.Model):
# 	Sno = db.Column(db.Integer, primary_key=True,nullable=True)
# 	First_Name = db.Column(db.String(80), unique=False,nullable=False)
# 	Last_name = db.Column(db.String(120), unique=False,nullable=False)
# 	City=db.Column(db.String(120), unique=False,nullable=False)
# 	State=db.Column(db.String(120),nullable=False)
# 	zip1=db.Column(db.Integer,nullable=False)
class user_registration(db.Model):
    Sno = db.Column(db.Integer, primary_key=True,nullable=True)
    User_Name = db.Column(db.String(50), unique=False,nullable=False)
    Email = db.Column(db.String(100), unique=False,nullable=False)
    Password=db.Column(db.String(100), unique=False,nullable=False)
	    
n=1
@app.route('/',methods=['GET', 'POST'])
def api():
   global system
   if request.method=="POST":
        #   First_Name=request.form.get('First')
        #   Last_name=request.form.get('Last') 
        #   City=request.form.get('City')
        #   State=request.form.get('State')
        #   zip1=request.form.get('zipi')
          User_Name=request.form.get('Username')
          Email=request.form.get('email')
          Password=request.form.get('password')
          pwd=request.form.get('password1')
          if(pwd!=Password):
              main.fun("PASWORD DO NOT MATCH")
              return regError("Passwords do not match please enter password again")
          for i in range(0,3):
            if i==0:
                main.fun("NOW YOU WILL LISTEN A FOUR DIGIT NUMBER .YOU HAVE TO GESTURE ACCORDINGLY TO SHOW YOUR IDENTITY u will get only 7 seconds for a particular number")
            else:
                main.fun("You get another chance to verify your identity Again You Will listen a 4 digit number to gesture")
            list1=[0,0,0,0]
            for i in range(0,4):
                num=random.randint(0,5)
                main.fun(num)
                
                list1[i]=num
            print(list1)    
            main.fun("Show the First Number ")
            s=Camera(list1)
            if s==False:
                    main.fun("Wrong Capatcha ")
            else:
                    #entry=Information(First_Name=First_Name,Last_name=Last_name,City=City,State=State,zip1=zip1)
                    entry=user_registration(User_Name=User_Name,Email=Email,Password=Password)
                    db.session.add(entry)
                    db.session.commit()
                    main.fun("Form Submitted ")
                    break
                 
   return (render_template('index2.html'))
def regError(message):
    return render_template('index2.html')       
if __name__ == "__main__":
    app.run()