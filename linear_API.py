from flask import Flask, request
#to save and loade ml models
import pickle
import json


#create variable  as name
app=Flask(__name__)


#decorator = to handle user request and response 
#check the value url name /usecase and match / map the function 

@app.route('/usecase_name',methods=['GET'])

def usecase():
    return ('prediction')

#http://127.0.0.1:8090/kiru
#user give value  {score}=2500----input
#json exchange data b/w client and server

@app.route('/kiru',methods=['POST'])
def kiru():
    request_data=request.json['score']
    

    pickel_m="C:/API/app/linear_reg.pkl"  
    with open(pickel_m,'rb') as file:
      lode_model=pickle.load(file)

    result=lode_model.predict([[request_data]])[0][0]#[[4.1]]
    return json.dumps({'score':result}) #return as json string  '{'score':result}'



if __name__=='__main__':
    app.run(port='8090')



