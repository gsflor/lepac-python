import paho.mqtt.client as mqttClient
import time
import sys

def on_connect(client, userdata, flags, rc):


    if rc == 0:

        print("Connected to broker")

        global Connected                #Use global variable
        Connected = True                #Signal connection

    else:

        print("Connection failed")

def on_message(client, userdata, message):
   
    with open('test1.txt','a+') as f:
         #f.write("Message received: "  + message.payload + "\n")
         #f.write(message.payload)
         f.write("Message received: " + str(message.payload) + "\n")

Connected = False   #global variable for the state of the connection

broker_address= "172.20.48.128"  #Broker address
port = 1883                         #Broker port
 #user =                    #Connection username
 #password =            #Connection password

client = mqttClient.Client("Python")               #create new instance
#client.username_pw_set(user, password=password)    #set username and password
client.on_connect= on_connect                      #attach function to callback
client.on_message= on_message                      #attach function to callback
client.connect(broker_address,port,60) #connect
client.subscribe("esp/testX1",1) #subscribe
#client.subscribe("casa/temperatura",1) #subscribe
#client.subscribe("casa/umidade",1) #subscribe
client.loop_forever() #then keep listening forever
