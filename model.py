from tensorflow.keras.models import model_from_json
from tensorflow.python.keras.backend import set_session
import numpy as np

import tensorflow as tf

config = tf.compat.v1.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.15
session = tf.compat.v1.Session(config=config)
set_session(session)


class FacialExpressionModel(object):

    EMOTIONS_LIST = ["Angry", "Disgust",
                     "Fear", "Happy",
                     "Neutral", "Sad",
                     "Surprise"]

    def __init__(self, model_json_file, model_weights_file):
        # load model from JSON file
        with open(model_json_file, "r") as json_file:
            loaded_model_json = json_file.read()
            self.loaded_model = model_from_json(loaded_model_json)

        # load weights into the new model
        self.loaded_model.load_weights(model_weights_file)
        #self.loaded_model.compile()
        #self.loaded_model._make_predict_function()

    def predict_emotion(self, img):
        global session
        set_session(session)
        self.preds = self.loaded_model.predict(img)
        accu=float(self.preds[0][np.argmax(self.preds)])
        accu=float("{:.2f}".format(accu*100));
        accu="("+str(accu)+"% )"
        return FacialExpressionModel.EMOTIONS_LIST[np.argmax(self.preds)]+accu

class GenderModel(object):

    Gender_LIST = ["Female", "Male"]

    def __init__(self, model_json_file, model_weights_file):
        # load model from JSON file
        with open(model_json_file, "r") as json_file:
            loaded_model_json = json_file.read()
            self.loaded_model = model_from_json(loaded_model_json)

        # load weights into the new model
        self.loaded_model.load_weights(model_weights_file)
        #self.loaded_model.compile()
        #self.loaded_model._make_predict_function()

    def predict_gender(self, img):
        global session
        set_session(session)
        self.preds = self.loaded_model.predict(img)
        accu=float(self.preds[0][np.argmax(self.preds)])
        accu=float("{:.2f}".format(accu*100));
        accu="("+str(accu)+"% )"
        return GenderModel.Gender_LIST[np.argmax(self.preds)]+accu



class LeafModel(object):

    output_LIST = ["Tomato,Late Blight",
                   "Tomato,Healthy",
                   "Grape,Healthy",
                   "Orange,Haunglongbing(Citrus greening)",
                   "Soybean,healthy",
                   "Squash,Powdery Mildew",
                   "Potato,healthy",
                   "Corn (maize),Northern Leaf Blight",
                   "Tomato,Early blight",
                   "Tomato,Septoria leaf spot",
                   "Corn (maize),Cercospora leaf spot Gray leaf spot",
                   "Strawberry,Leaf scorch",
                   "Peach,healthy",
                   "Apple,Apple scab",
                   "Tomato,Tomato Yellow Leaf Curl Virus",
                   "Tomato,Bacterial spot",
                   "Apple,Black rot",
                   "Blueberry,healthy",
                   "Cherry (including sour),Powdery mildew",
                   "Peach,Bacterial spot",
                   "Apple,Cedar apple rust",
                   "Tomato,Target Spot",
                   "Pepper, bell,healthy",
                   "Grape,Leaf blight (Isariopsis Leaf Spot)",
                   "Potato,Late blight",
                   "Tomato,Tomato mosaic virus",
                   "Strawberry,healthy",
                   "Apple,healthy",
                   "Grape,Black rot",
                   "Potato,Early blight",
                   "Cherry (including sour),healthy",
                   "Corn (maize),Common rust ",
                   "Grape,Esca (Black Measles)",
                   "Raspberry,healthy",
                   "Tomato,Leaf Mold",
                   "Tomato,Spider mites Two-spotted spider mite",
                   "Pepper/bell,Bacterial spot",
                   "Corn (maize),healthy"]
                   

    def __init__(self, model_json_file, model_weights_file):
        # load model from JSON file
        with open(model_json_file, "r") as json_file:
            loaded_model_json = json_file.read()
            self.loaded_model = model_from_json(loaded_model_json)

        # load weights into the new model
        self.loaded_model.load_weights(model_weights_file)
        #self.loaded_model.compile()
        #self.loaded_model._make_predict_function()

    def predict_leaf(self, img):
        global session
        set_session(session)
        self.preds = self.loaded_model.predict(img)
        accu=float(self.preds[0][np.argmax(self.preds)])
        accu=float("{:.2f}".format(accu*100));
        accu=" "+str(accu)+"% "
        ret_Args=list(LeafModel.output_LIST[np.argmax(self.preds)].split(","));
        ret_Args.append(accu);
        return ret_Args;

if __name__=="__main__":
  print("Total Classes(Leaf Detection):",len(LeafModel.output_LIST));
  plants=[];
  diseases=[];
  for i in LeafModel.output_LIST:
    if i.split(",")[0] not in plants:
      plants.append(i.split(",")[0]);
    if i.split(",")[1] not in diseases:
      diseases.append(i.split(",")[1])
  print("plants(count= {})".format(len(plants)))
  print("Diseases(count= {})".format(len(diseases))) 
