import pickle
import numpy as np

df = open('yolo_model.pkl', "rb")
m = pickle.load(df)
print(m)