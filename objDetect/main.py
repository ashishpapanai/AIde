import predict
from objUtils import *


if __name__ == '__main__':
    predict = predict.Predict(r'data\test_veg.jpg')
    labels = list(set(predict.predict()))
    with open('object.txt', 'w') as f:        
        list(map(lambda item : f.write("%s\n" % item), labels))   
    print(labels)

    Utils.speak('object.txt')
    
