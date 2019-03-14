from keras.utils import plot_model
from keras.models import load_model


model = load_model('testModel.h5')
plot_model(model, show_shapes=True, to_file='model.png')
