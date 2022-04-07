import numpy as np
import os, cv2

from keras import models
from keras.models import Sequential, save_model, load_model
from keras.layers import Dense, Flatten, Dropout
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from keras.callbacks import ModelCheckpoint
from keras.utils.vis_utils import plot_model

from PIL import Image
import matplotlib.pyplot as plt


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
 
train_datagen = ImageDataGenerator(
    rotation_range = 0,        
    width_shift_range=0.1,      
    height_shift_range=0.1,
    brightness_range=None,
    shear_range=0.0,
    zoom_range=0.0,
    fill_mode='nearest',
    horizontal_flip = False,
    vertical_flip = True,
    rescale = 1/255.0,
    data_format='channels_last'
) 

test_datagen =  ImageDataGenerator(rescale = 1/255.)

train_generator = train_datagen.flow_from_directory (
	'train_num',
    target_size=(36,30),#학습용 이미지의 타겟사이즈를 세로36,가로30 픽셀로 변경하였습니다.
    batch_size=3,
    class_mode='categorical',
    shuffle = True,
    )

test_generator = test_datagen.flow_from_directory (
	'test_num', target_size=(36,30), batch_size=1, shuffle = False, class_mode='categorical')


total_calss_no = 30 #분류할 총 클래스수=30개로 변경하였습니다.

model = Sequential()            

model.add(Conv2D(32, kernel_size=(3,3), activation='relu', input_shape=(36,30,3))) #인풋레이어의 인풋영상 크기를 세로36,가로30 픽셀로 수정하였습니다.

model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Dropout(0.2))

model.add(Conv2D(64, kernel_size=(3,3), activation='relu'))

model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Dropout(0.2))

model.add(Flatten())

model.add(Dense(64, activation='relu'))

model.add(Dense(total_calss_no, activation='softmax'))

save_model(model, 'classify_model.tf')

print(model.summary())
plot_model(model, to_file='model_plot.png', show_shapes=True, show_layer_names=True)

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

checkpointer = ModelCheckpoint(filepath="best_weights.hdf5", 
                               monitor= 'val_accuracy', mode='max', verbose=1, 
                               save_weights_only=True, save_best_only=True)

step_size_train = train_generator.n // train_generator.batch_size
#step_size_valid = valid_generator.n // valid_generator.batch_size
step_size_test = test_generator.n // test_generator.batch_size

history= model.fit_generator(train_generator, steps_per_epoch=step_size_train, 
    callbacks = [checkpointer],
	validation_data=train_generator,validation_steps=step_size_train, epochs=100)

model.load_weights('best_weights.hdf5')

save_model(model, 'classify_model.tf')

print('\n*********** Evaluation ***********')

scores = model.evaluate_generator(train_generator, steps=step_size_train)

for i in range(len(model.metrics_names)) :
    print('%s: %.2f %%' %(model.metrics_names[i], scores[i]*100))

print('\n*********** Prediction ***********')
np.set_printoptions(formatter={'float': lambda x: '{0:0.3f}'.format(x)})

print(test_generator.class_indices)
    
cls_md = "Class Mode: "+test_generator.class_mode

print(cls_md)
  
print('\n[[[ 학습한 데이터에 대한 predition 및 출력 - 임의순서로 출력 ]]]')
output = model.predict_generator(train_generator, steps=step_size_train)
print(output)

print('\n[[[ 테스트 데이터에 대한 predition 및 출력 - 폴더명 알파벳순으로 출력 ]]]')
output = model.predict_generator(test_generator, steps=step_size_test, verbose=1)

test_filenames = []
for file in test_generator.filenames :
    test_filenames.append(file)
    #print(file)

for no in range(len(output)) :
    print("\n[",no, "]번째 이미지 ", test_filenames[no], " 에 대한 분류 결과")
    print('\t',output[no])
    maxValIndices = [i for i, x in enumerate(output[no]) if x == max(output[no])]
    print("\t계산한 답은 ",end = '')
    if maxValIndices == [10] :  #결과를 편하게 확인할수있도록 결과내용처리 과정에 추가하였습니다.
        print("k1(meo)")
    elif maxValIndices == [11] :
        print("k10(neo)")
    elif maxValIndices == [12] :
        print("k11(moo)")
    elif maxValIndices == [13] :
        print("k12(deo)")
    elif maxValIndices == [14] :
        print("k13(goo)")
    elif maxValIndices == [15] :
        print("k14(bo)")
    elif maxValIndices == [16] :
        print("k15(jo)")
    elif maxValIndices == [17] :
        print("k16(joo)")
    elif maxValIndices == [18] :
        print("k17(ho)")
    elif maxValIndices == [19] :
        print("k18(ha)")
    elif maxValIndices == [20] :
        print("k19(seo)")
    elif maxValIndices == [21] :
        print("k2(go)")
    elif maxValIndices == [22] :
        print("k20(ga)")
    elif maxValIndices == [23] :
        print("k3(soo)")
    elif maxValIndices == [24] :
        print("k4(boo)")
    elif maxValIndices == [25] :
        print("k5(oh)")
    elif maxValIndices == [26] :
        print("k6(ro)")
    elif maxValIndices == [27] :
        print("k7(noo)")
    elif maxValIndices == [28] :
        print("k8(geo)")
    elif maxValIndices == [29] :
        print("k9(woo)")
    else :
        print(maxValIndices)  



print(history.history.keys())

plt.plot(history.history['accuracy'])
plt.plot(history.history['loss'])

plt.title('model accuracy & loss')

plt.ylabel('accuracy/loss')
plt.xlabel('epoch')

plt.legend(['accuracy', 'loss'], loc='upper left')
print("\n종료하려면 그래프 출력 창을 닫으시오.\n")
plt.show()
    
cv2.destroyAllWindows()
    

