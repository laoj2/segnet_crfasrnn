# SegNet + CRF as RNN.

The README will be updated soon. 
SegNet implementation: https://github.com/divamgupta/image-segmentation-keras
CRF as RNN implementation: https://github.com/sadeepj/crfasrnn_keras


## keras.json content
```json
{
    "epsilon": 1e-07, 
    "floatx": "float32", 
    "image_data_format": "channels_last", 

    "backend": "theano"
}
```


## Visualizing the prepared data

You can also visualize your prepared annotations for verification of the prepared data.

```shell
python visualizeDataset.py \
 --images="data/dataset1/images_prepped_train/" \
 --annotations="data/dataset1/annotations_prepped_train/" \
 --n_classes=11 
```


## Training the Model

To train the model run the following command:

```shell
TENSORFLOW_FLAGS=device=cuda0,image_data_format=channels_last,floatX=float32 python train.py --save_weights_path="weights/ex1" --train_images="data/dataset1/images_prepped_train/" --train_annotations="data/dataset1/annotations_prepped_train/" --val_images="data/dataset1/images_prepped_test/" --val_annotations="data/dataset1/annotations_prepped_test/" --n_classes=11 --model_name="segnet" --input_height=128 --input_width=128
```

## Getting the predictions
TBD




