
# SegNet + CRF as RNN.

This project has the implementation of SegNetResCRF, combination of SegNet with CRF as RNN.

SegNet implementation: https://github.com/divamgupta/image-segmentation-keras

CRF as RNN implementation: https://github.com/sadeepj/crfasrnn_keras

##  Instalation (The README will be updated soon)

Tested with:

	pip install --upgrade tensorflow-gpu==1.4

	conda install -c menpo opencv3 

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
TENSORFLOW_FLAGS=device=cuda0,image_data_format=channels_last,floatX=float32 python train.py --save_weights_path="weights/ex1/" --train_images="path/train/" --train_annotations="data_semantics/trainannot/" --val_images="data_semantics/val/" --val_annotations="data_semantics/valannot/" --n_classes=8 --model_name="segnet_res_crf" --input_height=128 --input_width=128
```

## Getting the predictions

```shell
TENSORFLOW_FLAGS=device=cuda0,image_data_format=channels_last,floatX=float32 python predict.py --output_path="teste/" --test_images="data_semantics/test/" --n_classes=8 --model_name="segnet_res_crf" --input_height=128 --input_width=128 --save_weights_path="weights_360_480_res_with_crf.hdf5"
```
