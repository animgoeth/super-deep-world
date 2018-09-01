This project is a an implementation of the Video Pixel Network deep learning model, configured to be applied to gameplay videos from Super Mario World.

#### 1. Installation

Run the following commands:

```
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

#### 2. Data preprocessing

You need to have your recordings stored in a single folder. In order to cut them up into required frame sequences, use the data_preprocessing script:

```
python data_preprocessing.py --videos_path VIDEOS_PATH --output_path OUTPUT_PATH --seq_len SEQUENCE_LENGTH --resize RESIZE
```

You can control the sequence length with the *seq_len* argument (default is 20).
Additionally, you can resize the frames to 64x64 resolution with the *resize* argument (default is False).

#### 3. Training

To train the model, execute the following command:

```
python sdw_main.py --arch_size ARCH_SIZE --data_path DATA_PATH --train_output_path TRAIN_OUTPUT --test_output_path TEST_OUTPUT --resume RESUME
```

The *arch_size* argument controls which version of the model will be used (either small, medium or full).
You can point to the directory that contains the training data (default ./data), as well as to the train output (default ./output) and test output (default ./test_output) directories.
The *resume* argument allows resuming the training from the latest saved checkpoint.

#### 4. Evaluating results

After the model is trained, it generates predictions based on the testing sequences. You can evaluate the accuracy with the eval_accuracy script:

```
python eval_accuracy.py --test_output_path TEST_OUTPUT --save SAVE
```

The *save* argument toggles whether or not to save JPG images for each loaded frame.

#### 5. Stitching predictions

Of you'd like to see the predictions against the actual frames, you can use the stitch_results script:

```
python stitch_results.py --test_output_path TEST_OUTPUT --output_filename FILENAME
```

The resulting image is a mosaic where there are 5 pairs of randomly selected sequences. In each pair the top sequence is the actual one, while the bottom sequence is the predicted one.