import numpy as np
import cv2
import os
import argparse as ap


def prepare_video_sequences(videos_path, output_path, seq_len, resize=False):
    if not os.path.exists(output_path):
        os.mkdir(output_path)

    for video in os.listdir(videos_path):
        capture = cv2.VideoCapture(os.path.join(videos_path, video))

        train_counter = 0
        training_data = []
        frames = []

        while True:
            ret, frame = capture.read()

            if not ret:
                break

            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            if resize:
                frame = cv2.resize(frame, (64, 64))

            if train_counter < seq_len:
                frames.append(frame)
                train_counter += 1
            else:
                training_data.append(frames)
                frames = []
                train_counter = 0

        np.save(os.path.join(output_path, '%s_train.npy' % video.split('.')[0]), training_data)
        capture.release()


if __name__ == "__main__":
    parser = ap.ArgumentParser()
    parser.add_argument("--videos_path", dest="videos_path", required=True)
    parser.add_argument("--output_path", dest="output_path", default="./data")
    parser.add_argument("--resize", dest="resize", default=False, type=bool)
    parser.add_argument("--seq_len", dest="seq_len", default=20, type=int)
    args = parser.parse_args()

    prepare_video_sequences(args.videos_path, args.output_path, args.seq_len, args.resize)
