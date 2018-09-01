import numpy as np
import os
import cv2
import argparse as ap

TEST_SEQUENCES_COUNT = 315
SEQUENCE_LENGTH = 10
TRESHOLD_VALUE = 0
COLS = 64
START_ROW = 0
END_ROW = 64


def eval_accuracy(test_output_path, save_jpg):
    total = []
    frame_totals = dict()
    for i in range(0, SEQUENCE_LENGTH):
        frame_totals[i] = []

    for seq_index in range(0, TEST_SEQUENCES_COUNT):
        accuracies = []

        for i in range(0, SEQUENCE_LENGTH):
            actual = np.load(os.path.join(test_output_path, '%d/actual_%s.npy' % (seq_index, str(i))))
            predicted = np.load(os.path.join(test_output_path, '%d/predicted_%s.npy' % (seq_index, str(i))))

            if save_jpg:
                cv2.imwrite(os.path.join(test_output_path, '%d/actual_%s.jpg' % (seq_index, str(i))), actual[0])
                cv2.imwrite(os.path.join(test_output_path, '%d/predicted_%s.jpg' % (seq_index, str(i))), predicted[0])

            good = 0

            for row in range(START_ROW, END_ROW):
                for col in range(0, COLS):
                    actual_pixel = actual[0][row][col]
                    predicted_pixel = predicted[0][row][col]

                    if abs(actual_pixel - predicted_pixel) <= TRESHOLD_VALUE:
                        good += 1

            frame_acc = good / (COLS * (END_ROW - START_ROW))
            accuracies.append(frame_acc)
            frame_totals[i].append(frame_acc)

        total.append(np.mean(accuracies))

    print("Total accuracy: %.4f" % round(float(np.mean(total)), 4))

    for key in frame_totals.keys():
        print("Total for frame %d: %.4f" % (key, round(float(np.mean(frame_totals[key])), 4)))


if __name__ == "__main__":
    parser = ap.ArgumentParser()
    parser.add_argument("--test_output_path", dest="test_output_path", required=True)
    parser.add_argument("--save", dest="save", default=False, type=bool)
    args = parser.parse_args()

    eval_accuracy(args.test_output_path, args.save)
