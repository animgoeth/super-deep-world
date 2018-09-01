from PIL import Image
import random
import os
import argparse as ap

TEST_SEQUENCES_COUNT = 315
FRAME_WIDTH = 256
FRAME_HEIGHT = 224


def prepare_image(test_output_path, filename):
    result_image = Image.new('RGB', ((FRAME_WIDTH + 2) * 10, (FRAME_HEIGHT + 2) * 10))
    selected_count = 0
    selected = set()
    while selected_count < 10:
        while True:
            i = random.randint(0, TEST_SEQUENCES_COUNT)
            if i not in selected:
                selected.add(i)
                break

        for j in range(0, 10):
            actual_image = Image.open(os.path.join(test_output_path, '%d/actual_%d.jpg' % (i, j)))
            result_image.paste(actual_image, (j * (FRAME_WIDTH + 2), selected_count * (FRAME_HEIGHT + 2)))

            predicted_image = Image.open(os.path.join(test_output_path, '%d/predicted_%d.jpg' % (i, j)))
            result_image.paste(predicted_image, (j * (FRAME_WIDTH + 2), (selected_count + 1) * (FRAME_HEIGHT + 2)))

        selected_count += 2

    result_image.save(filename)


if __name__ == "__main__":
    parser = ap.ArgumentParser()
    parser.add_argument("--test_output_path", dest="test_output_path", required=True)
    parser.add_argument("--output_filename", dest="filename", default="stitched.jpg")
    args = parser.parse_args()

    prepare_image(args.test_output_path, args.filename)
