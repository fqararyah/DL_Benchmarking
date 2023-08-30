import sys
import json
import numpy as np

IMAGE_NET_GROUND_TRUTH_FILE = 'ground_truth/ILSVRC2012_validation_ground_truth.json'
LABEL_TYPES = ['LABEL_KERAS-CAFFE', 'LABEL_ORG']
LABEL_MODES = [{'NAME': 'KERAS-CAFFE', 'LABEL_TYPE': 0, 'OFFSET': 0},
               {'NAME': 'KERAS_CAFFE+1', 'LABEL_TYPE': 0, 'OFFSET': 1},
               {'NAME': 'KERAS_CAFFE-1', 'LABEL_TYPE': 0, 'OFFSET': -1},
               {'NAME': 'ORIGINAL', 'LABEL_TYPE': 1, 'OFFSET': 0},
               {'NAME': 'ORIGINAL+1', 'LABEL_TYPE': 1, 'OFFSET': 1},
               {'NAME': 'ORIGINAL-1', 'LABEL_TYPE': 1, 'OFFSET': -1}]
INIT_IMG_TEST_CNT = 10
IMG_TEST_MIN_HITS = 3

conf_mat = np.zeros((1000, 1000))
# Return [ TOP_1, TOP_5, SKIP]


def testImage(data, mode):
    top1 = 0
    top5 = 0
    skip = 0
    # To prevent errors due to different capitalization convert the image name to lower case
    img = data['image'].lower()
    # Make sure the image is present in the ground truth dictionary
    if img not in imGT:
        print(
            f"ERROR: Image: {img} not present in ground truth file - skipping ...")
        skip = 1

    if not skip:
        try:
            # Get the correct label
            cDet = int(imGT[img][LABEL_TYPES[mode['LABEL_TYPE']]])
            # Check if any of the detected labels match the correct one
            conf_mat[cDet][int(data['dets'][0])] += 1
            for i, det in enumerate(data['dets']):
                if (int(det) + mode['OFFSET']) == cDet:
                    top5 = 1
                    if i == 0:
                        top1 = 1
        except Exception as e:
            skip = 1

    return top1, top5, skip


def runLabelTest(l, mode):
    testCnt = 0
    for i in range(l):
        _, hit, _ = testImage(res[i], mode)
        testCnt += hit
    return testCnt


def labelTest():
    l = INIT_IMG_TEST_CNT if len(res) >= INIT_IMG_TEST_CNT else len(res)

    for mode in LABEL_MODES:
        cnt = runLabelTest(l, mode)

        if cnt < IMG_TEST_MIN_HITS:
            print(f"[{mode['NAME']}] Less than {IMG_TEST_MIN_HITS} correct classifications in the first {INIT_IMG_TEST_CNT} images, testing again using different label mode...")
        else:
            print(
                f"[{mode['NAME']}] {cnt} / {INIT_IMG_TEST_CNT} correct classifications using this mode...")
            return mode

    print("[WARNING] Non of the known label modes achieved a sufficient number of correct classifications.")
    print("          There are several possible reasons for this.")
    print("          1) The number of test images was to small.")
    print("          2) The labels are mapped using a different scheme than any of those known to the script.")
    print("          3) The accuracy of the used model is very low (overlaps with 1).")
    print("          4) There was an error when extracting the labels outputted by the model.")
    print("          -------------------------------------------------------------------------")
    print("          To counter act some of the problems the accuracy for all modes will be calculated and printed.")
    print("          Should no mode produce a sufficiently high accuracy please contact Florian Porrmann (UBI)")
    print("          email: fporrmann@techfak.uni-bielefeld.de")
    print("")
    return None


def runAccuracyCheck(mode):
    top1 = 0
    top5 = 0
    results = len(res)
    skipCnt = 0

    print("Calculating accuracy ... ", end="")

    for v in res:
        t1, t5, sc = testImage(v, mode)
        top1 += t1
        top5 += t5
        skipCnt += sc

    print("Done")

    # Remove all skipped files from the final evaluation count
    results -= skipCnt

    print("--- Classification Accuracy ---")
    print(f" -- Mode: {mode['NAME']}")
    print(f" -- Top 1: {top1}/{results} - {(top1 / results) * 100:.2f}%")
    print(f" -- Top 5: {top5}/{results} - {(top5 / results) * 100:.2f}%")
    print("-------------------------------")


def accuracyCheck(mode):
    if mode:
        runAccuracyCheck(mode)
    else:
        for mode in LABEL_MODES:
            runAccuracyCheck(mode)


if len(sys.argv) < 2:
    print(f"Usage: {sys.argv[0]} <INPUT_FILE>")
    exit()

# Load imagenet 2012 ground truth data
print(f"Loading ground truth from: {IMAGE_NET_GROUND_TRUTH_FILE} ... ", end="")
imGT = json.load(open(IMAGE_NET_GROUND_TRUTH_FILE))
print("Done")

# Convert all image names to lower case
imGT = {k.lower(): v for k, v in imGT.items()}

# Load the results
resFile = sys.argv[1]
print(f"Loading results from: {resFile} ... ", end="")
res = json.load(open(resFile))
print("Done")

mode = labelTest()

accuracyCheck(mode)

np.savetxt('./conf_mat.txt', conf_mat, fmt='%i')

confusion_matrix = np.diag(np.full(1000, 1000 * 10)) + \
    np.random.randint(0, 20, (1000, 1000))
np.savetxt('./dummy_conf_mat.txt', confusion_matrix, fmt='%i')
