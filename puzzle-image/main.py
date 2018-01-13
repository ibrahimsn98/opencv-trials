import cv2
import numpy as np
import math

pieces = []
combinedCols = []
img = cv2.imread('img.jpg')

hash = {}

pieceCount = input('How many pieces? : ')

if not math.sqrt(pieceCount).is_integer():
    print('Please enter a square number!')
else:

    pieceCountSingle = int(math.sqrt(pieceCount))

    pieceW = np.size(img, 0) / pieceCountSingle
    pieceH = np.size(img, 1) / pieceCountSingle

    for i in range(pieceCount):
        partCol = int(i // pieceCountSingle)
        partRow = int(i % pieceCountSingle)
        pieces.append(img[partRow*pieceW:(partRow+1)*pieceW, partCol*pieceH:(partCol+1)*pieceH])
        print('Width: ' + str(pieceW) + ' Height: ' + str(pieceH) + " Row: " + str(partRow) + ' Col: ' + str(partCol))

    np.random.shuffle(pieces)

    for i in range(pieceCountSingle):
        partRow = int(i % pieceCountSingle)
        combinedCols.append(np.concatenate(pieces[partRow*pieceCountSingle:(partRow+1)*pieceCountSingle], axis=0))
        print('Col ' + str(i) + ' appended!')

    cv2.imshow('Original', img)
    cv2.imshow('Combined', np.concatenate(combinedCols, axis=1))

    cv2.waitKey(0)
    cv2.destroyAllWindows()