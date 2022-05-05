import numpy as np
import cv2 #opencv-python
from tensorflow.keras.models import load_model # install TF 2.5.0 to avoid error "'Sequential' object has no attribute 'predict_classes'"
from gtts import gTTS


print('Isaac will now start screening your Sudoku and let you know the missing numbers asap...')
model = load_model('ModelDigits.h5') # initialize the CNN model
picture = cv2.imread("SudokuUnfinished.jpg")
picture = cv2.resize(picture, (450,450)) # resize image to width 450 and height 450 so it becomes a square picture (widthImg, heightImg))
imgGrey = cv2.cvtColor(picture, cv2.COLOR_BGR2GRAY)
imgBlur = cv2.GaussianBlur(imgGrey, (5, 5), 1)
imgThreshold = cv2.adaptiveThreshold(imgBlur, 255, 1, 1, 11, 2)

# Outline spotting
PictureOutline = picture.copy()
PicBigOutline = picture.copy()
outline, hierarchy = cv2.findContours(imgThreshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE) # Get the outlines
cv2.drawContours(PictureOutline, outline, -1, (0, 255, 0), 3) # write back all outlines found

def GreatestOutline(outline): # capture the greatest outline which is expected to be the Soduko on that image
    biggest = np.array([])
    biggestArea = 0
    for i in outline:
        area = cv2.contourArea(i)
        if area > 50:
            peri = cv2.arcLength(i, True)
            approx = cv2.approxPolyDP(i, 0.02 * peri, True)
            if area > biggestArea and len(approx) == 4:
                biggest = approx
                biggestArea = area
    return biggest,biggestArea

biggest, target = GreatestOutline(outline) # get the greatest outline
def Reorder(biggest):
    biggest = biggest.reshape((4, 2))
    newTarget = np.zeros((4, 1, 2), dtype=np.int32)
    add = biggest.sum(1)
    newTarget[0] = biggest[np.argmin(add)]
    newTarget[3] =biggest[np.argmax(add)]
    diff = np.diff(biggest, axis=1)
    newTarget[1] =biggest[np.argmin(diff)]
    newTarget[2] = biggest[np.argmax(diff)]
    return newTarget
biggest = Reorder(biggest)
cv2.drawContours(PicBigOutline, biggest, -1, (0, 0, 255), 25) # paint the Sudoku frame
pts1 = np.float32(biggest) # Warp processing
pts2 = np.float32([[0, 0],[450, 0], [0, 450],[450, 450]]) #  width-height
matrix = cv2.getPerspectiveTransform(pts1, pts2)
warpColored = cv2.warpPerspective(picture, matrix, (450, 450)) # width-height
warpColored = cv2.cvtColor(warpColored,cv2.COLOR_BGR2GRAY)

def SplitBoxes(picture): # split Sudoku frame into its 81 squares
    rows = np.vsplit(picture,9)
    boxes=[]
    for r in rows:
        column= np.hsplit(r,9)
        for box in column:
            boxes.append(box)
    return boxes
boxes = SplitBoxes(warpColored)

def RecPrediction(boxes,model): # find available numbers
    result = []
    for image in boxes:
        picture = np.asarray(image)
        picture = picture[4:picture.shape[0] - 4, 4:picture.shape[1] -4]
        picture = cv2.resize(picture, (28, 28))
        picture = picture / 255
        picture = picture.reshape(1, 28, 28, 1)
        predictions = model.predict(picture)
        classIndex = model.predict_classes(picture)
        probabilityValue = np.amax(predictions)
        if probabilityValue > 0.8: # only append the number if more than 80% confident that the number is matching from the trained model
            result.append(classIndex[0])
        else:
            result.append(0) # if below 80%, than estimate it is a blank - not yet filled square - and treat that as zero ( 0 does not exist in Sudoku)
    return result
numbers = RecPrediction(boxes, model)

textfile = open("SudokuInput.txt", "w") # write all elements into Input.txt, # 0 means no number is in this square
for element in numbers:
    textfile.write(str(element))
textfile.close()

textfiles = open("SudokuInput.txt", "r") # write all elements into SudokuRev.txt with a new line after each 9th number
data = textfiles.read()
my_str = str(data)
group = 9
char = "\n"
hallo =char.join(my_str[i:i + group] for i in range(0, len(my_str), group))
textfile = open("SudokuRev.txt", "w")
for element in hallo:
    textfile.write(str(element))
textfiles.close()
textfile.close()


class Sudoku:
    def __init__(self, variables, areas, condition):
        self.variables = variables
        self.areas = areas
        self.condition = condition

    def solve(self, task={}):
        if len(task) == len(self.variables):
            return task
        unassigned = [var for var in self.variables if var not in task]
        test_variable = unassigned[0]
        for value in self.areas[test_variable]:
            test_transfer = task.copy()
            test_transfer[test_variable] = value

            if self.condition(test_transfer):
                result = self.solve(test_transfer)
                if result is not None:
                    return result
        return None

def Dic(dict_to_filter, condition):
    filtered_list = []
    for key in dict_to_filter:
        if condition(key):
            filtered_list.append(dict_to_filter[key])
    return filtered_list

def NoDuplicateRowColThree(task):
    for line in range(0, 9):
        line_fields = Dic(task, lambda key: key[0] == line)
        if len(line_fields) > len(set(line_fields)):
            return False
    for column in range(0, 9):
        column_fields = Dic(task, lambda key: key[1] == column)
        if len(column_fields) > len(set(column_fields)):
            return False
    for start_line in range(0, 10, 3):
        for start_column in range(0, 10, 3):
            square_fields = []
            for line in range(start_line, start_line + 3):
                for column in range(start_column, start_column + 3):
                    if (line, column) in task:
                        square_fields.append(task[(line, column)])
            if len(square_fields) > len(set(square_fields)):
                return False
    return True

def SpeakResult(task):
    voice_texts = ""
    for line in range(0, 9):
        voice_texts += '|'
        for column in range(0, 9):
            if (line, column) in task:
                voice_texts += str(task[(line, column)])
            else:
                voice_texts += ''
            voice_texts += ' '
    print(voice_texts)
    voice_text = ""
    for i in voice_texts.split():
        voice_text += i + ' '
    voice_text = voice_text[:-1]
    voice_text ="Hi, this is Isaac telling you your Sudoku numbers"+voice_text
    tts = gTTS(voice_text)
    tts.save("IsaacReadsYourSudokuSolution.mp3")

variables = []
possibilty = {}
filled = {}
result = {}
file = open("SudokuRev.txt", "r")
line_index = 0
for line in file:
    for char_index, char in enumerate(line):
        if char in ['1', '2', '3', '4', '5', '6', '7', '8', '9']:
            result[(line_index, char_index)] = int(char)
    line_index += 1
filled=result
print(result) # get row and column index of each number <> 0 in line in text file

for line in range(0, 9):
    for column in range(0, 9):
        field = (line, column)
        variables.append(field)
        if field in filled:
            possibilty[field] = [filled[field]]
        else:
            possibilty[field] = list(range(1, 10))
ResultClass = Sudoku(variables, possibilty,NoDuplicateRowColThree)
solution = ResultClass.solve(filled)
SpeakResult(solution)
print('Isaac screened your Sudoku, filled in missing numbers and saved a text to speech message.')

