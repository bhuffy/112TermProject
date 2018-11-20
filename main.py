## Project: Email Text Analyzer Tool: Display

"""
Author: Bennett Huffman
Last Modified: 11/19/18
Course: 15-112
Section: H
"""

## Imports

from tkinter import filedialog
from tkinter import *

## Colors

BRIGHTBLUE = "#3E92CC"
DARKBLUE = "#274060"
MAINBLUE = "#3E92CC"
MEDBLUE = "#004385"
WHITE = "#FFFAFF"

## Initialize Programs! ##

def init(data):
    # There is only one init, not one-per-mode
    data.mode = "startScreen"
    data.margin = 20
    data.filename = None
    
## Check Buttons

def clickedStartButton(data, x, y):
    if (x > data.width/2 - 75 and x < data.width/2 + 75 and
        y > data.height*2/3 - 40 and y < data.height*2/3 + 40):
            return True
    return False

# Checks if user clicks within bounds of continue button
def clickedContinueButton(data, x, y):
    if (x > data.width/2-85 and x < data.width/2+85 and
        y > data.height*11/12-30 and y < data.height*11/12+30):
            return True
    return False
    
# Checks if user clicks within bounds of upload button
def clickedUploadButton(data, x, y):
    if (x > data.width/2-75 and x < data.width/2+75 and
        y > data.height/2-30 and y < data.height/2+30):
            return True
    return False
    
# checks if user clicks within bounds of back to start button
def clickedBTSButton(data, x, y):
    if (x > data.width/2-90 and x < data.width/2+90 and
        y > data.height*11/12-30 and y < data.height*11/12+30):
            return True
    return False

# checks if user clicks within bounds of help button
def clickedHelpButton(data, x, y):
    if (x > data.width-60 and x < data.width and
        y > 0 and y < 40):
            return True
    return False

## Mode dispatcher

def mousePressed(event, data):
    if (data.mode == "startScreen"): startScreenMousePressed(event, data)
    elif (data.mode == "loadDataScreen"):   loadDataScreenMousePressed(event, data)
    elif (data.mode == "labelScreen"):   labelScreenMousePressed(event, data)
    elif (data.mode == "featuresScreen"):   featuresScreenMousePressed(event, data)
    elif (data.mode == "analyzeScreen"):   analyzeScreenMousePressed(event, data)
    elif (data.mode == "helpScreen"):       helpScreenMousePressed(event, data)

def keyPressed(event, data):
    if (data.mode == "startScreen"): startScreenKeyPressed(event, data)
    elif (data.mode == "loadDataScreen"):   loadDataScreenKeyPressed(event, data)
    elif (data.mode == "labelScreen"):   labelScreenKeyPressed(event, data)
    elif (data.mode == "featuresScreen"):   featuresScreenKeyPressed(event, data)
    elif (data.mode == "analyzeScreen"):   analyzeScreenKeyPressed(event, data)
    elif (data.mode == "helpScreen"):       helpScreenKeyPressed(event, data)

def timerFired(data):
    if (data.mode == "startScreen"): startScreenTimerFired(data)
    elif (data.mode == "loadDataScreen"):   loadDataScreenTimerFired(data)
    elif (data.mode == "labelScreen"):   labelScreenTimerFired(data)
    elif (data.mode == "featuresScreen"):   featuresScreenTimerFired(data)
    elif (data.mode == "analyzeScreen"):   analyzeScreenTimerFired(data)
    elif (data.mode == "helpScreen"):       helpScreenTimerFired(data)

def redrawAll(canvas, data):
    if (data.mode == "startScreen"): startScreenRedrawAll(canvas, data)
    elif (data.mode == "loadDataScreen"):   loadDataScreenRedrawAll(canvas, data)
    elif (data.mode == "labelScreen"):   labelScreenRedrawAll(canvas, data)
    elif (data.mode == "featuresScreen"):   featuresScreenRedrawAll(canvas, data)
    elif (data.mode == "analyzeScreen"):   analyzeScreenRedrawAll(canvas, data)
    elif (data.mode == "helpScreen"):       helpScreenRedrawAll(canvas, data)

## startScreen mode

def startScreenMousePressed(event, data):
    x = event.x
    y = event.y
    if clickedStartButton(data, x, y):
        data.mode = "loadDataScreen"
    elif clickedHelpButton(data, x, y):
        data.mode = "helpScreen"

def startScreenKeyPressed(event, data):
    pass

def startScreenTimerFired(data):
    pass

def startScreenRedrawAll(canvas, data):
    # draws background and description
    canvas.create_rectangle(0, 0, data.width, data.height, fill=MAINBLUE, outline="")
    canvas.create_text(data.width/2, data.height/3-20,
                       text="WELCOME TO CLASS-E!", font="Arial 48 bold", fill=WHITE)
    canvas.create_text(data.width/2, data.height/3+40,
                       text="An email labeling and analysis tool for company complaints.", font="Arial 18 bold", fill=WHITE)
    
    # help button
    canvas.create_text(data.width-30, 20,
                       text="HELP", font="Arial 16 bold", fill=WHITE)
    
    # start button
    canvas.create_rectangle(data.width/2-75, data.height*2/3-30, data.width/2+75, data.height*2/3+30, fill=DARKBLUE, outline="")
    canvas.create_text(data.width/2, data.height*2/3,
                       text="START", font="Arial 18", fill=WHITE)


## loadDataScreen mode

def loadDataScreenMousePressed(event, data):
    x = event.x
    y = event.y
    #make some sort of cool success animation if file is CSV and uploaded
    if clickedUploadButton(data, x, y):
        data.filename = filedialog.askopenfilename(initialdir = "/",title = "Select file",
                                                    filetypes = (("CSV files","*.csv"),("all files","*.")))
        print(data.filename)
    elif clickedContinueButton(data, x,y):
        data.mode = "labelScreen"
    elif clickedHelpButton(data, x, y):
        data.mode = "helpScreen"

def loadDataScreenKeyPressed(event, data):
    pass

def loadDataScreenTimerFired(data):
    pass

def loadDataScreenRedrawAll(canvas, data):
    # draws background and instructions
    canvas.create_rectangle(0,0,data.width,data.height, fill=MAINBLUE, outline="")
    canvas.create_rectangle(data.margin, data.margin+75,
                            data.width-data.margin, data.height-data.margin-75, outline=WHITE, dash=(5, 10), fill=MAINBLUE)
    canvas.create_text(data.width/2, data.height/12,
                       text="Drag and drop or press 'upload' to upload your csv of emails.", font="Arial 18 bold", fill=WHITE)
    
    # help button
    canvas.create_text(data.width-30, 20,
                       text="HELP", font="Arial 16 bold", fill=WHITE)
    
    # upload button
    canvas.create_rectangle(data.width/2-75, data.height/2-30, data.width/2+75, data.height/2+30, fill=DARKBLUE, outline="")
    canvas.create_text(data.width/2, data.height/2,
                       text="UPLOAD", font="Arial 18", fill=WHITE)
                 
    # continue button
    canvas.create_rectangle(data.width/2-85, data.height*11/12-30, data.width/2+85, data.height*11/12+30, fill=DARKBLUE, outline="")
    canvas.create_text(data.width/2, data.height*11/12,
                       text="CONTINUE", font="Arial 18", fill=WHITE)

## labelScreen mode

def labelScreenMousePressed(event, data):
    x = event.x
    y = event.y
    if clickedContinueButton(data, x, y):
        data.mode = "featuresScreen"
    elif clickedHelpButton(data, x, y):
        data.mode = "helpScreen"

def labelScreenKeyPressed(event, data):
    pass

def labelScreenTimerFired(data):
    pass

def labelScreenRedrawAll(canvas, data):
    # draws background and instructions
    canvas.create_rectangle(0,0,data.width,data.height, fill=MAINBLUE, outline="")
    canvas.create_text(data.width/2, data.height/12,
                       text="Do you have any pre-determined labels?", font="Arial 18 bold", fill=WHITE)
    
    # help button
    canvas.create_text(data.width-30, 20,
                       text="HELP", font="Arial 16 bold", fill=WHITE)
    
    # continue button
    canvas.create_rectangle(data.width/2-85, data.height*11/12-30, data.width/2+85, data.height*11/12+30, fill=DARKBLUE, outline="")
    canvas.create_text(data.width/2, data.height*11/12,
                       text="CONTINUE", font="Arial 18", fill=WHITE)
                       
## featuresScreen mode

def featuresScreenMousePressed(event, data):
    x = event.x
    y = event.y
    if clickedContinueButton(data, x, y):
        data.mode = "analyzeScreen"
        import loaddata as ld # will need this 
    elif clickedHelpButton(data, x, y):
        data.mode = "helpScreen"

def featuresScreenKeyPressed(event, data):
    pass

def featuresScreenTimerFired(data):
    pass

def featuresScreenRedrawAll(canvas, data):
    # draws background and instructions
    canvas.create_rectangle(0,0,data.width,data.height, fill=MAINBLUE, outline="")
    canvas.create_text(data.width/2, data.height/12,
                       text="Which features do you want to view?", font="Arial 18 bold", fill=WHITE)
    
    # help button
    canvas.create_text(data.width-30, 20,
                       text="HELP", font="Arial 16 bold", fill=WHITE)
    
    # continue button
    canvas.create_rectangle(data.width/2-85, data.height*11/12-30, data.width/2+85, data.height*11/12+30, fill=DARKBLUE, outline="")
    canvas.create_text(data.width/2, data.height*11/12,
                       text="CONTINUE", font="Arial 18", fill=WHITE)

## analyzeScreen mode

def analyzeScreenMousePressed(event, data):
    x = event.x
    y = event.y
    if clickedBTSButton(data, x, y):
        data.mode = "startScreen"
    elif clickedHelpButton(data, x, y):
        data.mode = "helpScreen"

def analyzeScreenKeyPressed(event, data):
    pass

def analyzeScreenTimerFired(data):
    pass

def analyzeScreenRedrawAll(canvas, data):
    # draws background and instructions
    canvas.create_rectangle(0,0,data.width,data.height, fill=MAINBLUE, outline="")
    canvas.create_text(data.width/2, data.height/2-20,
                       text="Data here!", font="Arial 24 bold", fill=WHITE)
    canvas.create_text(data.width/2, data.height/2+20,
                       text="oh what fun!", font="Arial 18", fill=WHITE)
    
    # help button
    canvas.create_text(data.width-30, 20,
                       text="HELP", font="Arial 16 bold", fill=WHITE)
    
    # back to start button
    canvas.create_rectangle(data.width/2-90, data.height*11/12-30, data.width/2+90, data.height*11/12+30, fill=DARKBLUE, outline="")
    canvas.create_text(data.width/2, data.height*11/12,
                       text="BACK TO START", font="Arial 18", fill=WHITE)


## helpScreen mode

def helpScreenMousePressed(event, data):
    x = event.x
    y = event.y
    if clickedBTSButton(data, x, y):
        data.mode = "startScreen"

def helpScreenKeyPressed(event, data):
    pass

def helpScreenTimerFired(data):
    pass

def helpScreenRedrawAll(canvas, data):
    canvas.create_rectangle(0,0,data.width,data.height, fill=MAINBLUE, outline="")
    canvas.create_text(data.width/2, data.height/2-40,
                       text="How to Use Class-E", font="Arial 26 bold")
    # back to start button
    canvas.create_rectangle(data.width/2-90, data.height*11/12-30, data.width/2+90, data.height*11/12+30, fill=DARKBLUE, outline="")
    canvas.create_text(data.width/2, data.height*11/12,
                       text="BACK", font="Arial 18", fill=WHITE)

## Run Function

def run(width=300, height=300):
    def redrawAllWrapper(canvas, data):
        canvas.delete(ALL)
        canvas.create_rectangle(0, 0, data.width, data.height,
                                fill='white', width=0)
        redrawAll(canvas, data)
        canvas.update()    

    def mousePressedWrapper(event, canvas, data):
        mousePressed(event, data)
        redrawAllWrapper(canvas, data)

    def keyPressedWrapper(event, canvas, data):
        keyPressed(event, data)
        redrawAllWrapper(canvas, data)

    def timerFiredWrapper(canvas, data):
        timerFired(data)
        redrawAllWrapper(canvas, data)
        # pause, then call timerFired again
        canvas.after(data.timerDelay, timerFiredWrapper, canvas, data)
    # Set up data and call init
    class Struct(object): pass
    data = Struct()
    data.width = width
    data.height = height
    data.timerDelay = 100 # milliseconds
    root = Tk()
    root.resizable(width=False, height=False) # prevents resizing window
    init(data)
    # create the root and the canvas
    canvas = Canvas(root, width=data.width, height=data.height)
    canvas.configure(bd=0, highlightthickness=0)
    canvas.pack()
    # set up events
    root.bind("<Button-1>", lambda event:
                            mousePressedWrapper(event, canvas, data))
    root.bind("<Key>", lambda event:
                            keyPressedWrapper(event, canvas, data))
    timerFiredWrapper(canvas, data)
    # and launch the app
    root.mainloop()  # blocks until window is closed
    print("bye!")

run(720, 580)