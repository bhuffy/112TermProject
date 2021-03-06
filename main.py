## Project: Email Text Analyzer Tool: Display

"""
Author: Bennett Huffman
Last Modified: 12/6/18
Course: 15-112
Section: H
"""

## Imports

#For using matplot inside 
import matplotlib
matplotlib.use("TkAgg")
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

# make the plot look good amiright
import matplotlib.animation as animation
from matplotlib import style
style.use("ggplot")

#tkinter is the best graphics library change my mind
from tkinter import filedialog
import tkinter as tk
from tkinter import *
from tkinter import messagebox

import random

# analysis tool file
import analyzer as a

## Colors

BRIGHTBLUE = "#3E92CC"
NODEBLUE = "#BEE1F7"
EDGEBLUE = "#61A8DA"
DARKBLUE = "#274060"
HOVER_DARKBLUE = "#152C49"
MAINBLUE = "#3E92CC"
MEDBLUE = "#004385"
OFFWHITE = "#D6E8F4"
WHITE = "#FFFFFF"
GREY = "#BEBEBE"
RED = "#FF003B"

## Initialize Programs! ##

def init(data):
    # There is only one init, not one-per-mode
    data.mode = "startScreen"
    data.btnColor = {
        "start": DARKBLUE,
        "continue": GREY,
        "upload": DARKBLUE,
        "bts": DARKBLUE,
        "back": OFFWHITE,
        "help": OFFWHITE,
        "labels": DARKBLUE,
        "freqdist": DARKBLUE,
        "netdiagram": DARKBLUE,
        "summary": DARKBLUE,
        "sentiment": DARKBLUE,
        "exportCSV": DARKBLUE
    }
    
    data.features = {
        'labels': [False, OFFWHITE],
        'freqdist': [False, OFFWHITE],
        'netdiagram': [False, OFFWHITE],
        'summary': [False, OFFWHITE],
        'sentiment': [False, OFFWHITE],
        'exportCSV': [False, OFFWHITE]
    }
    
    data.margin = 20
    data.filename = None
    data.results = None
    data.analysisInProgress = False
    
    data.speed = 1
    data.particles = generateParticles(data, 70)
    data.edges = generateEdges(data, 15)
    
# generates a list of particles
def generateParticles(data, num):
    particles = []
    for i in range(num):
        x = random.randint(0, data.width)
        y = random.randint(0, data.height)
        r = random.randint(2,3)
        particle = [x, y, r]
        particles.append(particle)
    return particles
    
# generates edges connecting randomly selected particles
def generateEdges(data, multiple):
    edges = []
    for i in range(len(data.particles)*multiple):
        x1,y1,r1 = data.particles[random.randint(0, len(data.particles)-1)]
        x2,y2,r2 = data.particles[random.randint(0, len(data.particles)-1)]
        edges.append((x1,y1,x2,y2))
    return edges
    
# draws the particles on canvas
def drawParticles(data, canvas):
    for particle in data.particles:
        x,y,r = particle
        canvas.create_oval(x-r,y-r,x+r,y+r,outline="", fill=NODEBLUE)
        
# checks to see if the particles are close to each other for style
def checkParticleDistance(x1,y1,x2,y2):
    dist = ((x2-x1)**2 + (y2-y1)**2) ** 0.5
    if dist < 125:
        return True
    return False

# draws the edges on canvas
def drawEdges(data, canvas):
    for edge in data.edges:
        x1,y1,x2,y2 = edge
        if checkParticleDistance(x1,y1,x2,y2):
            canvas.create_line(x1,y1,x2,y2, width=1, fill=EDGEBLUE)
        
    
## Check Buttons

# Checks if user clicks within bounds of start button
def onStartButton(data, x, y):
    if (x > data.width/2 - 75 and x < data.width/2 + 75 and
        y > data.height*2/3 - 40 and y < data.height*2/3 + 40):
            return True
    return False

# Checks if user clicks within bounds of continue button
def onContinueButton(data, x, y):
    if (x > data.width/2-85 and x < data.width/2+85 and
        y > data.height*11/12-30 and y < data.height*11/12+30):
            return True
    return False
    
# Checks if user clicks within bounds of upload button
def onUploadButton(data, x, y):
    if (x > data.width/2-75 and x < data.width/2+75 and
        y > data.height/2-30 and y < data.height/2+30):
            return True
    return False
    
# checks if user clicks within bounds of back to start button
def onBTSButton(data, x, y):
    if (x > data.width/2-90 and x < data.width/2+90 and
        y > data.height*11/12-30 and y < data.height*11/12+30):
            return True
    return False

# checks if user clicks within bounds of help button
def onHelpButton(data, x, y):
    if (x > data.width-60 and x < data.width and
        y > 0 and y < 40):
            return True
    return False
    
# checks if user clicks within bounds of help button
def onBackButton(data, x, y):
    if (x > 0 and x < 60 and
        y > 0 and y < 40):
            return True
    return False

## Mode dispatcher

def mousePressed(event, data):
    if (data.mode == "startScreen"): startScreenMousePressed(event, data)
    elif (data.mode == "loadDataScreen"):   loadDataScreenMousePressed(event, data)
    elif (data.mode == "featuresScreen"):   featuresScreenMousePressed(event, data)
    elif (data.mode == "analyzeScreen"):   analyzeScreenMousePressed(event, data)
    elif (data.mode == "helpScreen"):       helpScreenMousePressed(event, data)

def keyPressed(event, data):
    if (data.mode == "startScreen"): startScreenKeyPressed(event, data)
    elif (data.mode == "loadDataScreen"):   loadDataScreenKeyPressed(event, data)
    elif (data.mode == "featuresScreen"):   featuresScreenKeyPressed(event, data)
    elif (data.mode == "analyzeScreen"):   analyzeScreenKeyPressed(event, data)
    elif (data.mode == "helpScreen"):       helpScreenKeyPressed(event, data)
    
def mousePosition(event, data):
    if (data.mode == "startScreen"): startScreenMousePosition(event, data)
    elif (data.mode == "loadDataScreen"):   loadDataScreenMousePosition(event, data)
    elif (data.mode == "featuresScreen"):   featuresScreenMousePosition(event, data)
    elif (data.mode == "analyzeScreen"):   analyzeScreenMousePosition(event, data)
    elif (data.mode == "helpScreen"):       helpScreenMousePosition(event, data)
    
def timerFired(data):
    if (data.mode == "startScreen"): startScreenTimerFired(data)
    elif (data.mode == "loadDataScreen"):   loadDataScreenTimerFired(data)
    elif (data.mode == "featuresScreen"):   featuresScreenTimerFired(data)
    elif (data.mode == "analyzeScreen"):   analyzeScreenTimerFired(data)
    elif (data.mode == "helpScreen"):       helpScreenTimerFired(data)

def redrawAll(canvas, data):
    if (data.mode == "startScreen"): startScreenRedrawAll(canvas, data)
    elif (data.mode == "loadDataScreen"):   loadDataScreenRedrawAll(canvas, data)
    elif (data.mode == "featuresScreen"):   featuresScreenRedrawAll(canvas, data)
    elif (data.mode == "analyzeScreen"):   analyzeScreenRedrawAll(canvas, data)
    elif (data.mode == "helpScreen"):       helpScreenRedrawAll(canvas, data)

## startScreen mode

def startScreenMousePressed(event, data):
    if onStartButton(data, event.x, event.y):
        data.mode = "loadDataScreen"
    elif onHelpButton(data, event.x, event.y):
        data.mode = "helpScreen"

def startScreenMousePosition(event, data):
    if onStartButton(data, event.x, event.y):
        data.btnColor["start"] = HOVER_DARKBLUE
    else:
        data.btnColor["start"] = DARKBLUE
        
    if onHelpButton(data, event.x, event.y):
        data.btnColor["help"] = WHITE
    else:
        data.btnColor["help"] = OFFWHITE
    
def startScreenKeyPressed(event, data):
    pass

def startScreenTimerFired(data):
    pass

def startScreenRedrawAll(canvas, data):
    # draws background and description
    canvas.create_rectangle(0, 0, data.width, data.height, fill=MAINBLUE, outline="")
    
    # draws particles
    drawEdges(data, canvas)
    drawParticles(data, canvas)
    
    # draws text
    canvas.create_text(data.width/2, data.height/3-20,
                       text="Welcome to Labely!", font="Arial 48 bold", fill=WHITE)
    canvas.create_text(data.width/2, data.height/3+40,
                       text="An email labeling and analysis tool for company complaints.", font="Arial 18 bold", fill=WHITE)
    
    # help button
    canvas.create_text(data.width - 10, 20, text="HELP", font="Arial 16 bold", fill=data.btnColor["help"], anchor="e")
    
    # start button
    canvas.create_rectangle(data.width/2-75, data.height*2/3-30, data.width/2+75, data.height*2/3+30, fill=data.btnColor["start"], outline="")
    canvas.create_text(data.width/2, data.height*2/3, text="START", font="Arial 18", fill=WHITE)
    
    
## loadDataScreen mode

def loadDataScreenMousePressed(event, data):
    #make some sort of cool success animation if file is CSV and uploaded
    if onUploadButton(data, event.x, event.y):
        file = filedialog.askopenfilename(initialdir = "/Desktop",title = "Select file",
                                                    filetypes = (("CSV files","*.csv"),("all files","*.")))
        if file != "":
            fileRev = file.split("/")[::-1]
            data.filename = fileRev[1] + "/" + fileRev[0]
    elif onContinueButton(data, event.x, event.y):
        if data.filename != None:
            data.mode = "featuresScreen"
    elif onHelpButton(data, event.x, event.y):
        data.mode = "helpScreen"
    elif onBackButton(data, event.x, event.y):
        data.mode = "startScreen"

def loadDataScreenMousePosition(event, data):
    if onContinueButton(data, event.x, event.y):
        if data.filename == None:
            data.btnColor["continue"] = GREY
        else:
            data.btnColor["continue"] = HOVER_DARKBLUE
    else:
        if data.filename == None:
            data.btnColor["continue"] = GREY
        else:
            data.btnColor["continue"] = DARKBLUE
        
    if onUploadButton(data, event.x, event.y):
        data.btnColor["upload"] = HOVER_DARKBLUE
    else:
        data.btnColor["upload"] = DARKBLUE
    
    if onBackButton(data, event.x, event.y):
        data.btnColor["back"] = WHITE
    else:
        data.btnColor["back"] = OFFWHITE
    
    if onHelpButton(data, event.x, event.y):
        data.btnColor["help"] = WHITE
    else:
        data.btnColor["help"] = OFFWHITE

def loadDataScreenKeyPressed(event, data):
    pass

def loadDataScreenTimerFired(data):
    pass

def loadDataScreenRedrawAll(canvas, data):
    # draws background and instructions
    canvas.create_rectangle(0,0,data.width,data.height, fill=MAINBLUE, outline="")
    canvas.create_rectangle(data.margin, data.margin+75,
                            data.width-data.margin, data.height-data.margin-75,
                            outline=WHITE, dash=(5, 10), fill=MAINBLUE)
    canvas.create_text(data.width/2, data.height/12,
                       text="Press 'upload' to upload your csv of emails.",
                       font="Arial 18 bold", fill=WHITE)
    
    # help button
    canvas.create_text(data.width - 10, 20, text="HELP", font="Arial 16 bold",
                        fill=data.btnColor["help"], anchor="e")
    # back button
    canvas.create_text(10, 20, text="< BACK", font="Arial 16 bold",
                        fill=data.btnColor["back"], anchor="w")
    
    # upload button
    canvas.create_rectangle(data.width/2-75, data.height/2-30, data.width/2+75,
                            data.height/2+30, fill=data.btnColor["upload"], outline="")
    canvas.create_text(data.width/2, data.height/2,
                       text="UPLOAD", font="Arial 18", fill=WHITE)
                       
    # File upload text
    if data.filename != None:
        canvas.create_text(data.width/2, data.height/2-35, text=data.filename,
                            font="Arial 16", fill=WHITE, anchor="s")
                 
    # continue button
    canvas.create_rectangle(data.width/2-85, data.height*11/12-30,
                            data.width/2+85, data.height*11/12+30,
                            fill=data.btnColor["continue"], outline="")
    canvas.create_text(data.width/2, data.height*11/12,
                       text="CONTINUE", font="Arial 18", fill=WHITE)

## featuresScreen mode

def featureBounds(w, h, x, y):
    if (x > w - 20 and x < w + 20 and y > h - 20 and y < h + 20):
            return True
    return False

def featuresScreenMousePressed(event, data):
    if onContinueButton(data, event.x, event.y):
        data.analysisInProgress = True
    elif onHelpButton(data, event.x, event.y):
        data.mode = "helpScreen"
    elif onBackButton(data, event.x, event.y):
        data.mode = "loadDataScreen"
        
    w1, w2 = data.width/4, data.width*3/5
    startH = data.height/6
    
    # features
    if featureBounds(w1, startH*2, event.x, event.y):
        data.features['labels'][0] = not data.features['labels'][0]
        data.features['labels'][1] = OFFWHITE if data.features['labels'][1] == DARKBLUE else DARKBLUE
    elif featureBounds(w1, startH*3, event.x, event.y):
        data.features['freqdist'][0] = not data.features['freqdist'][0]
        data.features['freqdist'][1] = OFFWHITE if data.features['freqdist'][1] == DARKBLUE else DARKBLUE
    elif featureBounds(w1, startH*4, event.x, event.y):
        data.features['netdiagram'][0] = not data.features['netdiagram'][0]
        data.features['netdiagram'][1] = OFFWHITE if data.features['netdiagram'][1] == DARKBLUE else DARKBLUE
    elif featureBounds(w2, startH*2, event.x, event.y):
        data.features['summary'][0] =  not data.features['summary'][0]
        data.features['summary'][1] = OFFWHITE if data.features['summary'][1] == DARKBLUE else DARKBLUE
    elif featureBounds(w2, startH*3, event.x, event.y):
        data.features['sentiment'][0] = not data.features['sentiment'][0]
        data.features['sentiment'][1] = OFFWHITE if data.features['sentiment'][1] == DARKBLUE else DARKBLUE
    elif featureBounds(w2, startH*4, event.x, event.y):
        data.features['exportCSV'][0] = not data.features['exportCSV'][0]
        data.features['exportCSV'][1] = OFFWHITE if data.features['exportCSV'][1] == DARKBLUE else DARKBLUE

def featuresScreenMousePosition(event, data):
    w1, w2 = data.width/4, data.width*3/5
    startH = data.height/6
    
    # continue button hover
    if onContinueButton(data, event.x, event.y):
        data.btnColor["continue"] = HOVER_DARKBLUE
    else:
        data.btnColor["continue"] = DARKBLUE
    
    # back button hover
    if onBackButton(data, event.x, event.y):
        data.btnColor["back"] = WHITE
    else:
        data.btnColor["back"] = OFFWHITE
    
    # help button hover
    if onHelpButton(data, event.x, event.y):
        data.btnColor["help"] = WHITE
    else:
        data.btnColor["help"] = OFFWHITE
    
def featuresScreenKeyPressed(event, data):
    pass

def featuresScreenTimerFired(data):
    # if cannot complete analysis, direct user to load screen
    if data.analysisInProgress:
        try:
            data.results = a.analyze(data.filename, data.features)
            data.analysisInProgress = False
            data.mode = "analyzeScreen"
        except Exception as e:
            print(e)
            data.analysisInProgress = False
            data.mode = "loadDataScreen"

def featuresScreenRedrawAll(canvas, data):
    # draws background and instructions
    canvas.create_rectangle(0,0,data.width,data.height, fill=MAINBLUE, outline="")
    
    w1, w2 = data.width/4, data.width*3/5
    startH = data.height/6
    
    if data.analysisInProgress:
        drawEdges(data, canvas)
        drawParticles(data, canvas)
        canvas.create_text(data.width/2, data.height/2-10,
                            text="Analyzing... Please wait.",
                            font="Arial 36", fill=WHITE)
        canvas.create_text(data.width/2, data.height/2+30,
                            text="(This process may take a while.)",
                            font="Arial 12", fill=RED)
    else:
        canvas.create_text(data.width/2, data.height/12,
                            text="Which features do you want to view?",
                            font="Arial 18 bold", fill=WHITE)
        # feature1: labels
        canvas.create_rectangle(w1-20, startH*2-20, w1+20, startH*2+20,
                                fill=data.features['labels'][1], outline="")
        canvas.create_text(w1 + 30, startH*2, text="Suggested Labels",
                                font="Arial 16", fill=WHITE, anchor="w")
        if data.features['labels'][0]:
            canvas.create_text(w1 - 20, startH*2+30,
                                text="*Individual labels only viewable in CSV export.",
                                font="Arial 8", fill=RED, anchor="w")
        
        # feature2: word frequency
        canvas.create_rectangle(w1-20, startH*3-20, w1+20, startH*3+20,
                                fill=data.features['freqdist'][1], outline="")
        canvas.create_text(w1 + 30, startH*3, text="Word Frequencies",
                            font="Arial 16", fill=WHITE, anchor="w")
        
        # feature3: network diagram
        canvas.create_rectangle(w1-20, startH*4-20, w1+20, startH*4+20,
                                fill=data.features['netdiagram'][1], outline="")
        canvas.create_text(w1 + 30, startH*4, text="Network Diagram",
                            font="Arial 16", fill=WHITE, anchor="w")
        
        # feature4: summarization
        canvas.create_rectangle(w2-20, startH*2-20, w2+20, startH*2+20,
                                fill=data.features['summary'][1], outline="")
        canvas.create_text(w2 + 30, startH*2, text="Summarization",
                                font="Arial 16", fill=WHITE, anchor="w")
        if data.features['summary'][0]:
            canvas.create_text(w2 - 20, startH*2+30,
                                text="*Only viewable in CSV export.",
                                font="Arial 8", fill=RED, anchor="w")
        
        # feature5: sentiment analysis
        canvas.create_rectangle(w2-20, startH*3-20, w2+20, startH*3+20,
                                fill=data.features['sentiment'][1], outline="")
        canvas.create_text(w2 + 30, startH*3, text="Sentiment Analysis",
                                font="Arial 16", fill=WHITE, anchor="w")
        
        # feature6: csv export
        canvas.create_rectangle(w2-20, startH*4-20, w2+20, startH*4+20,
                                fill=data.features['exportCSV'][1], outline="")
        canvas.create_text(w2 + 30, startH*4, text="Export to CSV",
                                font="Arial 16", fill=WHITE, anchor="w")
        canvas.create_text(w2 - 20, startH*4+30, text="*Recommended.",
                                font="Arial 8", fill=RED, anchor="w")
        
        # help button
        canvas.create_text(data.width - 10, 20, text="HELP",
                            font="Arial 16 bold", fill=data.btnColor["help"],
                            anchor="e")
        # back button
        canvas.create_text(10, 20, text="< BACK", font="Arial 16 bold",
                            fill=data.btnColor["back"], anchor="w")
        
        # continue button
        canvas.create_rectangle(data.width/2-85, data.height*11/12-30,
                                data.width/2+85, data.height*11/12+30,
                                fill=data.btnColor["continue"], outline="")
        canvas.create_text(data.width/2, data.height*11/12,
                        text="CONTINUE", font="Arial 18", fill=WHITE)

## analyzeScreen mode

def analyzeScreenMousePressed(event, data):
    if onBTSButton(data, event.x, event.y):
        data.mode = "startScreen"
    elif onHelpButton(data, event.x, event.y):
        data.mode = "helpScreen"
    elif onBackButton(data, event.x, event.y):
        data.mode = "featuresScreen"
        
    if data.features['labels'][0] and onLabelsButton(data, event.x, event.y):
        a.graphLabels(data.results['labels'][1])
    if data.features['freqdist'][0] and onFreqDistButton(data, event.x, event.y):
        a.plotFrequencyDist(data.results['freqdist'][1])
    if data.features['netdiagram'][0] and onNetDiagramButton(data, event.x, event.y):
        a.graphNetDiagram(data.results['netdiagram'])
    if data.features['summary'][0] and onSummarizationButton(data, event.x, event.y):
        messagebox.showinfo("NOTICE", "Summary Completed. View individual summaries in the CSV export of data located in the data/out.csv.")
    if data.features['sentiment'][0] and onSentimentButton(data, event.x, event.y):
        a.graphSentiment(data.results['sentiment'])
    if data.features['exportCSV'][0] and onExportCSVButton(data, event.x, event.y):
        messagebox.showinfo("NOTICE", "Export Completed. View your exported CSV of data located in the data/out.csv.")

def analyzeScreenMousePosition(event, data):
    # continue button hover
    if onContinueButton(data, event.x, event.y):
        data.btnColor["bts"] = HOVER_DARKBLUE
    else:
        data.btnColor["bts"] = DARKBLUE
    
    # back button hover
    if onBackButton(data, event.x, event.y):
        data.btnColor["back"] = WHITE
    else:
        data.btnColor["back"] = OFFWHITE
    
    # help button hover
    if onHelpButton(data, event.x, event.y):
        data.btnColor["help"] = WHITE
    else:
        data.btnColor["help"] = OFFWHITE
    
    # feature button hovers
    if data.features['labels'][0]:
        if onLabelsButton(data, event.x, event.y):
            data.btnColor["labels"] = HOVER_DARKBLUE
        else:
            data.btnColor["labels"] = DARKBLUE
    if data.features['freqdist'][0]:
        if onFreqDistButton(data, event.x, event.y):
            data.btnColor["freqdist"] = HOVER_DARKBLUE
        else:
            data.btnColor["freqdist"] = DARKBLUE
    if data.features['netdiagram'][0]:
        if onNetDiagramButton(data, event.x, event.y):
            data.btnColor["netdiagram"] = HOVER_DARKBLUE
        else:
            data.btnColor["netdiagram"] = DARKBLUE
    if data.features['summary'][0]:
        if onSummarizationButton(data, event.x, event.y):
            data.btnColor["summary"] = HOVER_DARKBLUE
        else:
            data.btnColor["summary"] = DARKBLUE
    if data.features['sentiment'][0]:
        if onSentimentButton(data, event.x, event.y):
            data.btnColor["sentiment"] = HOVER_DARKBLUE
        else:
            data.btnColor["sentiment"] = DARKBLUE
    if data.features['exportCSV'][0]:
        if onExportCSVButton(data, event.x, event.y):
            data.btnColor["exportCSV"] = HOVER_DARKBLUE
        else:
            data.btnColor["exportCSV"] = DARKBLUE

def analyzeScreenKeyPressed(event, data):
    pass
    
def analyzeScreenTimerFired(data):
    if data.analysisInProgress:
        data.analysisInProgress = False

def analyzeScreenRedrawAll(canvas, data):
    # background and screen title
    canvas.create_rectangle(0,0,data.width,data.height, fill=MAINBLUE, outline="")
    canvas.create_text(data.width/2, 20, text="ANALYSIS",
                        font="Arial 20 bold", fill=WHITE, anchor="n")
    
    if data.features['labels'][0]:
        drawLabels(canvas, data)
        drawLabelsButton(canvas, data)
    if data.features['freqdist'][0]:
        drawFreqWords(canvas, data)
        drawFreqDistButton(canvas, data)
    if data.features['netdiagram'][0]:
        drawNetworkDiagram(canvas, data)
    if data.features['summary'][0]:
        drawSummarization(canvas, data)
    if data.features['sentiment'][0]:
        drawSentimentAnalysis(canvas, data)
    if data.features['exportCSV'][0]:
        drawExportCSVButton(canvas, data)
    
    # help button
    canvas.create_text(data.width - 10, 20, text="HELP", font="Arial 16 bold",
                        fill=data.btnColor["help"], anchor="e")
    # back button
    canvas.create_text(10, 20, text="< BACK", font="Arial 16 bold",
                        fill=data.btnColor["back"], anchor="w")
    # bts button
    canvas.create_rectangle(data.width/2-90, data.height*11/12-30,
                            data.width/2+90, data.height*11/12+30,
                            fill=data.btnColor["bts"], outline="")
    canvas.create_text(data.width/2, data.height*11/12,
                        text="BACK TO START", font="Arial 18", fill=WHITE)

# draws labels list on analysis screen
def drawLabels(canvas, data):
    canvas.create_text(data.width/12, data.height*2/10 - 25, text="Labels",
                        font="Arial 20 bold", fill=WHITE, anchor="sw")
    for i in range(len(data.results['labels'][0])):
        canvas.create_text(data.width/12, data.height*2/10 + 25*i,
                            text=str(i+1) + ". " + data.results['labels'][0][i],
                            font="Arial 12", fill=WHITE, anchor="w")
    
# draws frequency list on analysis screen
def drawFreqWords(canvas, data):
    canvas.create_text(data.width*4/12, data.height*2/10 - 25,
                        text="Top Words", font="Arial 20 bold",
                        fill=WHITE, anchor="sw")
    for i in range(len(data.results['freqdist'][0])):
        canvas.create_text(data.width*4/12, data.height*2/10 + 25*i,
                       text=("%d %s %-20s") % (i+1, ".", data.results['freqdist'][0][i][0]),
                       font="Arial 12", fill=WHITE, anchor="w")
        canvas.create_text(data.width*7/12-10, data.height*2/10 + 25*i,
                       text=("%d") % (data.results['freqdist'][0][i][1]),
                       font="Arial 12", fill=WHITE, anchor="e")
    
def drawSentimentAnalysis(canvas, data):
    canvas.create_rectangle(data.width*9/12-90, data.height*4/20-30,
                            data.width*9/12+90, data.height*4/20+30,
                            fill=data.btnColor["sentiment"], outline="")
    canvas.create_text(data.width*9/12, data.height*4/20,
                       text="Sentiment Analysis", font="Arial 18", fill=WHITE)
    
def drawNetworkDiagram(canvas, data):
    canvas.create_rectangle(data.width*9/12-90, data.height*10/20-6,
                            data.width*9/12+90, data.height*10/20+54,
                            fill=data.btnColor["netdiagram"], outline="")
    canvas.create_text(data.width*9/12, data.height*10/20+24,
                       text="Network Diagram", font="Arial 18", fill=WHITE)
    
def drawSummarization(canvas, data):
    canvas.create_rectangle(data.width*9/12-90, data.height*7/20-18,
                            data.width*9/12+90, data.height*7/20+42,
                            fill=data.btnColor["summary"], outline="")
    canvas.create_text(data.width*9/12, data.height*7/20+12,
                       text="Summarization", font="Arial 18", fill=WHITE)
    
def drawExportCSVButton(canvas, data):
    canvas.create_rectangle(data.width*9/12-90, data.height*13/20+6,
                            data.width*9/12+90, data.height*13/20+66,
                            fill=data.btnColor["exportCSV"], outline="")
    canvas.create_text(data.width*9/12, data.height*13/20+36,
                       text="Download CSV", font="Arial 18", fill=WHITE)

def drawLabelsButton(canvas, data):
    canvas.create_rectangle(data.width/12, data.height*13/20+6,
                            data.width/12+152, data.height*13/20+66,
                            fill=data.btnColor["labels"], outline="")
    canvas.create_text(data.width/12+76, data.height*13/20+36,
                       text="Labels Graph", font="Arial 18", fill=WHITE)

def drawFreqDistButton(canvas, data):
    canvas.create_rectangle(data.width*4/12, data.height*13/20+6,
                            data.width*4/12+180, data.height*13/20+66,
                            fill=data.btnColor["freqdist"], outline="")
    canvas.create_text(data.width*4/12+90, data.height*13/20+36,
                       text="Top Words Graph", font="Arial 18", fill=WHITE)
    
# feature button hovers
def onLabelsButton(data, x, y):
    if (x > data.width/12 and x < data.width/12+152 and
        y > data.height*13/20+6 and y < data.height*13/20+66):
        return True
    return False

def onFreqDistButton(data, x, y):
    if (x > data.width*4/12 and x < data.width*4/12+180 and
        y > data.height*13/20+6 and y < data.height*13/20+66):
        return True
    return False

def onNetDiagramButton(data, x, y):
    if (x > data.width*9/12-90 and x < data.width*9/12+90 and
        y > data.height*10/20-6 and y < data.height*10/20+54):
            return True
    return False
    
def onSummarizationButton(data, x, y):
    if (x > data.width*9/12-90 and x < data.width*9/12+90 and
        y > data.height*7/20-18 and y < data.height*7/20+42):
            return True
    return False

def onSentimentButton(data, x, y):
    if (x > data.width*9/12-90 and x < data.width*9/12+90 and
        y > data.height*4/20-30 and y < data.height*4/20+30):
            return True
    return False

def onExportCSVButton(data, x, y):
    if (x > data.width*9/12-90 and x < data.width*9/12+90 and
        y > data.height*13/20+6 and y < data.height*13/20+66):
            return True
    return False


## helpScreen mode

def helpScreenMousePressed(event, data):
    if onBTSButton(data, event.x, event.y):
        data.mode = "startScreen"

def helpScreenMousePosition(event, data):
    if onContinueButton(data, event.x, event.y):
        data.btnColor["bts"] = HOVER_DARKBLUE
    else:
        data.btnColor["bts"] = DARKBLUE

def helpScreenKeyPressed(event, data):
    pass

def helpScreenTimerFired(data):
    pass

def helpScreenRedrawAll(canvas, data):
    # background and screen title
    canvas.create_rectangle(0,0,data.width,data.height, fill=MAINBLUE, outline="")
    drawEdges(data, canvas)
    drawParticles(data, canvas)
    canvas.create_text(data.width/2, 40,
                       text="How to Use Labely", font="Arial 26 bold", fill=WHITE, anchor="n")
    
    instructions = ["Press start.",
                    "Click on 'upload' to select a csv file containing your email export from Gmail.",
                    "Select the features you'd like to analyze. Some features are only viewable in the CSV export.",
                    "Wait for the analysis to complete...",
                    "...and voila! See results and click on the buttons see visualizations!",
                    "Navigate visualizations with the tools in the bottom left corner.",
                    "To find your CSV export, navigate to the data folder inside the project files and look for 'out.csv'."]
    # print off instructions
    for i in range(len(instructions)):
        canvas.create_text(data.width/12, data.height*1/5 + 50*i, text=str(i+1) + ". " + instructions[i],
                                font="Arial 14", fill=WHITE, anchor = "nw")
                       
    # back to start button
    canvas.create_rectangle(data.width/2-90, data.height*11/12-30, data.width/2+90, data.height*11/12+30, fill=data.btnColor["bts"], outline="")
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
        
    def mousePositionWrapper(event, canvas, data):
        mousePosition(event, data)
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
    data.timerDelay = 50 # millisecondsg
    
    root = Tk()
    root.wm_title("Labely - Email Labeling & Text Analysis")
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
    root.bind("<Motion>", lambda event:
                            mousePositionWrapper(event, canvas, data))
    timerFiredWrapper(canvas, data)
    
    # and launch the app
    root.mainloop()  # blocks until window is closed
    print("bye!")

run(720, 580)