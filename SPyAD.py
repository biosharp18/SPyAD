#Finished 2020-08-11! Reach me at rory.gao@mail.utoronto.ca for any problems with SPyAD.py
import cv2, pytesseract, re, numpy as np, argparse, csv, os

pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

parser = argparse.ArgumentParser()
parser.add_argument("-img", "--i", help="image file")
parser.add_argument("-bulk", "--b", help = "name of subfolder with data to be processed")
parser.add_argument("-preview", "--p", action='store_true',help = "toggle image display preview")
parser.add_argument("-threshold", "--t", default=-50, help = "set threshold value")
parser.add_argument("-contoursize", "--c", default=4, help = "set minimum contour size")
args = parser.parse_args()
files = os.listdir(args.b)
#python wants me to make global variables outside of functions
whiteline2 = True
conversionfactor = True

def OCR(img):

    pic = cv2.imread(img)
        
    def upscanner(x, y, value):
        while True:
            pixel = pic[x,y]
            if pixel[0] == value:
                    return(x - 1)
                    break
            x += -1

    def leftscanner(x, y, value):
        while True:
            pixel = pic[x,y]
            if pixel[0] == value:
                    return(y - 1)
                    break
            y += -1

    def leftscannerblack(x, y):
        while True:
            pixel = pic[x,y]
            if  127 > pixel[0] >= 0:
                    return(y - 1)
                    break
            y += -1



    xstart = pic.shape[0] - 2 
    ystart = pic.shape[1] - 2

    whiteline1 = upscanner(xstart, ystart, 252)
    distance = xstart - whiteline1
    global whiteline2
    whiteline2 = upscanner(whiteline1, ystart, 252)
    scalebox = leftscanner(whiteline1, ystart, 252)
    scaleboxwidth = ystart - scalebox
    scalebarpos = int(round(((whiteline1 + whiteline2)/2), 0))
    barspace = leftscanner(scalebarpos, ystart, 252)
    barspacewidth = ystart - barspace

    scalebar = scaleboxwidth - 2 * barspacewidth

    #Get dimensions of the OCR box

    OCRboxrightbound = leftscannerblack(scalebarpos, barspace)

    OCRboxleftbound = barspace - (scalebar - (barspace - OCRboxrightbound))
   
    crop = pic[whiteline2:whiteline1, OCRboxleftbound:OCRboxrightbound]
    
    crop = cv2.resize(crop,(0,0),fx=5,fy=5)
    OCRtext = pytesseract.image_to_string(crop, lang='eng')
    scaledistance = int(((re.findall('\d+', OCRtext))[0]))
    for text in OCRtext:
        global conversionfactor
        if text == "u":
            conversionfactor = 1000 * int(scaledistance) / scalebar 
            print("Pixel > Distance conversion factor:", conversionfactor)
        if text == "n":
            conversionfactor = 1 * int(scaledistance) / scalebar
            print("Pixel to Distance conversion factor:", conversionfactor)





def getcoords(img):


    # now begins the image processing pipeline
    pic = cv2.imread(img)
    image = pic[0:whiteline2, 0:pic.shape[1]]#####################################
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image2 = cv2.imread(img)
    clahe = cv2.createCLAHE(clipLimit = 50, tileGridSize=(25,25)) 
    image = clahe.apply(image)
    image = cv2.GaussianBlur(image,(5,5),cv2.BORDER_DEFAULT)
    #sharpen image, create sharpening filter
    filter = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
    # Applying cv2.filter2D function on our image
    image=cv2.filter2D(image,-1,filter)
    #median blur
    image = cv2.medianBlur(image,5)
    #adaptive threshold
    image = cv2.adaptiveThreshold(image,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,75,int(args.t))
    image = cv2.threshold(image,5,25, cv2.THRESH_BINARY)[1]
    cnts = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]


    #min_area = 4
    white_dots = []
    if args.b is not None:
        directory = os.path.join(os.getcwd(), args.b)
        filename = os.path.join(directory, os.path.splitext(img)[0] + ".csv")
    else:
        filename = os.path.splitext(args.i)[0] + ".csv"
    with open(filename,'w', newline='') as file:
            writer = csv.writer(file)
            for c in cnts:
                area = cv2.contourArea(c)
                if area > int(args.c):
                    cv2.drawContours(image, [c], -1, (36, 255, 12), 2)
                    (x,y),radius = cv2.minEnclosingCircle(c)
                    center = (int(x),int(y))
                    radius = int(radius)
                    circle = cv2.circle(pic, center,0,(0,255,255),1)
                    coords = [x, y]
                    coordsdistance = [conversionfactor * i for i in coords]
                    writer.writerow(coordsdistance)
                    white_dots.append(c)
    print("Particles detected:", len(white_dots))
    if args.p is True:
        cv2.imshow('Preview', circle)
        cv2.waitKey(0)
    




if args.b is not None:
    for file in files:
        print(file)
        OCR(file)
        getcoords(file)
else:
    print(args.i)
    OCR(args.i)
    getcoords(args.i)







