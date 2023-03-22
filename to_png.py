from PIL import Image
import os

for i in range(6):
    folders = ["/home/nikodem/hand_detection/tmp", "/home/nikodem/hand_detection/dislike", "/home/nikodem/hand_detection/fist", "/home/nikodem/hand_detection/like", "/home/nikodem/hand_detection/palm", "/home/nikodem/hand_detection/peace",]
    directory = folders[i]
    c=0
    for filename in os.listdir(directory):
        if filename.endswith(".jpg"):
            im = Image.open(directory + "/" + filename)
            name = str(c) + '.png'
            rgb_im = im.convert('RGB')
            name = directory + "1" + "/" + name
            os.remove(directory + "/" + filename) 
            rgb_im.save(name)
            file_list = '"' + name + '"'
            c+=1
            #print(os.path.join(directory, filename))
            print(file_list)
            continue
        else:
            continue
    print(len(os.listdir(directory)))