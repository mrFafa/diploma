from cv2 import cv2
import numpy as np
import pandas as pd
import os

def resize_image(file_path, img, new_size=300): 
    hs=new_size/img.shape[0]
    ws=new_size/img.shape[1]
    ss= img.shape[0]/img.shape[1]

    if hs>ws:
        scale = hs
    else:
        scale = ws

    dim = (int(img.shape[0]*scale), int(img.shape[0]*scale*ss))
    resized = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
    #cv2.imwrite("./obr/"+file_path[:-4]+".resized.jpg", resized)
    return resized

colorsHSV = [(35,14.4,46.3), (240,70.3,35.7), (170,48.6,28.2), (11,84.3,87.5), (52,75.8,95.7), (354,77.6,61.2), (24,77,68.2), (314,61.9,8.2)]

temp = []
for i in range(8):
    h = int(round(colorsHSV[i][0]*180/360, 0))
    s = int(round(colorsHSV[i][1]*255/100, 0))
    v = int(round(colorsHSV[i][2]*255/100, 0))
    temp.append((h,s,v))
colorsHSV = temp

def doAnything(file_path):
    import warnings
    warnings.filterwarnings("ignore")
    # собираем всё во едино 
    ### читаем картинку и переводит её во фрейм     
    image = cv2.imread(file_path)
    image = resize_image(file_path, image)
    
    #pixel_colors = image.reshape((np.shape(image)[0]*np.shape(image)[1], 3))
    #norm = colors.Normalize(vmin=-1.,vmax=1.)
    #norm.autoscale(pixel_colors)
    #pixel_colors = norm(pixel_colors).tolist()
    
    
    image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
    
    #h, s, v = cv2.split(image)
    #fig = plt.figure()
    #axis = fig.add_subplot(1, 1, 1, projection="3d")

    #axis.scatter(h.flatten(), s.flatten(), v.flatten(), facecolors=pixel_colors, marker=".")
    #axis.set_xlabel("Hue")
    #axis.set_ylabel("Saturation")
    #axis.set_zlabel("Value")
    #plt.show()
    
    size = image.shape

    img = np.reshape(image, (1, -1, 3))[0]
    df = pd.DataFrame(img)

    resultCount = np.zeros(8)

    ### предобработаем палитру цветов и найдем интервалы
    s_clr = pd.DataFrame(colorsHSV)
    s_clr.sort_values(by = [0], axis = 0, inplace = True)

    ########################################################

    ### отсекаем черный цвет
    #black = df[df[2] <= (df[df[2]<=85].mode()[2][0])]
    #df = df[df[2] > df[df[2]<=85].mode()[2][0]]
    bo = 36
    #black = df[df[2] <= df[df[2]<=bo][2].median()]
    #df = df[df[2] > df[df[2]<=bo][2].median()]
    black = df[df[2] <= bo]
    df = df[df[2] > bo]

    resultCount[7] += black.shape[0]
    black.loc[:,[0, 1, 2]] = s_clr[s_clr.index==7][[0, 1, 2]].to_numpy()[0]


    ### отсекаем серый цвет
    go = 36
    #grey = df[df[1] <= np.sqrt(3.5*(255-df[2]))]
    #df = df[df[1] > np.sqrt(3.5*(255-df[2]))]
    grey = df[df[1] <= go]
    df = df[df[1] > go]


    resultCount[0] += grey.shape[0]
    grey.loc[:,[0, 1, 2]] = s_clr[s_clr.index==0][[0, 1, 2]].to_numpy()[0]
    ########################################################

    s_clr.drop([0,7], inplace =True)

    n_clr = s_clr[0].to_numpy()
    n_clr.shape[0]

    dif = np.array([n_clr[1]-n_clr[0]])
    for i in range(n_clr.shape[0]-2):
        dif = np.append(dif, [n_clr[i+2]-n_clr[i+1]])
    dif = np.append(dif, 180-n_clr[5]+n_clr[0])


    if(n_clr[0]-(0.85*dif[5]/2) < -0.5 ):
        st = np.around(np.array([180+n_clr[0]-(0.85*dif[5]/2)]))
    else:    
        st = np.around(np.array([n_clr[0]-(0.85*dif[5]/2)]))

    fin = np.around(np.array([n_clr[0]+(0.85*dif[0]/2)]))

    for i in range(n_clr.shape[0]-1):
        st = np.append(st, [np.around(np.array([n_clr[i+1]-(0.85*dif[i]/2)]))])
        fin = np.append(fin, [np.around(np.array([n_clr[i+1]+(0.85*dif[i+1]/2)]))%180])
        
    #st = np.array([175., 18., 26., 35., 89., 129.])
    #fin = np.array([17., 25., 30., 72., 120., 168.])

    s_clr['st'], s_clr['f'] = st, fin

    ### зададим точно определенные цвета
    s_ind = s_clr.index.to_numpy()
    for index, row in s_clr.iterrows():
        itind = (np.argwhere(s_ind==index))[0][0] 

        #-------------------------------------------------------
        if index == 1:
            if(row['st']>row['f']):
                blue = pd.concat([df[(df[0]>=row['st'])], df[(df[0]<row['f'])]], axis=0)
            else:
                blue = df[(df[0]>=row['st']) & (df[0]<row['f'])]

            if(fin[itind]>st[(itind+1)%6]):
                semiblue = pd.concat([df[(df[0]>=fin[itind])], df[(df[0]<st[(itind+1)%6])]], axis=0)
            else:
                semiblue = df[(df[0]>=fin[itind]) & (df[0]<st[(itind+1)%6])]

            blues = semiblue[((semiblue[0]-s_clr[s_clr.index==index][0].to_numpy()[0])**2 + \
                         (semiblue[1]-s_clr[s_clr.index==index][1].to_numpy()[0])**2 + \
                         (semiblue[2]-s_clr[s_clr.index==index][2].to_numpy()[0])**2) <\
                        ((semiblue[0]-s_clr[s_clr.index==s_ind[(itind+1)%6]][0].to_numpy()[0])**2 + \
                         (semiblue[1]-s_clr[s_clr.index==s_ind[(itind+1)%6]][1].to_numpy()[0])**2 + \
                         (semiblue[2]-s_clr[s_clr.index==s_ind[(itind+1)%6]][2].to_numpy()[0])**2)]
            nonBlue = semiblue[(semiblue.isin(blues))[0]==False] 

            resultCount[index] += (blue.shape[0] + blues.shape[0])
            resultCount[s_ind[(itind+1)%6]] += nonBlue.shape[0]        
            blue.loc[:,[0, 1, 2]] = s_clr[s_clr.index==index][[0, 1, 2]].to_numpy()[0]
            blues.loc[:,[0, 1, 2]] = s_clr[s_clr.index==index][[0, 1, 2]].to_numpy()[0]
            nonBlue.loc[:,[0, 1, 2]] = s_clr[s_clr.index==s_ind[(itind+1)%6]][[0, 1, 2]].to_numpy()[0]
            #df = df[(df.isin(blue))[0]==False]

        #-------------------------------------------------------
        elif index == 2:
            if(row['st']>row['f']):
                green = pd.concat([df[(df[0]>=row['st'])], df[(df[0]<row['f'])]], axis=0)
            else:
                green = df[(df[0]>=row['st']) & (df[0]<row['f'])]

            if(fin[itind]>st[(itind+1)%6]):
                semigreen = pd.concat([df[(df[0]>=fin[itind])], df[(df[0]<st[(itind+1)%6])]], axis=0)
            else:
                semigreen = df[(df[0]>=fin[itind]) & (df[0]<st[(itind+1)%6])]

            greens = semigreen[((semigreen[0]-s_clr[s_clr.index==index][0].to_numpy()[0])**2 + \
                         (semigreen[1]-s_clr[s_clr.index==index][1].to_numpy()[0])**2 + \
                         (semigreen[2]-s_clr[s_clr.index==index][2].to_numpy()[0])**2) <\
                        ((semigreen[0]-s_clr[s_clr.index==s_ind[(itind+1)%6]][0].to_numpy()[0])**2 + \
                         (semigreen[1]-s_clr[s_clr.index==s_ind[(itind+1)%6]][1].to_numpy()[0])**2 + \
                         (semigreen[2]-s_clr[s_clr.index==s_ind[(itind+1)%6]][2].to_numpy()[0])**2)]
            nonGreen = semigreen[(semigreen.isin(greens))[0]==False] 

            resultCount[index] += (green.shape[0] + greens.shape[0])
            resultCount[s_ind[(itind+1)%6]] += nonGreen.shape[0]
            green.loc[:,[0, 1, 2]] = s_clr[s_clr.index==index][[0, 1, 2]].to_numpy()[0]
            greens.loc[:,[0, 1, 2]] = s_clr[s_clr.index==index][[0, 1, 2]].to_numpy()[0]
            nonGreen.loc[:,[0, 1, 2]] = s_clr[s_clr.index==s_ind[(itind+1)%6]][[0, 1, 2]].to_numpy()[0]
            #df = df[(df.isin(green))[0]==False]

        #-------------------------------------------------------
        elif index == 3:
            if(row['st']>row['f']):
                red = pd.concat([df[(df[0]>=row['st'])], df[(df[0]<row['f'])]], axis=0)
            else:
                red = df[(df[0]>=row['st']) & (df[0]<row['f'])]

            if(fin[itind]>st[(itind+1)%6]):
                semired = pd.concat([df[(df[0]>=fin[itind])], df[(df[0]<st[(itind+1)%6])]], axis=0)
            else:
                semired = df[(df[0]>=fin[itind]) & (df[0]<st[(itind+1)%6])]

            reds = semired[((semired[0]-s_clr[s_clr.index==index][0].to_numpy()[0])**2 + \
                         (semired[1]-s_clr[s_clr.index==index][1].to_numpy()[0])**2 + \
                         (semired[2]-s_clr[s_clr.index==index][2].to_numpy()[0])**2) <\
                        ((semired[0]-s_clr[s_clr.index==s_ind[(itind+1)%6]][0].to_numpy()[0])**2 + \
                         (semired[1]-s_clr[s_clr.index==s_ind[(itind+1)%6]][1].to_numpy()[0])**2 + \
                         (semired[2]-s_clr[s_clr.index==s_ind[(itind+1)%6]][2].to_numpy()[0])**2)]
            nonRed = semired[(semired.isin(reds))[0]==False] 

            resultCount[index] += (red.shape[0] + reds.shape[0])
            resultCount[s_ind[(itind+1)%6]] += nonRed.shape[0]
            red.loc[:,[0, 1, 2]] = s_clr[s_clr.index==index][[0, 1, 2]].to_numpy()[0]
            reds.loc[:,[0, 1, 2]] = s_clr[s_clr.index==index][[0, 1, 2]].to_numpy()[0]
            nonRed.loc[:,[0, 1, 2]] = s_clr[s_clr.index==s_ind[(itind+1)%6]][[0, 1, 2]].to_numpy()[0]
            #df = df[(df.isin(red))[0]==False]

        #-------------------------------------------------------
        elif index == 4:
            if(row['st']>row['f']):
                yellow = pd.concat([df[(df[0]>=row['st'])], df[(df[0]<row['f'])]], axis=0)
            else:
                yellow = df[(df[0]>=row['st']) & (df[0]<row['f'])]

            if(fin[itind]>st[(itind+1)%6]):
                semiyellow = pd.concat([df[(df[0]>=fin[itind])], df[(df[0]<st[(itind+1)%6])]], axis=0)
            else:
                semiyellow = df[(df[0]>=fin[itind]) & (df[0]<st[(itind+1)%6])]

            yellows = semiyellow[((semiyellow[0]-s_clr[s_clr.index==index][0].to_numpy()[0])**2 + \
                         (semiyellow[1]-s_clr[s_clr.index==index][1].to_numpy()[0])**2 + \
                         (semiyellow[2]-s_clr[s_clr.index==index][2].to_numpy()[0])**2) <\
                        ((semiyellow[0]-s_clr[s_clr.index==s_ind[(itind+1)%6]][0].to_numpy()[0])**2 + \
                         (semiyellow[1]-s_clr[s_clr.index==s_ind[(itind+1)%6]][1].to_numpy()[0])**2 + \
                         (semiyellow[2]-s_clr[s_clr.index==s_ind[(itind+1)%6]][2].to_numpy()[0])**2)]
            nonYellow = semiyellow[(semiyellow.isin(yellows))[0]==False] 

            resultCount[index] += (yellow.shape[0] + yellows.shape[0])
            resultCount[s_ind[(itind+1)%6]] += nonYellow.shape[0]
            yellow.loc[:,[0, 1, 2]] = s_clr[s_clr.index==index][[0, 1, 2]].to_numpy()[0]
            yellows.loc[:,[0, 1, 2]] = s_clr[s_clr.index==index][[0, 1, 2]].to_numpy()[0]
            nonYellow.loc[:,[0, 1, 2]] = s_clr[s_clr.index==s_ind[(itind+1)%6]][[0, 1, 2]].to_numpy()[0]
            #df = df[(df.isin(yellow))[0]==False]

        #-------------------------------------------------------
        elif index == 5:
            if(row['st']>row['f']):
                purple = pd.concat([df[(df[0]>=row['st'])], df[(df[0]<row['f'])]], axis=0)
            else:
                purple = df[(df[0]>=row['st']) & (df[0]<row['f'])]

            if(fin[itind]>st[(itind+1)%6]):
                semipurple = pd.concat([df[(df[0]>=fin[itind])], df[(df[0]<st[(itind+1)%6])]], axis=0)
            else:
                semipurple = df[(df[0]>=fin[itind]) & (df[0]<st[(itind+1)%6])]

            purples = semipurple[((semipurple[0]-s_clr[s_clr.index==index][0].to_numpy()[0])**2 + \
                         (semipurple[1]-s_clr[s_clr.index==index][1].to_numpy()[0])**2 + \
                         (semipurple[2]-s_clr[s_clr.index==index][2].to_numpy()[0])**2) <\
                        ((semipurple[0]-s_clr[s_clr.index==s_ind[(itind+1)%6]][0].to_numpy()[0])**2 + \
                         (semipurple[1]-s_clr[s_clr.index==s_ind[(itind+1)%6]][1].to_numpy()[0])**2 + \
                         (semipurple[2]-s_clr[s_clr.index==s_ind[(itind+1)%6]][2].to_numpy()[0])**2)]
            nonPurple = semipurple[(semipurple.isin(purples))[0]==False] 

            resultCount[index] += (purple.shape[0] + purples.shape[0])
            resultCount[s_ind[(itind+1)%6]] += nonPurple.shape[0]
            purple.loc[:,[0, 1, 2]] = s_clr[s_clr.index==index][[0, 1, 2]].to_numpy()[0]
            purples.loc[:,[0, 1, 2]] = s_clr[s_clr.index==index][[0, 1, 2]].to_numpy()[0]
            nonPurple.loc[:,[0, 1, 2]] = s_clr[s_clr.index==s_ind[(itind+1)%6]][[0, 1, 2]].to_numpy()[0]
            #df = df[(df.isin(purple))[0]==False]

        #-------------------------------------------------------
        elif index == 6:
            if(row['st']>row['f']):
                brown = pd.concat([df[(df[0]>=row['st'])], df[(df[0]<row['f'])]], axis=0)
            else:
                brown = df[(df[0]>=row['st']) & (df[0]<row['f'])]

            if(fin[itind]>st[(itind+1)%6]):
                semibrown = pd.concat([df[(df[0]>=fin[itind])], df[(df[0]<st[(itind+1)%6])]], axis=0)
            else:
                semibrown = df[(df[0]>=fin[itind]) & (df[0]<st[(itind+1)%6])]

            browns = semibrown[((semibrown[0]-s_clr[s_clr.index==index][0].to_numpy()[0])**2 + \
                         (semibrown[1]-s_clr[s_clr.index==index][1].to_numpy()[0])**2 + \
                         (semibrown[2]-s_clr[s_clr.index==index][2].to_numpy()[0])**2) <\
                        ((semibrown[0]-s_clr[s_clr.index==s_ind[(itind+1)%6]][0].to_numpy()[0])**2 + \
                         (semibrown[1]-s_clr[s_clr.index==s_ind[(itind+1)%6]][1].to_numpy()[0])**2 + \
                         (semibrown[2]-s_clr[s_clr.index==s_ind[(itind+1)%6]][2].to_numpy()[0])**2)]
            nonBrown = semibrown[(semibrown.isin(browns))[0]==False] 

            resultCount[index] += (brown.shape[0] + browns.shape[0])
            resultCount[s_ind[(itind+1)%6]] += nonBrown.shape[0]
            brown.loc[:,[0, 1, 2]] = s_clr[s_clr.index==index][[0, 1, 2]].to_numpy()[0]
            browns.loc[:,[0, 1, 2]] = s_clr[s_clr.index==index][[0, 1, 2]].to_numpy()[0]
            nonBrown.loc[:,[0, 1, 2]] = s_clr[s_clr.index==s_ind[(itind+1)%6]][[0, 1, 2]].to_numpy()[0]
            #df = df[(df.isin(brown))[0]==False] 

        #-------------------------------------------------------

    totalDF = pd.concat([grey,\
                         blue, blues, nonBlue,\
                         green, greens, nonGreen,
                         red, reds, nonRed,\
                         yellow, yellows, nonYellow,\
                         purple, purples, nonPurple,\
                         brown, browns, nonBrown,\
                         black], axis=0)
    result = {}
    result['arrays'] = []
    # result['arrays'].append(list(reversed(np.sort(resultCount))))
    summ = np.sum(resultCount)
    result['arrays'].append(list(reversed(np.round(np.sort(resultCount)/summ*100, 2))))
    result['arrays'].append(list(reversed(np.argsort(resultCount))))

    # result['result'] = int(np.sum((np.round(np.sort(resultCount)/summ*100, 2))))

    totalDF.sort_index(inplace=True)
    resultIMG = totalDF.to_numpy()
    resultIMG = np.reshape(resultIMG, size)
    resultIMG=resultIMG.astype(np.uint8)
    image = cv2.cvtColor(resultIMG, cv2.COLOR_HSV2BGR) 

    outImgDir =  os.path.dirname(file_path)
    outImgName =  os.path.basename(file_path).rsplit('.', 1)[0]
    result['out'] = outImgName

    cv2.imwrite(os.path.join(outImgDir, outImgName + "_out.jpg"), image)
    return result
