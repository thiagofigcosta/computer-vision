#!/bin/python

import cv2,sys,time
import numpy as np

def inputNumber(is_float=False,greater_or_eq=0,lower_or_eq=None):
    out=0
    converted=False
    while not converted:
        try:
            if is_float:
                out=float(input())
            else:
                out=int(input())
            if (lower_or_eq==None or out <= lower_or_eq) and (greater_or_eq==None or out >= greater_or_eq):
                converted=True
            else:
                print('ERROR. Out of boundaries [{},{}], type again: '.format(greater_or_eq,lower_or_eq))
        except ValueError:
            if not is_float:
                print('ERROR. Not an integer, type again: ')
            else:
                print('ERROR. Not a float number, type again: ')
    return out

def resizeImg(img,width):
    return cv2.resize(img,(width,int(width*img.shape[0]/img.shape[1])))

def fromHSItoBGR(hsi,rgb=False):
    if hsi[1]==0:
        return[0]*3
    else:
        rgb_color=[0]*3
        p=255*hsi[2]
        pi23=2*np.pi/3
        pi3=np.pi/3
        i=0
        while hsi[0]>=pi23:
            i+=1
            hsi[0]-=pi23
        rgb_color[i%3]=p*(1-hsi[1])
        rgb_color[(1+i)%3]=p*(1+hsi[1]*np.cos(hsi[0])/np.cos(pi3-hsi[0]))
        rgb_color[(2+i)%3]=p*(1+hsi[1]*np.cos(pi23-hsi[0])/np.cos(pi3-hsi[0]))
        if not rgb:
            rgb_color.reverse()
        return rgb_color

def getAmpAndPse(img_freq):
    r,i=cv2.split(img_freq)
    return cv2.magnitude(r,i),cv2.phase(r,i)

def cart2pol(x, y):
    rho=np.sqrt(x**2 + y**2)
    phi=np.arctan2(y, x)
    if phi<0:
        phi=2*np.pi+phi
    return rho,phi

def pol2cart(rho, phi):
    x=rho*np.cos(phi)
    y=rho*np.sin(phi)
    return x,y

def buildImgFromAmpAndPse(amp,pse):
    mixed=np.zeros((amp.shape[0],amp.shape[1],2),dtype=np.float64)
    mixed[:mixed.shape[0],:mixed.shape[1],0]=cv2.polarToCart(amp,pse)[0]
    mixed[:mixed.shape[0],:mixed.shape[1],1]=cv2.polarToCart(amp,pse)[1]
    return cv2.idft(mixed,flags=cv2.DFT_REAL_OUTPUT|cv2.DFT_SCALE)

def ex1(path='djonga.png'):
    img=cv2.imread(path) # open image
    bgr_channels=cv2.split(img) # split channels
    # hist destination image size
    bits=8
    hist_range=(0,2**bits)
    hist_img_size={'w':512,'h':512}
    bin_width=int(round( hist_img_size['w']/hist_range[1] ))
    accumulate=False
    # calculates the histogram
    r_hist=cv2.calcHist(bgr_channels, [2], None, [hist_range[1]], hist_range, accumulate=accumulate)
    g_hist=cv2.calcHist(bgr_channels, [1], None, [hist_range[1]], hist_range, accumulate=accumulate)
    b_hist=cv2.calcHist(bgr_channels, [0], None, [hist_range[1]], hist_range, accumulate=accumulate)
    # creates RGB image with zeros (unsigned int 256 bits)
    hist_image=np.zeros((hist_img_size['h'], hist_img_size['w'], 3), dtype=np.uint8)
    # normalize the histogram between 0 and the histogram image size
    cv2.normalize(r_hist, r_hist, alpha=0, beta=hist_img_size['h'], norm_type=cv2.NORM_MINMAX)
    cv2.normalize(g_hist, g_hist, alpha=0, beta=hist_img_size['h'], norm_type=cv2.NORM_MINMAX)
    cv2.normalize(b_hist, b_hist, alpha=0, beta=hist_img_size['h'], norm_type=cv2.NORM_MINMAX)
    thickness=2
    for i in range(1, hist_range[1]): # for each bin draws a line between the histogram points
        cv2.line(hist_image, ( bin_width*(i-1), hist_img_size['h'] - int(round(r_hist[i-1][0])) ),
                ( bin_width*(i), hist_img_size['h'] - int(round(r_hist[i][0])) ),
                ( 0, 0, 255), thickness=thickness) # red
        cv2.line(hist_image, ( bin_width*(i-1), hist_img_size['h'] - int(round(g_hist[i-1][0])) ),
                ( bin_width*(i), hist_img_size['h'] - int(round(g_hist[i][0])) ),
                ( 0, 255, 0), thickness=thickness) # green
        cv2.line(hist_image, ( bin_width*(i-1), hist_img_size['h'] - int(round(b_hist[i-1][0])) ),
                ( bin_width*(i), hist_img_size['h'] - int(round(b_hist[i][0])) ),
                ( 255, 0, 0), thickness=thickness) # blue
    img_name='Djonga'
    cv2.imshow(img_name, img.copy())
    cv2.imshow('Histogram', hist_image)
    def mouseHoverCB(event,x,y,flag,param):
        size=11
        size_per_2=int(size/2)
        if event==cv2.EVENT_MOUSEMOVE and x>size_per_2 and y>size_per_2 and x<img.shape[1]-size_per_2 and y<img.shape[0]-size_per_2:
            grey_intensity=180
            intensity=np.average(img[y,x])
            mean=np.average(img[y-size_per_2:y+size_per_2+1, x-size_per_2:x+size_per_2+1])
            std_dev=np.std(img[y-size_per_2:y+size_per_2+1, x-size_per_2:x+size_per_2+1])
            tmp_img=img.copy()
            cv2.rectangle(tmp_img, (x-size_per_2, y-size_per_2), (x+size_per_2,y+size_per_2), (grey_intensity, grey_intensity, grey_intensity), 1) # draw rectangle
            print('x: {} y:{} intensity: {} mean: {} std_dev: {}'.format(x,y,intensity,mean,std_dev))
            cv2.imshow(img_name, tmp_img)
    cv2.setMouseCallback(img_name,mouseHoverCB)
    cv2.waitKey()

def ex2(path='rocky.mp4'):
    vid=cv2.VideoCapture(path) # 0 to use webcam
    if not vid.isOpened():
        print('Error, could not open video')
        exit(1)
    # gets video FpS
    (major_ver, minor_ver, subminor_ver)=(cv2.__version__).split('.')
    if int(major_ver)<3:
        fps=vid.get(cv2.cv.CV_CAP_PROP_FPS)
    else:
        fps=vid.get(cv2.CAP_PROP_FPS)
    ms_p_f=1000/fps
    means=[]
    std_devs=[]
    contrasts=[]
    means_norm=[]
    std_devs_norm=[]
    # iterate video
    while(vid.isOpened()):
        ret, frame=vid.read() # gets fram
        if ret:
            frame=resizeImg(frame,500) # resize frame
            start=time.time()
            frame_grey=cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            contrast=frame_grey.std()
            mean=np.average(frame)
            std_dev=np.std(frame)
            print('contrast {}'.format(contrast))
            print('mean {}'.format(mean))
            print('std_dev {}'.format(std_dev))
            means.append(mean)
            std_devs.append(std_dev)
            contrasts.append(contrast)
            # normalize
            alfa=(std_dev/std_devs[0])*means[0]-mean
            beta=std_devs[0]/std_dev
            normalized=cv2.normalize(src=frame.copy(),dst=None,alpha=alfa,beta=beta,dtype=cv2.CV_32F,norm_type=cv2.NORM_MINMAX)
            means_norm.append(np.average(normalized))
            std_devs_norm.append(np.std(normalized))
            cv2.imshow(path,frame)
            cv2.imshow('normalized_'+path,normalized)
            # respect fps
            delta=int(ms_p_f-(time.time()-start))
            if delta<=0:
                delta=1
            if cv2.waitKey(delta)==27: #escape key
                break
        else: 
            break
    print('normalized mean std {}'.format(np.std(np.array(means_norm))))
    print('normalized std std {}'.format(np.std(np.array(std_devs_norm))))
    print('regular mean std {}'.format(np.std(np.array(means))))
    print('regular std std {}'.format(np.std(np.array(std_devs))))
    vid.release()
    cv2.waitKey()

def ex3(path_mix1='luffy_1.png',path_mix2='luffy_2.png',path_transform1='tex_1.png',path_transform2='tex_2.png',wait=True):
    size=300
    # load with same size
    img_amp=resizeImg(cv2.imread(path_mix1),size) 
    img_pse=resizeImg(cv2.imread(path_mix2),size)
    # convert to grey
    img_amp_grey=cv2.cvtColor(img_amp,cv2.COLOR_BGR2GRAY)
    img_pse_grey=cv2.cvtColor(img_pse,cv2.COLOR_BGR2GRAY)
    # convert to frequency domain
    img_amp_freq=cv2.dft(np.float64(img_amp_grey),flags=cv2.DFT_COMPLEX_OUTPUT)
    img_pse_freq=cv2.dft(np.float64(img_pse_grey),flags=cv2.DFT_COMPLEX_OUTPUT)
    # split into magnitude/amplitude and phase
    amp1,pse1=getAmpAndPse(img_amp_freq)
    amp2,pse2=getAmpAndPse(img_pse_freq)
    #mix images
    mixed1=buildImgFromAmpAndPse(amp1,pse2)
    cv2.imshow('mixed_'+path_mix1+'(amp)_'+path_mix2+'(pse)',np.array(mixed1,dtype=np.uint8))
    mixed2=buildImgFromAmpAndPse(amp2,pse1)
    cv2.imshow('mixed_'+path_mix2+'(amp)_'+path_mix1+'(pse)',np.array(mixed2,dtype=np.uint8))
    img_tex1=resizeImg(cv2.imread(path_transform1),size)
    img_tex2=resizeImg(cv2.imread(path_transform2),size)
    img_tex1_grey=cv2.cvtColor(img_tex1,cv2.COLOR_BGR2GRAY)
    img_tex2_grey=cv2.cvtColor(img_tex2,cv2.COLOR_BGR2GRAY)
    img_tex1_freq=cv2.dft(np.float64(img_tex1_grey),flags=cv2.DFT_COMPLEX_OUTPUT)
    img_tex2_freq=cv2.dft(np.float64(img_tex2_grey),flags=cv2.DFT_COMPLEX_OUTPUT)
    amp3,pse3=getAmpAndPse(img_tex1_freq)
    amp4,pse4=getAmpAndPse(img_tex2_freq)
    # multiply uniformly
    for i in range(len(amp3)):
        for j in range(len(amp3[i])):
            amp3[i][j]*=3
    for i in range(len(pse4)):
        for j in range(len(pse4[i])):
            pse4[i][j]*=3
    trans1=buildImgFromAmpAndPse(amp3,pse3)
    trans2=buildImgFromAmpAndPse(amp4,pse4)
    cv2.imshow('orig1'+path_transform1,img_tex1_grey)
    cv2.imshow('orig2'+path_transform2,img_tex2_grey)
    cv2.imshow('trans1'+path_transform1,np.array(trans1,dtype=np.uint8))
    cv2.imshow('trans2'+path_transform2,np.array(trans2,dtype=np.uint8))
    if wait:
        cv2.waitKey()

def ex4(animation=True,square=True):
    intensity=0
    if not animation:
        print('Enter a intensity value ranging from 0 to 255: ',end='')
        intensity=inputNumber(greater_or_eq=0,lower_or_eq=255)
    if not animation:
        print('Enter a image size >50: ',end='')
    else:
        print('Enter a image size >50 (if your computer is not powerful choose a small value): ',end='')
    size=inputNumber(greater_or_eq=51)
    half_size=size/2.0
    ms_p_f=10
    signal=1
    max_val=255.0
    animating=True
    while animating:
        animating=animation
        start=time.time()
        hsi_space_img=np.zeros((size,size,3),np.uint8) # create 3 channel image
        hsi_saturation_img=np.zeros((size,size,1),np.uint8) # create grey scale iamge
        hsi_color=[0,0,intensity/max_val]
        for i in range(size):
            for j in range(size):
                hsi_color[:2]=[0,0]
                centralized_x=(i-half_size)/half_size
                centralized_y=(half_size-j)/half_size
                saturation,hue=cart2pol(centralized_x,centralized_y)
                sqrt2=2**.5
                if saturation*sqrt2<=1/np.cos((2/6)*np.arcsin(np.sin((6/2)*hue-np.pi/2))) and square: # produces square cut
                    hsi_color[:2]=[hue,saturation/sqrt2]
                    hsi_space_img[j,i]=fromHSItoBGR(hsi_color)
                elif saturation<=1 and not square: # produces circle
                    hsi_color[:2]=[hue,saturation]
                    hsi_space_img[j,i]=fromHSItoBGR(hsi_color)
                else:
                    hsi_space_img[j,i]=[0,0,0]
                hsi_saturation_img[j,i]=hsi_color[1]*max_val
        hsi_space_img[int(half_size),int(half_size)]=[intensity]*3
        cv2.imshow("HSI Space", hsi_space_img)
        cv2.imshow("HSI Saturation", hsi_saturation_img)
        intensity+=signal
        if intensity>max_val or intensity<0:
            signal*=-1
            if intensity<0:
                intensity=1
            if intensity>max_val:
                intensity=max_val-1
        delta=int(ms_p_f-(time.time()-start)*1000)
        if delta<=0:
            delta=1
        if not animation:
            delta=None
        cv2.waitKey(delta)


def main(argv):
    print('Select an exercise to run [1-4]:')
    ex=inputNumber(greater_or_eq=1,lower_or_eq=4)
    if ex==1:
        ex1()
    elif ex==2:
        ex2()
    elif ex==3:
        ex3(wait=False)
        ex3('taylor_1.png','mc_rick_1.png','fred.png','renato.png')
    elif ex==4:
        print('Running fixed intensity and then continue to see animation')
        ex4(False)
        ex4()
if __name__ == "__main__":
    main(sys.argv[1:])