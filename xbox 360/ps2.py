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

def getAmpAndPse(img_freq):
    r,i=cv2.split(img_freq)
    return cv2.magnitude(r,i),cv2.phase(r,i)

def buildImgFromAmpAndPse(amp,pse):
    mixed=np.zeros((amp.shape[0],amp.shape[1],2),dtype=np.float64)
    mixed[:mixed.shape[0],:mixed.shape[1],0]=cv2.polarToCart(amp,pse)[0]
    mixed[:mixed.shape[0],:mixed.shape[1],1]=cv2.polarToCart(amp,pse)[1]
    return cv2.idft(mixed,flags=cv2.DFT_REAL_OUTPUT|cv2.DFT_SCALE)

def buildImgFromAmpMeanAndPse(amp,pse):
    amp_mean=amp.copy().fill(amp.mean())
    mixed=np.zeros((amp_mean.shape[0],amp_mean.shape[1],2),dtype=np.float64)
    mixed[:mixed.shape[0],:mixed.shape[1],0]=cv2.polarToCart(amp_mean,pse)[0]
    mixed[:mixed.shape[0],:mixed.shape[1],1]=cv2.polarToCart(amp_mean,pse)[1]
    return cv2.idft(mixed,flags=cv2.DFT_REAL_OUTPUT|cv2.DFT_SCALE)

def getHist(img):
    # hist destination image size
    bits=8
    hist_range=(0,2**bits)
    hist_img_size={'w':512,'h':512}
    bin_width=int(round(hist_img_size['w']/hist_range[1]))
    accumulate=False
    # calculates the histogram
    hist=cv2.calcHist(img, [0], None, [hist_range[1]], hist_range, accumulate=accumulate)
    # creates image with zeros (unsigned int 256 bits)
    hist_image=np.zeros((hist_img_size['h'], hist_img_size['w'], 1), dtype=np.uint8)
    # normalize the histogram between 0 and the histogram image size
    cv2.normalize(hist, hist, alpha=0, beta=hist_img_size['h'], norm_type=cv2.NORM_MINMAX)
    thickness=2
    for i in range(1, hist_range[1]): # for each bin draws a line between the histogram points
        cv2.line(hist_image,(bin_width*(i-1),hist_img_size['h']-int(round(hist[i-1][0]))),
                (bin_width*(i),hist_img_size['h']-int(round(hist[i][0]))),
                (180,180,180),thickness=thickness)
    return hist,hist_image

def cart2pol(x, y):
    rho=np.sqrt(x**2+y**2)
    phi=np.arctan2(y,x)
    if phi<0:
        phi=2*np.pi+phi
    return rho,phi

def pol2cart(rho, phi):
    x=rho*np.cos(phi)
    y=rho*np.sin(phi)
    return x,y

def Q(hist,g_max,r):
	q=0
	for w in range(g_max):
		q+=hist[w]**r
	return q

def gEqual(hist,u,g_max,r,q):
	g_eq=0
	for w in range(u):
		g_eq+=hist[w]**r
	return g_max/q*g_eq

def fftGetMagAndPse(img_grey):
    img_freq=cv2.dft(np.float64(img_grey),flags=cv2.DFT_COMPLEX_OUTPUT)
    r,i=cv2.split(img_freq)
    return cv2.magnitude(r,i),cv2.phase(r,i)

def buildImgFromMagAndPse(mag,pse):
    mixed=np.zeros((mag.shape[0],mag.shape[1],2),dtype=np.float64)
    mixed[:mixed.shape[0],:mixed.shape[1],0]=cv2.polarToCart(mag,pse)[0]
    mixed[:mixed.shape[0],:mixed.shape[1],1]=cv2.polarToCart(mag,pse)[1]
    return cv2.idft(mixed,flags=cv2.DFT_REAL_OUTPUT|cv2.DFT_SCALE)

def noisy(noise_typ,image):
    img_shape=image.shape
    if len(img_shape)==3:
        row=img_shape[0]
        col=img_shape[1]
        ch=img_shape[2]
    else:
        row=img_shape[0]
        col=img_shape[1]
        ch=1
    if noise_typ=='gauss':
        mean=0
        var=0.1
        sigma=var**0.5
        if ch==1:
            gauss=np.random.normal(mean,sigma,(row,col))
            gauss=gauss.reshape(row,col)
        else:
            gauss=np.random.normal(mean,sigma,(row,col,ch))
            gauss=gauss.reshape(row,col,ch)
        noisy=image + gauss
        return noisy
    elif noise_typ=='s&p':
        s_vs_p=0.5
        amount=0.004
        out=np.copy(image)
        # Salt mode
        num_salt=np.ceil(amount * image.size * s_vs_p)
        coords=[np.random.randint(0, i - 1, int(num_salt))
                for i in image.shape]
        out[tuple(coords)]=1
        # Pepper mode
        num_pepper=np.ceil(amount* image.size * (1. - s_vs_p))
        coords=[np.random.randint(0, i - 1, int(num_pepper))
                for i in image.shape]
        out[tuple(coords)]=0
        return out
    elif noise_typ=='poisson':
        vals=len(np.unique(image))
        vals=2 ** np.ceil(np.log2(vals))
        noisy=np.random.poisson(image * vals) / float(vals)
        return noisy
    elif noise_typ =='speckle':
        if ch==1:
            gauss=np.random.randn(row,col)
            gauss=gauss.reshape(row,col)
        else:
            gauss=np.random.randn(row,col,ch)
            gauss=gauss.reshape(row,col,ch)
        noisy=image + image * gauss
        return noisy

def sobel(img,kernel,order=1):
    sobel_x=np.absolute(cv2.Sobel(img,cv2.CV_64F,order,0,ksize=kernel)) # sobel x operator
    sobel_y=np.absolute(cv2.Sobel(img,cv2.CV_64F,0,order,ksize=kernel)) # sobel y operator
    sobel_xy=cv2.addWeighted(sobel_x,0.5,sobel_y,0.5,0) # mean output = sobelx/2+sobely/2
    return np.uint8(sobel_xy)

def scharr(img):
    scharr_x=np.absolute(cv2.Scharr(img,cv2.CV_64F,1,0)) # scharr x operator
    scharr_y=np.absolute(cv2.Scharr(img,cv2.CV_64F,0,1)) # scharr y operator
    scharr_xy=cv2.addWeighted(scharr_x,0.5,scharr_y,0.5,0) # mean output = scharrx/2+scharry/2
    return np.uint8(scharr_xy)

def thithiEdgeDetector(img,kernel,first_order_presence=.3,sobel_presence=.5,apply_nested_laplacian=True):
    sobel_x1=np.absolute(cv2.Sobel(img,cv2.CV_64F,1,0,ksize=kernel)) # sobel x operator
    sobel_y1=np.absolute(cv2.Sobel(img,cv2.CV_64F,0,1,ksize=kernel)) # sobel y operator
    scharr_x1=np.absolute(cv2.Scharr(img,cv2.CV_64F,1,0)) # scharr x operator
    scharr_y1=np.absolute(cv2.Scharr(img,cv2.CV_64F,0,1)) # scharr y operator
    sobel_x2=np.absolute(cv2.Sobel(img,cv2.CV_64F,2,0,ksize=kernel)) # sobel x operator 2nd derivative
    sobel_y2=np.absolute(cv2.Sobel(img,cv2.CV_64F,0,2,ksize=kernel)) # sobel y operator 2nd derivative
    sobel_x=cv2.addWeighted(sobel_x1,first_order_presence,sobel_x2,(1-first_order_presence),0)
    sobel_y=cv2.addWeighted(sobel_y1,first_order_presence,sobel_y2,(1-first_order_presence),0)
    scharr_x=scharr_x1
    scharr_y=scharr_y1
    th_ed_x=cv2.addWeighted(sobel_x,sobel_presence,scharr_x,(1-sobel_presence),0)
    th_ed_y=cv2.addWeighted(sobel_y,sobel_presence,scharr_y,(1-sobel_presence),0)
    if apply_nested_laplacian:
        th_ed_x=255-np.absolute(cv2.Laplacian(th_ed_x,cv2.CV_64F,kernel)) # apply and invert laplace
        th_ed_y=255-np.absolute(cv2.Laplacian(th_ed_y,cv2.CV_64F,kernel)) # apply and invert laplace
    th_xy=cv2.addWeighted(th_ed_x,0.5,th_ed_y,0.5,0)    
    return np.uint8(th_xy)

def getWindow(img,x,y,size):
    if size<=0:
        return np.empty([0,0])
    size_per_2=int(size/2)
    if x>size_per_2 and y>size_per_2 and x<img.shape[1]-size_per_2 and y<img.shape[0]-size_per_2:
        return img[y-size_per_2:y+size_per_2+1, x-size_per_2:x+size_per_2+1]
    else:
        return getWindow(img,x,y,size-1)

def ex1(path='tulips.png'):
    img=cv2.imread(path) # open image
    img_grey=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY) # convert to grey
    img_noise=noisy('gauss',img_grey) # noise image
    img_noise=img_noise.astype(np.uint8)
    img_noise_filtered=cv2.bilateralFilter(img_noise,1,100,100) # sigma filter
    hist,img_hist_orig=getHist(img_noise_filtered)
    cv2.imshow('Filtered noisy image', img_noise_filtered)
    cv2.imshow('Filtered noisy image histogram', img_hist_orig)
    g_max=255
    img_noise_filtered_shape=img_noise_filtered.shape
    for r in (0,0.5,1,2,3):
        equalized_img=np.zeros((img_noise_filtered_shape[0],img_noise_filtered_shape[1],1),dtype=np.uint8)
        q=Q(hist.copy(),g_max,r)
        for i in range(img_noise_filtered_shape[1]):
            for j in range(img_noise_filtered_shape[0]):
                equalized_img[j][i]=gEqual(hist,img_grey[j][i],g_max,r,q)
        _,eq_img_hist_orig=getHist(equalized_img)
        cv2.imshow('Equalized ({}) image'.format(r), equalized_img)
        cv2.imshow('Equalized ({}) image histogram'.format(r), eq_img_hist_orig)
    cv2.waitKey()

def ex2(path='pulp.png'):
    kernel=5
    img=cv2.imread(path) # open image
    img_grey=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY) # convert to grey
    img_grey=cv2.GaussianBlur(img_grey,(kernel,kernel),2) # reduce noise
    img_sobel=sobel(img_grey,kernel)
    img_sobel2=sobel(img_grey,kernel,2)
    img_scharr=scharr(img_grey)
    img_laplacian=np.uint8(np.absolute(cv2.Laplacian(img_grey,cv2.CV_64F,kernel)))
    img_th=thithiEdgeDetector(img_grey,kernel)
    cv2.imshow('Filtered Image', img_grey)
    cv2.imshow('Sobel Edge Detector', img_sobel)
    cv2.imshow('Sobel 2 Edge Detector', img_sobel2)
    cv2.imshow('Scharr Edge Detector', img_scharr)
    cv2.imshow('Laplacian Edge Detector', img_laplacian)
    cv2.imshow('Inverted Laplacian Edge Detector', (255-img_laplacian))
    cv2.imshow('Custom Edge Detector', img_th)
    cv2.waitKey()

def ex3(path='beatrix.png'):
    kernel=3
    img=cv2.imread(path) # open image
    img_window=getWindow(img,200,200,250)
    img_window_grey=cv2.cvtColor(img_window,cv2.COLOR_BGR2GRAY) # convert to grey
    mag,pse=fftGetMagAndPse(img_window_grey)
    mag_avg=mag.copy()
    mag_avg.fill(mag.mean())
    pse_avg=pse.copy()
    pse_avg.fill(pse.mean())
    img_mag=buildImgFromMagAndPse(mag,pse_avg)
    img_pse=buildImgFromMagAndPse(mag_avg,pse)
    img_window_grey=cv2.GaussianBlur(img_window_grey,(kernel,kernel),2) # reduce noise
    img_mag=cv2.GaussianBlur(img_mag,(kernel,kernel),2) # reduce noise
    img_pse=cv2.GaussianBlur(img_pse,(kernel,kernel),2) # reduce noise
    img_window_grey_ed=thithiEdgeDetector(img_window_grey,kernel)
    img_mag_ed=thithiEdgeDetector(img_mag,kernel)
    img_pse_ed=thithiEdgeDetector(img_pse,kernel)
    cv2.imshow('Img', img_window_grey)
    cv2.imshow('Ed Img', img_window_grey_ed)
    cv2.imshow('Img Mag', img_mag)
    cv2.imshow('Ed Mag', img_mag_ed)
    cv2.imshow('Img Pse', img_pse)
    cv2.imshow('Ed Pse', img_pse_ed)
    cv2.waitKey()

def main(argv):
    print('Select an exercise to run [1-3]:')
    ex=inputNumber(greater_or_eq=1,lower_or_eq=3)
    if ex==1:
        ex1()
    elif ex==2:
        ex2()
    elif ex==3:
        ex3()
if __name__=='__main__':
    main(sys.argv[1:])