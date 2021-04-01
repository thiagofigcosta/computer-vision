#!/bin/python

import cv2,sys,time,math
import random as rd
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


def derivativeConvolution(img_1,img_2,kx,ky,kz):
    ddepth=-1
    fx=cv2.filter2D(img_1,ddepth,kx)+cv2.filter2D(img_2,ddepth,kx)
    fy=cv2.filter2D(img_1,ddepth,ky)+cv2.filter2D(img_2,ddepth,ky)
    fz=cv2.filter2D(img_1,ddepth,kz)+cv2.filter2D(img_2,ddepth,kz)
    return fx,fy,fz

def derivativeSobel(img_1,img_2,order=1):
    ddepth=-1
    kernel=5
    fx=np.absolute(cv2.Sobel(img_1,ddepth,order,0,ksize=kernel))+np.absolute(cv2.Sobel(img_2,ddepth,order,0,ksize=kernel))
    fy=np.absolute(cv2.Sobel(img_1,ddepth,0,order,ksize=kernel))+np.absolute(cv2.Sobel(img_2,ddepth,0,order,ksize=kernel))
    fz=img_2-img_1
    return fx,fy,fz

def derivativeScharr(img_1,img_2):
    ddepth=-1
    fx=np.absolute(cv2.Scharr(img_1,ddepth,1,0))+np.absolute(cv2.Scharr(img_2,ddepth,1,0))
    fy=np.absolute(cv2.Scharr(img_1,ddepth,0,1))+np.absolute(cv2.Scharr(img_2,ddepth,0,1))
    fz=img_2-img_1
    return fx,fy,fz

def hornSchunckAlgorithm(img_1,img_2,alpha,max_iters,method=1):
    ddepth=-1
    corners=1/12.0
    adjacent=1/6.0
    kernel_x=np.matrix([[-1,1],[-1,1]])*.25
    kernel_y=np.matrix([[-1,-1],[1,1]])*.25
    kernel_z=np.matrix([[1,1],[1,1]])*.25
    kernel_mean=np.matrix([[corners,adjacent,corners],[adjacent,0,adjacent],[corners,adjacent,corners]])
    img_1=np.float64(img_1) # increase precision
    img_2=np.float64(img_2) # increase precision
    u=np.zeros((img_1.shape[:2]))
    v=np.zeros((img_1.shape[:2]))
    if method==1:
        fx,fy,fz=derivativeConvolution(img_1,img_2,kernel_x,kernel_y,kernel_z)
    elif method==2:
        fx,fy,fz=derivativeSobel(img_1,img_2)
    elif method==3:
        fx,fy,fz=derivativeScharr(img_1,img_2)
    else:
        raise Exception('Not implemented yet')
    normalization=alpha**2+fx**2+fy**2 # calculate step normalization
    for i in range(max_iters):
        u_mean=cv2.filter2D(u,ddepth,kernel_mean) # computate surround mean
        v_mean=cv2.filter2D(v,ddepth,kernel_mean) # computate surround mean
        step=(fx*u_mean+fy*v_mean+fz)/normalization # error reduction module
        grad_x=fx*step # gradient
        grad_y=fy*step # gradient
        print('Iter: {} - Absolute gradient x: {:.2f} - Absolute gradient y: {:.2f}'.format(i+1,np.sum(np.absolute(grad_x)),np.sum(np.absolute(grad_y))))
        u=u_mean-grad_x
        v=v_mean-grad_y
    if method==1:
        u=np.uint8(u)
        v=np.uint8(v)
    return u,v 

def findCorners(img_grey,img_out=None,max_corners=1000,base_name='Img'):
    point_size=5
    quality_level=0.01
    min_distance=10
    block_size=3
    gradient_size=3
    harris_detector=False
    k=0.04
    win_size=(5,5)
    zero_zone=(-1,-1)
    criteria=(cv2.TERM_CRITERIA_EPS + cv2.TermCriteria_COUNT,500,0.0001)
    corners=cv2.goodFeaturesToTrack(img_grey,max_corners,quality_level,min_distance,None,blockSize=block_size, gradientSize=gradient_size, useHarrisDetector=harris_detector,k=k) # corner detection
    found_corners=corners.shape[0]
    print('Corners: {}'.format(found_corners))
    color=(0,0,255)
    if img_out is not None:
        for i in range(corners.shape[0]):
            cv2.circle(img_out,(int(corners[i,0,0]),int(corners[i,0,1])),point_size,color,cv2.FILLED)
    cv2.imshow('{} - Corners'.format(base_name), img_out)
    # sub pixel precision
    corners=cv2.cornerSubPix(img_grey,corners,win_size,zero_zone,criteria) # subpixel corner detection
    color=(255,0,0)
    if img_out is not None:
        for i in range(corners.shape[0]):
            cv2.circle(img_out,(int(corners[i,0,0]),int(corners[i,0,1])),point_size,color,cv2.FILLED)
    cv2.imshow('{} - SubCorners'.format(base_name), img_out)
    return corners

def ex1(paths=[('inglorious_1.png','inglorious_2.png'),('hateful_1.png','hateful_2.png')]):
    alpha=0.001
    max_iters=20
    for path_1,path_2 in paths:
        img_1=cv2.imread(path_1) # open image
        img_2=cv2.imread(path_2) # open image
        img_1=cv2.cvtColor(img_1,cv2.COLOR_BGR2GRAY) # convert to grey
        img_2=cv2.cvtColor(img_2,cv2.COLOR_BGR2GRAY) # convert to grey
        cv2.imshow('{} - Raw'.format(path_1), img_1)
        cv2.imshow('{} - Raw'.format(path_2), img_2)
        u_1,v_1=hornSchunckAlgorithm(img_1,img_2,alpha,max_iters) # calc horn
        u_2,v_2=hornSchunckAlgorithm(img_1,img_2,alpha,max_iters,2) # calc horn
        u_3,v_3=hornSchunckAlgorithm(img_1,img_2,alpha,max_iters,3) # calc horn
        cv2.imshow('{} - M1 - U'.format(path_1), u_1)
        cv2.imshow('{} - M1 - V'.format(path_1), v_1)
        cv2.imshow('{} - M2 - U'.format(path_1), u_2)
        cv2.imshow('{} - M2 - V'.format(path_1), v_2)
        cv2.imshow('{} - M3 - U'.format(path_1), u_3)
        cv2.imshow('{} - M3 - V'.format(path_1), v_3)
        cv2.waitKey()

def ex2(paths=['spring.png','django.png','death.png']):
    sps=[1,5,12,25]
    crs=[1,19,24,25]
    ls=[1,3,5]
    for path in paths:
        img=cv2.imread(path)
        img=resizeImg(img,512)
        cv2.imshow(path,img)
        for sp in sps:
            for cr in crs:
                for l in ls:
                    img_cur=img.copy()
                    img_cur=cv2.pyrMeanShiftFiltering(img_cur,sp=sp,sr=cr,maxLevel=l)
                    cv2.imshow('{} - sp:{} cr:{} l:{} Filtered'.format(path,sp,cr,l), img_cur)
        print('Finished filtering for {}'.format(path))
        cv2.waitKey()

def ex3(paths=['chessboard_0.png','chessboard_0_barrel.png','chessboard_1.png','chessboard_2.png','chessboard_3.png']):
    thickness=1
    color=(0,255,0)
    for path in paths:
        img=cv2.imread(path) # open image
        img_grey=cv2.cvtColor(img.copy(),cv2.COLOR_BGR2GRAY) # convert to grey
        cv2.imshow('{}'.format(path), img_grey)
        img_hough_out=cv2.cvtColor(img_grey.copy(), cv2.COLOR_GRAY2BGR)
        img_hough_in=cv2.Canny(img_grey,50,150) # get edges
        cv2.imshow('{} - Edges'.format(path), img_hough_in)
        lines=cv2.HoughLinesP(img_hough_in,1,np.pi/180,80,minLineLength=80,maxLineGap=10) # find lines
        found_lines=len(lines)
        print('Lines: {}'.format(found_lines))
        for line in lines:
            for x1,y1,x2,y2 in line:
                cv2.line(img_hough_out,(x1, y1),(x2, y2),color,thickness,cv2.LINE_AA)
        cv2.imshow('{} - Lines'.format(path), img_hough_out)
        corners=findCorners(img_grey,img_hough_out,max_corners=100,base_name=path)
        # calibrating
        try:
            board_size=math.sqrt(len(corners))
            if board_size.is_integer():
                board_rows=int(board_size)
                board_cols=int(board_size)
            else:
                board_rows=int(board_size)+1
                board_cols=int(board_size)+1
            object_points=np.zeros((board_cols*board_rows,3),np.float32)
            object_points[:,:2]=np.mgrid[0:board_cols,0:board_rows].T.reshape(-1,2) # object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
            if not board_size.is_integer():
                object_points=object_points[:len(corners)]
            ret,mtx,dist,rvecs,tvecs=cv2.calibrateCamera([object_points],[corners],img.shape[:2],None,None) # calibration parameters
            fixed_camera_matrix,roi=cv2.getOptimalNewCameraMatrix(mtx,dist,(img.shape[1],img.shape[0]),1,(img.shape[1],img.shape[0]))
            img_calibrated=cv2.undistort(img,mtx,dist,None,fixed_camera_matrix)
            cv2.imshow('{} - Calibrated'.format(path), img_calibrated)
        except:
            print('Could not calibrate image.\n\tBoard size: {}x{}\n\tObject points: {} - Image Points: {}'.format(board_rows,board_cols,len(object_points),len(corners)))
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