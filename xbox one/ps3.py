#!/bin/python

import matplotlib.pyplot as plt
import cv2,sys,time,math
import random as rd
import numpy as np

# code from skimage.feature.texture https://github.com/scikit-image/scikit-image/blob/main/skimage/feature/texture.py#L15-L155
def _glcm_loop(image,distances,angles, levels, out):
    """Perform co-occurrence matrix accumulation.
    Parameters
    ----------
    image : ndarray
        Integer typed input image. Only positive valued images are supported.
        If type is other than uint8, the argument `levels` needs to be set.
    distances : ndarray
        List of pixel pair distance offsets.
    angles : ndarray
        List of pixel pair angles in radians.
    levels : int
        The input image should contain integers in [0, `levels`-1],
        where levels indicate the number of gray-levels counted
        (typically 256 for an 8-bit image).
    out : ndarray
        On input a 4D array of zeros, and on output it contains
        the results of the GLCM computation.
    """

    
    rows = image.shape[0]
    cols = image.shape[1]

    for a_idx in range(angles.shape[0]):
        angle = angles[a_idx]
        for d_idx in range(distances.shape[0]):
            distance = distances[d_idx]
            offset_row = round(math.sin(angle) * distance)
            offset_col = round(math.cos(angle) * distance)
            start_row = max(0, -offset_row)
            end_row = min(rows, rows - offset_row)
            start_col = max(0, -offset_col)
            end_col = min(cols, cols - offset_col)
            for r in range(start_row, end_row):
                for c in range(start_col, end_col):
                    i = image[r, c]
                    # compute the location of the offset pixel
                    row = r + offset_row
                    col = c + offset_col
                    j = image[row, col]
                    if 0 <= i < levels and 0 <= j < levels:
                        out[i, j, d_idx, a_idx] += 1


def _bit_rotate_right(value, length):
    """Cyclic bit shift to the right.
    Parameters
    ----------
    value : int
        integer value to shift
    length : int
        number of bits of integer
    """
    value=int(value)
    length=int(length)
    return (value >> 1) | ((value & 1) << (length - 1))

def check_nD(array, ndim, arg_name='image'):
    """
    Verify an array meets the desired ndims and array isn't empty.
    Parameters
    ----------
    array : array-like
        Input array to be validated
    ndim : int or iterable of ints
        Allowable ndim or ndims for the array.
    arg_name : str, optional
        The name of the array in the original function.
    """
    array = np.asanyarray(array)
    msg_incorrect_dim = "The parameter `%s` must be a %s-dimensional array"
    msg_empty_array = "The parameter `%s` cannot be an empty array"
    if isinstance(ndim, int):
        ndim = [ndim]
    if array.size == 0:
        raise ValueError(msg_empty_array % (arg_name))
    if not array.ndim in ndim:
        raise ValueError(msg_incorrect_dim % (arg_name, '-or-'.join([str(n) for n in ndim])))

def greycomatrix(image, distances, angles, levels=None, symmetric=False,
                 normed=False):
    """Calculate the grey-level co-occurrence matrix.
    A grey level co-occurrence matrix is a histogram of co-occurring
    greyscale values at a given offset over an image.
    Parameters
    ----------
    image : array_like
        Integer typed input image. Only positive valued images are supported.
        If type is other than uint8, the argument `levels` needs to be set.
    distances : array_like
        List of pixel pair distance offsets.
    angles : array_like
        List of pixel pair angles in radians.
    levels : int, optional
        The input image should contain integers in [0, `levels`-1],
        where levels indicate the number of grey-levels counted
        (typically 256 for an 8-bit image). This argument is required for
        16-bit images or higher and is typically the maximum of the image.
        As the output matrix is at least `levels` x `levels`, it might
        be preferable to use binning of the input image rather than
        large values for `levels`.
    symmetric : bool, optional
        If True, the output matrix `P[:, :, d, theta]` is symmetric. This
        is accomplished by ignoring the order of value pairs, so both
        (i, j) and (j, i) are accumulated when (i, j) is encountered
        for a given offset. The default is False.
    normed : bool, optional
        If True, normalize each matrix `P[:, :, d, theta]` by dividing
        by the total number of accumulated co-occurrences for the given
        offset. The elements of the resulting matrix sum to 1. The
        default is False.
    Returns
    -------
    P : 4-D ndarray
        The grey-level co-occurrence histogram. The value
        `P[i,j,d,theta]` is the number of times that grey-level `j`
        occurs at a distance `d` and at an angle `theta` from
        grey-level `i`. If `normed` is `False`, the output is of
        type uint32, otherwise it is float64. The dimensions are:
        levels x levels x number of distances x number of angles.
    References
    ----------
    .. [1] The GLCM Tutorial Home Page,
           http://www.fp.ucalgary.ca/mhallbey/tutorial.htm
    .. [2] Haralick, RM.; Shanmugam, K.,
           "Textural features for image classification"
           IEEE Transactions on systems, man, and cybernetics 6 (1973): 610-621.
           :DOI:`10.1109/TSMC.1973.4309314`
    .. [3] Pattern Recognition Engineering, Morton Nadler & Eric P.
           Smith
    .. [4] Wikipedia, https://en.wikipedia.org/wiki/Co-occurrence_matrix
    Examples
    --------
    Compute 2 GLCMs: One for a 1-pixel offset to the right, and one
    for a 1-pixel offset upwards.
    >>> image = np.array([[0, 0, 1, 1],
    ...                   [0, 0, 1, 1],
    ...                   [0, 2, 2, 2],
    ...                   [2, 2, 3, 3]], dtype=np.uint8)
    >>> result = greycomatrix(image, [1], [0, np.pi/4, np.pi/2, 3*np.pi/4],
    ...                       levels=4)
    >>> result[:, :, 0, 0]
    array([[2, 2, 1, 0],
           [0, 2, 0, 0],
           [0, 0, 3, 1],
           [0, 0, 0, 1]], dtype=uint32)
    >>> result[:, :, 0, 1]
    array([[1, 1, 3, 0],
           [0, 1, 1, 0],
           [0, 0, 0, 2],
           [0, 0, 0, 0]], dtype=uint32)
    >>> result[:, :, 0, 2]
    array([[3, 0, 2, 0],
           [0, 2, 2, 0],
           [0, 0, 1, 2],
           [0, 0, 0, 0]], dtype=uint32)
    >>> result[:, :, 0, 3]
    array([[2, 0, 0, 0],
           [1, 1, 2, 0],
           [0, 0, 2, 1],
           [0, 0, 0, 0]], dtype=uint32)
    """
    check_nD(image, 2)
    check_nD(distances, 1, 'distances')
    check_nD(angles, 1, 'angles')

    image = np.ascontiguousarray(image)

    image_max = image.max()

    if np.issubdtype(image.dtype, np.floating):
        raise ValueError("Float images are not supported by greycomatrix. "
                         "Convert the image to an unsigned integer type.")

    # for image type > 8bit, levels must be set.
    if image.dtype not in (np.uint8, np.int8) and levels is None:
        raise ValueError("The levels argument is required for data types "
                         "other than uint8. The resulting matrix will be at "
                         "least levels ** 2 in size.")

    if np.issubdtype(image.dtype, np.signedinteger) and np.any(image < 0):
        raise ValueError("Negative-valued images are not supported.")

    if levels is None:
        levels = 256

    if image_max >= levels:
        raise ValueError("The maximum grayscale value in the image should be "
                         "smaller than the number of levels.")

    distances = np.ascontiguousarray(distances, dtype=np.float64)
    angles = np.ascontiguousarray(angles, dtype=np.float64)

    P = np.zeros((levels, levels, len(distances), len(angles)),
                 dtype=np.uint32, order='C')

    # count co-occurences
    _glcm_loop(image, distances, angles, levels, P)

    # make each GLMC symmetric
    if symmetric:
        Pt = np.transpose(P, (1, 0, 2, 3))
        P = P + Pt

    # normalize each GLCM
    if normed:
        P = P.astype(np.float64)
        glcm_sums = np.apply_over_axes(np.sum, P, axes=(0, 1))
        glcm_sums[glcm_sums == 0] = 1
        P /= glcm_sums

    return P

def greycoprops(P, prop='contrast'):
    """Calculate texture properties of a GLCM.
    Compute a feature of a grey level co-occurrence matrix to serve as
    a compact summary of the matrix. The properties are computed as
    follows:
    - 'contrast': :math:`\\sum_{i,j=0}^{levels-1} P_{i,j}(i-j)^2`
    - 'dissimilarity': :math:`\\sum_{i,j=0}^{levels-1}P_{i,j}|i-j|`
    - 'homogeneity': :math:`\\sum_{i,j=0}^{levels-1}\\frac{P_{i,j}}{1+(i-j)^2}`
    - 'ASM': :math:`\\sum_{i,j=0}^{levels-1} P_{i,j}^2`
    - 'energy': :math:`\\sqrt{ASM}`
    - 'correlation':
        .. math:: \\sum_{i,j=0}^{levels-1} P_{i,j}\\left[\\frac{(i-\\mu_i) \\
                  (j-\\mu_j)}{\\sqrt{(\\sigma_i^2)(\\sigma_j^2)}}\\right]
    Each GLCM is normalized to have a sum of 1 before the computation of texture
    properties.
    Parameters
    ----------
    P : ndarray
        Input array. `P` is the grey-level co-occurrence histogram
        for which to compute the specified property. The value
        `P[i,j,d,theta]` is the number of times that grey-level j
        occurs at a distance d and at an angle theta from
        grey-level i.
    prop : {'contrast', 'dissimilarity', 'homogeneity', 'energy', \
            'correlation', 'ASM'}, optional
        The property of the GLCM to compute. The default is 'contrast'.
    Returns
    -------
    results : 2-D ndarray
        2-dimensional array. `results[d, a]` is the property 'prop' for
        the d'th distance and the a'th angle.
    References
    ----------
    .. [1] The GLCM Tutorial Home Page,
           http://www.fp.ucalgary.ca/mhallbey/tutorial.htm
    Examples
    --------
    Compute the contrast for GLCMs with distances [1, 2] and angles
    [0 degrees, 90 degrees]
    >>> image = np.array([[0, 0, 1, 1],
    ...                   [0, 0, 1, 1],
    ...                   [0, 2, 2, 2],
    ...                   [2, 2, 3, 3]], dtype=np.uint8)
    >>> g = greycomatrix(image, [1, 2], [0, np.pi/2], levels=4,
    ...                  normed=True, symmetric=True)
    >>> contrast = greycoprops(g, 'contrast')
    >>> contrast
    array([[0.58333333, 1.        ],
           [1.25      , 2.75      ]])
    """
    check_nD(P, 4, 'P')

    (num_level, num_level2, num_dist, num_angle) = P.shape
    if num_level != num_level2:
        raise ValueError('num_level and num_level2 must be equal.')
    if num_dist <= 0:
        raise ValueError('num_dist must be positive.')
    if num_angle <= 0:
        raise ValueError('num_angle must be positive.')

    # normalize each GLCM
    P = P.astype(np.float64)
    glcm_sums = np.apply_over_axes(np.sum, P, axes=(0, 1))
    glcm_sums[glcm_sums == 0] = 1
    P /= glcm_sums

    # create weights for specified property
    I, J = np.ogrid[0:num_level, 0:num_level]
    if prop == 'contrast':
        weights = (I - J) ** 2
    elif prop == 'dissimilarity':
        weights = np.abs(I - J)
    elif prop == 'homogeneity':
        weights = 1. / (1. + (I - J) ** 2)
    elif prop in ['ASM', 'energy', 'correlation']:
        pass
    else:
        raise ValueError('%s is an invalid property' % (prop))

    # compute property for each GLCM
    if prop == 'energy':
        asm = np.apply_over_axes(np.sum, (P ** 2), axes=(0, 1))[0, 0]
        results = np.sqrt(asm)
    elif prop == 'ASM':
        results = np.apply_over_axes(np.sum, (P ** 2), axes=(0, 1))[0, 0]
    elif prop == 'correlation':
        results = np.zeros((num_dist, num_angle), dtype=np.float64)
        I = np.array(range(num_level)).reshape((num_level, 1, 1, 1))
        J = np.array(range(num_level)).reshape((1, num_level, 1, 1))
        diff_i = I - np.apply_over_axes(np.sum, (I * P), axes=(0, 1))[0, 0]
        diff_j = J - np.apply_over_axes(np.sum, (J * P), axes=(0, 1))[0, 0]

        std_i = np.sqrt(np.apply_over_axes(np.sum, (P * (diff_i) ** 2),
                                           axes=(0, 1))[0, 0])
        std_j = np.sqrt(np.apply_over_axes(np.sum, (P * (diff_j) ** 2),
                                           axes=(0, 1))[0, 0])
        cov = np.apply_over_axes(np.sum, (P * (diff_i * diff_j)),
                                 axes=(0, 1))[0, 0]

        # handle the special case of standard deviations near zero
        mask_0 = std_i < 1e-15
        mask_0[std_j < 1e-15] = True
        results[mask_0] = 1

        # handle the standard case
        mask_1 = mask_0 == False
        results[mask_1] = cov[mask_1] / (std_i[mask_1] * std_j[mask_1])
    elif prop in ['contrast', 'dissimilarity', 'homogeneity']:
        weights = weights.reshape((num_level, num_level, 1, 1))
        results = np.apply_over_axes(np.sum, (P * weights), axes=(0, 1))[0, 0]

    return results
# end code from skimage.feature.texture

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

def noisy_peper(image,percent=0.07):
    amount=rd.uniform(percent*.8,percent*1.15)
    out=np.copy(image)
    num_pepper=np.ceil(amount*image.size)
    coords=[np.random.randint(0, i, int(num_pepper))
            for i in image.shape]
    out[tuple(coords)]=0
    return out

def getWindow(img,x,y,size):
    if size<=0:
        return np.empty([0,0])
    size_per_2=int(size/2)
    if x>size_per_2 and y>size_per_2 and x<img.shape[1]-size_per_2 and y<img.shape[0]-size_per_2:
        return img[y-size_per_2:y+size_per_2+1, x-size_per_2:x+size_per_2+1]
    else:
        return getWindow(img,x,y,size-1)

def maxDistOfContour(cnt):
    extreme_left=tuple(cnt[cnt[:,:,0].argmin()][0]) # boulos
    extreme_right=tuple(cnt[cnt[:,:,0].argmax()][0]) # bolsonaro
    extreme_top=tuple(cnt[cnt[:,:,1].argmin()][0])
    extreme_bottom=tuple(cnt[cnt[:,:,1].argmax()][0])
    extremes=(extreme_left,extreme_right,extreme_top,extreme_bottom)
    max_dist=0
    for a in range(len(extremes)):
        for b in range(a,len(extremes)):
            dist=math.dist(extremes[a], extremes[b])
            if dist>max_dist:
                max_dist=dist
    return max_dist

def approximateContour(cnt):
    # x,y,w,h = cv2.boundingRect(cnt)
    # return np.array([[x,y],[x+w,y],[x+w,y+h],[x,y+h]])
    return cv2.convexHull(cnt)

def getContoursOfImage(img,show_edges=False):
    img_edges=cv2.Canny(img,50,200) # edge detection
    if show_edges:
        cv2.imshow('Edge image', img_edges)
    cnts,hierarchy=cv2.findContours(img_edges,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
    return cnts

def greyToBinaryInline(img_grey,img_bin,threshold,copy=True):
    if copy:
        img_bin=img_grey.copy()
    for i in range(img_grey.shape[1]):
        for j in range(img_grey.shape[0]):
            img_bin[j][i]=255 if img_grey[j][i]>(255-threshold) else 0

def greyToBinary(img_grey,threshold):
    img_bin=img_grey.copy()
    greyToBinaryInline(img_grey,img_bin,threshold,copy=False)
    return img_bin

def ex1_updateThreshold(threshold,img_grey,img_bin):
    global ex1_image_contours
    greyToBinaryInline(img_grey,img_bin,threshold,False)
    cnts=getContoursOfImage(img_bin)
    img_draw=cv2.cvtColor(img_bin,cv2.COLOR_GRAY2BGR)
    font_size=1
    offset=(8*font_size,-8*font_size)
    thickness=2
    bgr_text_color=(255,255,255)
    bgr_cnt_color=(255,50,50)
    for i,cnt in enumerate(cnts):
        i=i+1
        x,y=cv2.minEnclosingCircle(cnt)[0] # find center
        x=int(x)
        y=int(y)
        cv2.putText(img_draw,str(i),(x-offset[0],y-offset[1]),cv2.FONT_HERSHEY_SCRIPT_SIMPLEX,font_size,bgr_text_color,thickness)
        cv2.drawContours(img_draw,[cnt],-1,bgr_cnt_color,thickness)
    print('Amount of objects: {}'.format(len(cnts)))
    ex1_image_contours=cnts
    cv2.imshow('Binary image', img_draw)

def ex1_mouse(event,x,y,flag,param):
    global ex1_image_contours
    if(event==cv2.EVENT_LBUTTONDOWN):
        for i,cnt in enumerate(ex1_image_contours):
            i=i+1
            if cv2.pointPolygonTest(approximateContour(cnt),(x,y),True)>=0: # check if point is inside contour
                area=cv2.contourArea(cnt)
                perimeter=cv2.arcLength(cnt,True)
                max_dist=maxDistOfContour(cnt)
                print('Object {}\n\tArea: {:.3f} px^2\n\tMax dist: {:.3f} px\n\tPerimeter: {:.3f} px'.format(i,area,max_dist,perimeter))
                break

def filterSmallContours(contours,area_threshold,dist_threshold,peri_threshold):
    new_contours=[]
    for cnt in contours:
        area=cv2.contourArea(cnt)
        perimeter=cv2.arcLength(cnt,True)
        max_dist=maxDistOfContour(cnt)
        if area>=area_threshold and max_dist>=dist_threshold and perimeter>=peri_threshold:
            new_contours.append(cnt)
    return new_contours

def ex3_mouse(event,x,y,flag,param):
    ex3_image_contours=param[0]
    img_draw=param[1]
    # color=(0,190,80)
    color=(rd.randint(0,256),rd.randint(0,256),rd.randint(0,256))
    if(event==cv2.EVENT_LBUTTONDOWN):
        for i,cnt in enumerate(ex3_image_contours):
            i=i+1
            if cv2.pointPolygonTest(approximateContour(cnt),(x,y),False)>=0: # check if point is inside contour
                area=cv2.contourArea(cnt)
                perimeter=cv2.arcLength(cnt,True)
                max_dist=maxDistOfContour(cnt)
                x,y=cv2.minEnclosingCircle(cnt)[0] # find center
                x=int(x)
                y=int(y)
                ellipse=cv2.fitEllipse(cnt)
                axis_angle=ellipse[2]
                axis_angle=axis_angle-90 if axis_angle > 90 else axis_angle+90
                ellipse_a=ellipse[1][0]
                ellipse_b=ellipse[1][1]
                if ellipse_b>ellipse_a:
                    ellipse_tmp=ellipse_a
                    ellipse_a=ellipse_b
                    ellipse_b=ellipse_tmp
                eccentricity=math.sqrt(1-(ellipse_b**2/ellipse_a**2))
                x_line_1=int(x+math.cos(math.radians(axis_angle+180))*(max_dist/2))
                y_line_1=int(y+math.sin(math.radians(axis_angle+180))*(max_dist/2))
                x_line_2=int(x+math.cos(math.radians(axis_angle))*(max_dist/2))
                y_line_2=int(y+math.sin(math.radians(axis_angle))*(max_dist/2))
                cv2.line(img_draw,(x_line_2,y_line_2),(x_line_1,y_line_1), color, 2) # draw main axis
                cv2.ellipse(img_draw,ellipse,color,1) # draw ellipse
                cv2.circle(img_draw,(x,y),5,color,-1,cv2.FILLED) # draw centroid
                print('Object {}\n\tArea: {:.3f} px^2\n\tMax dist: {:.3f} px\n\tPerimeter: {:.3f} px\n\tEccentricity: {:.3f}'.format(i,area,max_dist,perimeter,eccentricity))
                cv2.imshow('Grey image', img_draw)
                break

def ex1(path='objects.png'):
    ex1_image_contours=None
    img=cv2.imread(path) # open image
    img=resizeImg(img,400)
    img_grey=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY) # convert to grey
    cv2.imshow('Grey image', img_grey)
    img_bin=img_grey.copy()
    threshold=100
    ex1_updateThreshold(threshold,img_grey,img_bin)
    cv2.createTrackbar('threshold','Binary image',0,255,lambda x: ex1_updateThreshold(x,img_grey,img_bin))
    cv2.setTrackbarPos('threshold','Binary image',threshold)
    cv2.setMouseCallback('Binary image',ex1_mouse)
    cv2.waitKey()

def ex2(path='jackie.png'):
    kernel=3
    distances=[1]
    angles=[0,np.pi/4,np.pi/2,3*np.pi/4]
    iterations=30
    uniformities_smoth=[]
    uniformities_residual=[]
    homogeneities_smoth=[]
    homogeneities_residual=[]
    img=cv2.imread(path) # open image
    img=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY) # convert to grey
    img=resizeImg(img,256)
    img_smoth=img.copy()
    comatrix_img=greycomatrix(img,distances,angles)
    t=np.sum(comatrix_img)
    cv2.imshow('Original '+path,img)
    i=0
    while(i<iterations):
        i+=1
        print('Calculating iteration {} of {}'.format(i,iterations))
        img_smoth=cv2.boxFilter(img_smoth,-1,(kernel,kernel)) # filter
        img_residual=img-img_smoth
        comatrix_img_smoth=greycomatrix(img_smoth,distances,angles)
        comatrix_img_residual=greycomatrix(img_residual,distances,angles)
        uniformity_smoth=greycoprops(comatrix_img_smoth,'energy')/t
        uniformity_residual=greycoprops(comatrix_img_residual,'energy')/t
        homogeneity_smoth=greycoprops(comatrix_img_smoth,'homogeneity')/t
        homogeneity_residual=greycoprops(comatrix_img_residual,'homogeneity')/t
        uniformities_smoth.append(uniformity_smoth)
        uniformities_residual.append(uniformity_residual)
        homogeneities_smoth.append(homogeneity_smoth)
        homogeneities_residual.append(homogeneity_residual)
        cv2.imshow('Smooth {} - {}'.format(i,path),img_smoth)

    uniformities_smoth=np.transpose(uniformities_smoth,axes=(1,2,0)).tolist() # transform entry-distance-angle to distance-angle-entry
    uniformities_residual=np.transpose(uniformities_residual,axes=(1,2,0)).tolist() # transform entry-distance-angle to distance-angle-entry
    homogeneities_smoth=np.transpose(homogeneities_smoth,axes=(1,2,0)).tolist() # transform entry-distance-angle to distance-angle-entry
    homogeneities_residual=np.transpose(homogeneities_residual,axes=(1,2,0)).tolist() # transform entry-distance-angle to distance-angle-entry
    samples=list(range(iterations))
    i=0
    for angle_plot in angles:
        angle_plot=int(np.degrees(angle_plot))
        plt.plot(samples,uniformities_smoth[0][i],label='Uniformity Smooth - {}°'.format(angle_plot))
        plt.plot(samples,uniformities_residual[0][i],label='Uniformity Residual - {}°'.format(angle_plot))
        plt.plot(samples,homogeneities_smoth[0][i],label='Homogeneity Smooth - {}°'.format(angle_plot))
        plt.plot(samples,homogeneities_residual[0][i],label='Homogeneity Residual - {}°'.format(angle_plot))
        i+=1
    plt.legend(bbox_to_anchor=(1.05, 1.05), loc='upper left')
    plt.title('All Results - '+path) 
    plt.show()  
    i=0
    for angle_plot in angles:
        angle_plot=int(np.degrees(angle_plot))
        plt.plot(samples,uniformities_smoth[0][i],label='Uniformity Smooth - {}°'.format(angle_plot))
        plt.plot(samples,uniformities_residual[0][i],label='Uniformity Residual - {}°'.format(angle_plot))
        i+=1
    plt.legend(bbox_to_anchor=(1.05, 1.05), loc='upper left')
    plt.title('All Uniformoties Results - '+path) 
    plt.show()
    i=0
    for angle_plot in angles:
        angle_plot=int(np.degrees(angle_plot))
        plt.plot(samples,homogeneities_smoth[0][i],label='Homogeneity Smooth - {}°'.format(angle_plot))
        plt.plot(samples,homogeneities_residual[0][i],label='Homogeneity Residual - {}°'.format(angle_plot))
        i+=1
    plt.legend(bbox_to_anchor=(1.05, 1.05), loc='upper left')
    plt.title('All Homogeneities Results - '+path) 
    plt.show()
    cv2.waitKey()
            

def ex2_non_recursive_on_movie(path='jackie.mp4'):
    kernel=3
    distances=[1]
    angles=[0,np.pi/4,np.pi/2,3*np.pi/4]
    crop=30

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
    uniformities_smoth=[]
    uniformities_residual=[]
    homogeneities_smoth=[]
    homogeneities_residual=[]
    i=0
    while(vid.isOpened() and i<crop):
        ret, frame=vid.read() # gets fram
        if ret:
            start=time.time()
            i+=1
            print('Calculating frame {} of {}'.format(i,crop))
            img=resizeImg(frame,256)
            img=img[:,:,0] # remove third dimenssion
            # computations
            img_smoth=cv2.boxFilter(img.copy(),-1,(kernel,kernel)) # filter
            img_residual=img-img_smoth

            comatrix_img=greycomatrix(img,distances,angles)
            comatrix_img_smoth=greycomatrix(img_smoth,distances,angles)
            comatrix_img_residual=greycomatrix(img_residual,distances,angles)
            t=np.sum(comatrix_img)

            uniformity_smoth=greycoprops(comatrix_img_smoth,'energy')/t
            uniformity_residual=greycoprops(comatrix_img_residual,'energy')/t
            homogeneity_smoth=greycoprops(comatrix_img_smoth,'homogeneity')/t
            homogeneity_residual=greycoprops(comatrix_img_residual,'homogeneity')/t

            uniformities_smoth.append(uniformity_smoth)
            uniformities_residual.append(uniformity_residual)
            homogeneities_smoth.append(homogeneity_smoth)
            homogeneities_residual.append(homogeneity_residual)
            
            cv2.imshow('smooth video '+path,img_smoth)
            # respect fps
            delta=int(ms_p_f-(time.time()-start))
            if delta<=0:
                delta=1
            if cv2.waitKey(delta)==27: #escape key
                break
    vid.release()
    samples=list(range(crop))
    uniformities_smoth=uniformities_smoth[:crop]
    uniformities_residual=uniformities_residual[:crop]
    homogeneities_smoth=homogeneities_smoth[:crop]
    homogeneities_residual=homogeneities_residual[:crop]
    uniformities_smoth=np.transpose(uniformities_smoth,axes=(1,2,0)).tolist() # transform entry-distance-angle to distance-angle-entry
    uniformities_residual=np.transpose(uniformities_residual,axes=(1,2,0)).tolist() # transform entry-distance-angle to distance-angle-entry
    homogeneities_smoth=np.transpose(homogeneities_smoth,axes=(1,2,0)).tolist() # transform entry-distance-angle to distance-angle-entry
    homogeneities_residual=np.transpose(homogeneities_residual,axes=(1,2,0)).tolist() # transform entry-distance-angle to distance-angle-entry
    i=0
    for angle_plot in angles:
        angle_plot=int(np.degrees(angle_plot))
        plt.plot(samples,uniformities_smoth[0][i],label='Uniformity Smooth - {}°'.format(angle_plot))
        plt.plot(samples,uniformities_residual[0][i],label='Uniformity Residual - {}°'.format(angle_plot))
        plt.plot(samples,homogeneities_smoth[0][i],label='Homogeneity Smooth - {}°'.format(angle_plot))
        plt.plot(samples,homogeneities_residual[0][i],label='Homogeneity Residual - {}°'.format(angle_plot))
        i+=1
    plt.legend(bbox_to_anchor=(1.05, 1.05), loc='upper left')
    plt.title('All Results - '+path) 
    plt.show()  
    i=0
    for angle_plot in angles:
        angle_plot=int(np.degrees(angle_plot))
        plt.plot(samples,uniformities_smoth[0][i],label='Uniformity Smooth - {}°'.format(angle_plot))
        plt.plot(samples,uniformities_residual[0][i],label='Uniformity Residual - {}°'.format(angle_plot))
        i+=1
    plt.legend(bbox_to_anchor=(1.05, 1.05), loc='upper left')
    plt.title('All Uniformoties Results - '+path) 
    plt.show()
    i=0
    for angle_plot in angles:
        angle_plot=int(np.degrees(angle_plot))
        plt.plot(samples,homogeneities_smoth[0][i],label='Homogeneity Smooth - {}°'.format(angle_plot))
        plt.plot(samples,homogeneities_residual[0][i],label='Homogeneity Residual - {}°'.format(angle_plot))
        i+=1
    plt.legend(bbox_to_anchor=(1.05, 1.05), loc='upper left')
    plt.title('All Homogeneities Results - '+path) 
    plt.show()

def ex3(path='reservoir.png'):
    img=cv2.imread(path) # open image
    img_grey=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY) # convert to grey
    threshold=111
    img_bin=greyToBinary(img_grey,threshold)
    cv2.imshow('Bin image', img_bin)
    # morphological transformations to smooth image
    kernel_size=3
    kernel=cv2.getStructuringElement(cv2.MORPH_RECT,(kernel_size,kernel_size))
    close=cv2.morphologyEx(img_bin,cv2.MORPH_CLOSE,kernel,iterations=2)
    dilate=cv2.dilate(close,kernel,iterations=1)
    cv2.imshow('Morph image', dilate)
    img_bin=dilate
    # morphological transformations to smooth image
    contours=getContoursOfImage(img_bin)
    print('Contours before: {}'.format(len(contours)))
    contours=filterSmallContours(contours,7,7,400)
    print('Contours after: {}'.format(len(contours)))
    img_draw=cv2.cvtColor(img_grey,cv2.COLOR_GRAY2BGR)
    thickness=2
    font_size=1
    offset=(6*font_size,-8*font_size)
    bgr_text_color=(50,50,255)
    bgr_cnt_color=(255,50,50)
    for i,cnt in enumerate(contours):
        i=i+1
        x,y=cv2.minEnclosingCircle(cnt)[0] # find center
        x=int(x)
        y=int(y)
        # cv2.putText(img_draw,str(i),(x-offset[0],y-offset[1]),cv2.FONT_HERSHEY_SCRIPT_SIMPLEX,font_size,bgr_text_color,thickness)
        cv2.drawContours(img_draw,[cnt],-1,bgr_cnt_color,thickness)
        cv2.drawContours(img_draw,[approximateContour(cnt)],-1,(255,255,255),1)
    cv2.imshow('Grey image', img_draw)
    cv2.setMouseCallback('Grey image',ex3_mouse,(contours,img_draw))
    cv2.waitKey()

def ex4(path='beatrix.png'):
    lines_density=0.7
    background_density=0.07
    thickness=2
    dst_img_size=600
    color=(50,50,50)
    img_rnd=np.zeros((dst_img_size,dst_img_size,1),dtype=np.uint8)
    img_rnd.fill(255)
    rd.seed(time.time())
    amount_lines=rd.randint(8,15)
    x_min=-dst_img_size/1.5
    x_max=dst_img_size*1.5 
    y_min=-dst_img_size/1.5
    y_max=dst_img_size*1.5
    img_dense_background=noisy_peper(img_rnd,percent=lines_density)
    for i in range(amount_lines):
        x1=int(rd.uniform(x_min,x_max))
        x2=int(rd.uniform(x_min,x_max))
        y1=int(rd.uniform(y_min,y_max))
        y2=int(rd.uniform(y_min,y_max))
        cv2.line(img_rnd,(x2,y2),(x1,y1), color, 2)
    img_rnd=cv2.bitwise_not(img_rnd)
    img_dense_background=cv2.bitwise_not(img_dense_background)
    img_rnd=cv2.bitwise_and(img_rnd,img_dense_background)
    img_rnd=cv2.bitwise_not(img_rnd)
    img_rnd=noisy_peper(img_rnd,percent=background_density)
    img_hough_out=cv2.cvtColor(img_rnd.copy(), cv2.COLOR_GRAY2BGR)
    img_hough_in=255-img_rnd.copy() # invert
    lines=cv2.HoughLinesP(img_hough_in,1,np.pi/180,111,minLineLength=30,maxLineGap=15)
    for line in lines:
        for x1,y1,x2,y2 in line:
            cv2.line(img_hough_out,(x1, y1),(x2, y2),(0,255,0),thickness)
    cv2.imshow('Random generated image', img_rnd)
    cv2.imshow('Hough Lines image', img_hough_out)
    cv2.waitKey()

def main(argv):
    print('Select an exercise to run [1-4]:')
    ex=inputNumber(greater_or_eq=1,lower_or_eq=4)
    if ex==1:
        ex1()
    elif ex==2:
        ex2()
        ex2('outdoor_1.png')
        ex2('outdoor_2.png')
        # ex2_non_recursive_on_movie()
        # ex2_non_recursive_on_movie('outdoor_1.mp4')
        # ex2_non_recursive_on_movie('outdoor_2.mp4')
    elif ex==3:
        ex3()
    elif ex==4:
        ex4()
if __name__=='__main__':
    main(sys.argv[1:])