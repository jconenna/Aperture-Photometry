
"""
Routines for project on aperture photometry

data_collect()
sigma_reject()
driver()
disk()
apphot()
dophot()

"""

import numpy as np
import pyfits as p


"""
    data_collect()    

    This extracts data from files containing image cubes and header files. A 3D cube of frames is initialized and later filled with data. A 2d Parameter array is initialized.

    The routine first extracts the data and header from the first block file to get dimensions of the file arrays. A 3D data cube `data_cube` is allocated using np.zeros(). A 2D parameter array `photometry` is allocated using a routine to fill such an array with [None] values. The data from a range of files is then extracted and placed in `data_cube` while the appropriate times of each frame is determined and placed in `photometry` with the corresponding image number. 

    Parameters
    ----------
    file_directory: string
            A string containing the directory that the files are located within.

    file_prefix:    string
            A string containg the appropriate string for the file name before an integer variable is concatenated.

    file_postfix:   string
            A string containing the appropriate string for the file name after the integer variable for the image number is concatenated.

    first_image:    integer
            An integer value for the lower end of the range of image numbers.

    num_images:     integer
            An integer value for the number of images to work on from the first image number.

    num_paramters:  integer
            An integer value for the number of paramters to be preallocated for each frame in the array `photometry`.

  

    Returns
    -------
    photometry: ndarray, 2D
	The 2D array containing the preallocation for parameters to be filled at a later time.
   
    data_cube:  ndarray, 3D
        The 3D array preallocated with zeros that is then filled with the data from the file range.

    num_frames:  integer
        An integer value of the number of frames contained in the 3D frame cube `data_cube`.

    cube_shape:  list
        A list containing the dimensions of an image block file with the number of frames per image in index 0 and the y,x dimensions in indexes 1,2. 


    Examples
    --------
    >>>photometry, data_cube, num_frames, cube_shape = pr.data_collect(file_directory, file_prefix, file_postfix, first_image, num_images, num_parameters)

    >>>print(photometry[0], data_cube[0], num_frames, cube_shape)

       ([187, 215354061.366, None, None, None, None], array([[ 3.15738058,  3.10227704,  2.0291872 , ..., -1.37925923,
         1.93387616,  1.95367968],
       [ 3.16099381,  2.42469811,  1.37589574, ..., -0.13898581,
         6.24502993,  1.34681213],
       [ 5.70151901,  3.04282498,  1.37344515, ...,  1.71073639,
         2.57766867,  1.3778038 ],
       ..., 
       [ 3.09476733,  1.75190687,  0.71945632, ...,  1.74242222,
         1.97999918,  1.36883926],
       [ 1.20120168,  2.41319132, -1.16971123, ...,  2.37498522,
         2.61092544, -0.51742119],
       [-0.07284214,  5.50754642,  4.4745512 , ...,  2.98323894,
         1.94193757,  3.23147297]]), 31936, (64, 32, 32))

    Revisions
    ---------

    2011-11-22  0.1  Joseph A Conenna- First Draft

"""

def data_collect(file_directory, file_prefix, file_postfix, first_image, num_images, num_parameters):

    # Extracts data and header from first image block, gets dimensions image block
    # Determines the number of frames total for the file range

    x = p.getdata(file_directory + file_prefix + str(first_image) + file_postfix)

    h = p.getheader(file_directory + file_prefix + str(first_image) + file_postfix)

    cube_shape = x.shape

    num_frames = num_images * cube_shape[0]

   
    # Preallocates the 3d data cube
 
    data_cube = np.zeros((num_frames, cube_shape[1], cube_shape[2]))


    # Preallocates the 2D parameter array: (number of frames,  number of 
    # parameters)

    photometry = [None]*((num_images) * cube_shape[0])  

    for i in np.arange(num_frames):
        photometry[i] = [None] * num_parameters

 
    # Loop for each image cube, extracts frame data to data cube and updates
    # photometry parameters.

    image_num = 0

    for image in np.arange(num_images):

        data = p.getdata(file_directory + file_prefix + str(first_image + image) + file_postfix)

        hdr = p.getheader(file_directory + file_prefix + str(first_image + image) + file_postfix)
    
        # Reads in time data from header
        time     = hdr['UTCS_OBS']
        interval = hdr['FRAMTIME'] 
    
        for i in np.arange(cube_shape[0]):
         
            # Reads in each frame in the image to the data cube
            data_cube[image_num + i] = data[i]
     
            # Writes image number to photometry parameter 0 for each frame
            photometry[image_num + i][0] = first_image + image_num
       
            # Writes time the frame was taken to photometry parameter 1
            photometry[image_num + i][1] = (interval * i) + time
        
        image_num += cube_shape[0]

    return(photometry, data_cube, num_frames, cube_shape)


"""
    sigma_reject()    

    This routine takes in data and returns a mask of locations where the value is greater than 5 sigma from the median.

    The routine first calculates the median and standard deviation from a copy of the data. A mask is then created using ma.masked_where for the condition [(data_copy-median)/std < 5*sigma] where sigma equals one. The mask is then multiplied with the data_copy and the routine is iterated again. The final mask is then returned.
    
    Parameters
    ----------
    data: ndarray, array like
	An array of data.

  
    Returns
    -------
    the_mask: ndarray, array like. 
        A mask of boolean values of the same dimension as `data`.


    Examples
    --------
    >>> import project_routines as pr

    >>> mean = 1

    >>> sigma = 2

    >>> size = 10

    >>> x = np.random.normal(mean, sigma, size)

    >>> pr.sigma_reject(x)
        array([ True,  True,  True,  True,  True,  True,  True,  True,  True,  True], dtype=bool)



    Revisions

    ---------
    2011-11-22  0.1  Joseph A Conenna- First Draft

"""

def sigma_reject(data):
    
    
    data_copy = np.copy(data)
    
    # Takes median and standard deviation.
    median = np.median(data_copy.flat)
    std = np.std(data_copy.flat)
   
    # Gets mask of false values where data is higher than 5 sigma from 
    # The median.
    the_mask = np.ma.masked_where(np.abs((data_copy-median)/std) < 5., data_copy)
    
    the_mask = np.ma.getmask(the_mask)
    
    # Multiplies the data copy by the first mask.
    data_copy *= the_mask
    
    # Second iteration

    median = np.median(data_copy[np.where(the_mask != 0)])

    std = np.std(data_copy[np.where(the_mask != 0)])

    mask_2 = np.ma.masked_where(np.abs((data_copy-median)/std) < 5., data_copy)
    
    mask_2 = np.ma.getmask(mask_2)

    # Multiplies the first mask by the second 
    the_mask *= mask_2

    return(the_mask)


"""
    driver()    

    This routine runs sigma_reject() on the data cube. 

    This routine contains two loops, the first jumps over 64 frames in `data_cube`, while the inner work at every y and x location in the frames. The routine sigma_reject() is run on a column of 64 values at every y and x location and the resulting mask is put into the mask_array.
    
    Parameters
    ----------

    data_cube:  ndarray
             A 3D array containing data to be used.

    mask_array: ndarray
             A 3D array containing boolean values.

    num_images: integer
             An integer value containing the number of images.

    ube_shape:  list
        A list containing the dimensions of an image block file with the number of frames per image in index 0 and the y,x dimensions in indexes 1,2.              
  
    Returns
    -------
    mask_array: ndarray, array like. 
        A mask of boolean values of the same dimension as `data`. Now contains False values where data was sigma rejected. 


    Examples
    --------
    >>> mask_array = pr.driver(data_cube, mask_array, num_images, cube_shape)

    >>> print(mask_array, np.mean(mask_array))

        (array([[[ 1.,  1.,  1., ...,  1.,  1.,  1.],
        [ 1.,  1.,  1., ...,  1.,  1.,  1.],
        [ 1.,  1.,  1., ...,  1.,  1.,  1.],
        ..., 
        [ 1.,  1.,  1., ...,  1.,  1.,  1.],
        [ 1.,  1.,  1., ...,  1.,  1.,  1.],
        [ 1.,  1.,  1., ...,  1.,  1.,  1.]],

       [[ 1.,  1.,  1., ...,  1.,  1.,  1.],
        [ 1.,  1.,  1., ...,  1.,  1.,  1.],
        [ 1.,  1.,  1., ...,  1.,  1.,  1.],
        ..., 
        [ 1.,  1.,  1., ...,  1.,  1.,  1.],
        [ 1.,  1.,  1., ...,  1.,  1.,  1.],
        [ 1.,  1.,  1., ...,  1.,  1.,  1.]],

       [[ 1.,  1.,  1., ...,  1.,  1.,  1.],
        [ 1.,  1.,  1., ...,  1.,  1.,  1.],
        [ 1.,  1.,  1., ...,  1.,  1.,  1.],
        ..., 
        [ 1.,  1.,  1., ...,  1.,  1.,  1.],
        [ 1.,  1.,  1., ...,  1.,  1.,  1.],
        [ 1.,  1.,  1., ...,  1.,  1.,  1.]],

       ..., 
       [[ 1.,  1.,  1., ...,  1.,  1.,  1.],
        [ 1.,  1.,  1., ...,  1.,  1.,  1.],
        [ 1.,  1.,  1., ...,  1.,  1.,  1.],
        ..., 
        [ 1.,  1.,  1., ...,  1.,  1.,  1.],
        [ 1.,  1.,  1., ...,  1.,  1.,  1.],
        [ 1.,  1.,  1., ...,  1.,  1.,  1.]],

       [[ 1.,  1.,  1., ...,  1.,  1.,  1.],
        [ 1.,  1.,  1., ...,  1.,  1.,  1.],
        [ 1.,  1.,  1., ...,  1.,  1.,  1.],
        ..., 
        [ 1.,  1.,  1., ...,  1.,  1.,  1.],
        [ 1.,  1.,  1., ...,  1.,  1.,  1.],
        [ 1.,  1.,  1., ...,  1.,  1.,  1.]],

       [[ 1.,  1.,  1., ...,  1.,  1.,  1.],
        [ 1.,  1.,  1., ...,  1.,  1.,  1.],
        [ 1.,  1.,  1., ...,  1.,  1.,  1.],
        ..., 
        [ 1.,  1.,  1., ...,  1.,  1.,  1.],
        [ 1.,  1.,  1., ...,  1.,  1.,  1.],
        [ 1.,  1.,  1., ...,  1.,  1.,  1.]]]), 0.99999223300115858)


    Revisions
    ---------

    2011-11-22  0.1  Joseph A Conenna- First Draft

"""

def driver(data_cube, mask_array, num_images, cube_shape):
    
    jump = 64

    for image in np.arange(num_images):
        
        #Sets range for frames to be used in slicing

        low = image * jump
        high = low + jump
        

        for y in np.arange(cube_shape[1]):
            for x in np.arange(cube_shape[2]):  
                
                res = sigma_reject(data_cube[low:high, y, x])
                mask_array[low:high, y, x] = res
        
        # Prints frame worked on.
        print(i)

    return(mask_array)


### disk() : *** Copying routine for efficiency.

'''
	Makes a bool array containing an N-dimensional ellipsoid mask.

    Parameters
    ----------
    r : scalar or N-dimensional tuple     
        The radii of the ellipsoid, may be fractional.  If N-dimensional,
        elliptical radii are specified in each dimension.  If scalar,
        same radius applies to all dimensions. 
    center : tuple
        Gives the position of the center of the ellipsoid, may be
        fractional and of any dimension.  Note that if the desired
        "ellipsoid" is 1D, specifying (20) on the command line results
        in an int, not a tuple containing an int.  Say (20,) to force a
        tuple containing an int.
    shape :  tuple, int
        Gives the shape of the output array.  Must be integer and same
        length as center.

    Returns
    -------
    output : boolean array
    	This function returns a bool array containing an N-dimensional
    	ellipsoid (line segment, filled ellipse, ellipsoid, etc.).
    	The ellipsoid is centered at center and has the radii given by
    	r.  Shape specifies the shape.  The type is bool.  Array
    	values of 1 indicate that the center of a pixel is within the
    	given ellipsoid.  Pixel values of 0 indicate the opposite.
    	The center of each pixel is the integer position of that
    	pixel.

    Examples
    --------
    >>> import matplotlib.pyplot as plt
    >>> import disk
    >>> r      = (110, 150)
    >>> center = (230, 220)
    >>> shape  = (600, 500)
    >>> plt.imshow(disk.disk(r, center, shape), origin='lower', interpolation='nearest')
    >>> plt.gray()

    Revisions
    ---------
	2003-04-04 0.1  jh@oobleck.astro.cornell.edu Initial version.
	2007-11-25 0.2  jh@physics.ucf.edu           IDL->Python, made N-dim.
	2008-11-04 0.3 	kstevenson@physics.ucf.edu   Updated docstring.
	2010-10-26 0.4  jh@physics.ucf.edu           Updated docstring.
  '''
import numpy as np

def disk(radius, center, shape):

    idisk      = np.indices(shape, dtype=float)
    cctr       = np.asarray(center)
    sshape     = np.ones(1 + cctr.size)
    sshape[0]  = cctr.size
    cctr.shape = sshape
    rr         = np.asarray(radius)
    rshape     = np.ones(1+rr.size)
    rshape[0]  = rr.size
    rr.shape   = rshape

    return np.sum(((idisk - cctr)/rr)**2, axis=0) <= 1.



"""
    apphot()

    This function performs aperture photometry on a stellar source in an array of data.

    The function disk is called twice to create a sky annulus of True values. The mask is then multiplied to the sky annulus such that only good pixels are true. The data is then multiplied by the sky annulus and the average is taken where values are not False. This value is then subtracted from the data. The disk function is then used to create a photometry aperture. The total flux within this aperture is then computed. A Tuple containing the total flux, average sky, and number of bad pixels in the aperture is returned.

    Parameters 
    ----------
    data:             ndaray
                      2D array containing the data to perform aperture photometry on.

    mask:             ndarray
                      2D mask array of boolean values.
 
    cy, cx:           float or int
	              Values giving the approximate centre of the star in the frame in the form (y, x)

    outer_annulus:    float or int
                      Radius of outer sky annulus to be passed to disk()
    
    inner_annulus:    float or int
                      Radius of inner sky annulus to be passed to disk()

    aperture_radius:
                      float or int
                      Radius of photometry aperture to be passed to disk()


    Returns 
    -------
    tuple: (`stellar_flux`, `average_sky`, 'bad_pixels')
           Tuple containing floating point values of the total flux, average sky values, and number of bad pixels.

    Examples
    --------
    >>> apphot(data[0], sub_size = 18, cy = 689, cx = 512, outer_annulus = 8, inner_annulus = 5, aperture_radius = 3)
        (16884.388133407352, 421.1711827561993, 0)



    Revisions
    ---------
    2011-11-06  0.1  Joseph A Conenna- First Draft

    2011-11-30  0.2  Joseph A Conenna- Adds calculation for bad pixels, removed sub-array around center.
  """



def apphot(data, mask_array, cy, cx, outer_annulus, inner_annulus, aperture_radius):
   
    # Makes a copy of the data to work on. 
    data_copy = np.copy(data)
   
    # Determines shape of data array.
    shape = np.shape(data_copy)
    
    # Makes mask for sky annulus by subtracting a disk mask of radius
    # inner_annulus from one of radius outer_annulus.
    
    sky_annulus =  disk(outer_annulus, (cy, cx), (shape[0], shape[1]))
    
    sky_annulus -= disk(inner_annulus, (cy, cx), (shape[0], shape[1]))


    #Multiplies sky_annulus by mask_array so all good pixels are True

    sky_annulus *= mask_array


    # Multiplies sky_annulus by data_copy and find average value to find
    # average sky pixel in annulus. Uses np.where to ensure mean is taken
    # over True values.
    
    sky_annulus *= data_copy
    
    average_sky = np.mean(data_copy[np.where(sky_annulus != 0)])


    # Subtracts the average sky from each pixel in the data copy.

    data_copy  -= average_sky 


    # Creates a mask for the aperature photometry.

    ap_mask = disk(aperture_radius, (cy, cx), (shape[0], shape[1]))


    # Calculates the total flux in the aperture. 
    
    aperture = ap_mask * data_copy

    stellar_flux = np.sum(aperture[np.where(aperture != 0)])


    #Calculates the number of bad pixels in the aperture.

    aperture = data_copy * ap_mask * mask_array
    bad_pixels = np.abs( (np.size(aperture) - np.count_nonzero(aperture)) - (np.size(ap_mask) - np.count_nonzero(ap_mask)) )
    

    return (stellar_flux, average_sky, bad_pixels)



"""
   dophot()

   This function performs aperture photometry on a stellar sources in a 3D array of data.

   The routine multiplies the radii passed in for the masks by the average width from the photometry table. A loop works on each frame  by calling apphot() with centres found in the passed in photometry table.

    Parameters 
    ----------
    photometry:       ndarray
                      3D array containing photometric data that will be appended.

    data_cube:        ndaray
                      3D array containing the data to perform aperture photometry on.

    num_frames:       int 
                      Integer value of number of frames to work on in data_cube

    mask_array:       ndarray
                      3D array containing masked values for bad pixels.

    outer_annulus:    float or int
                      Radius of outer sky annulus to be passed to disk()
    
    inner_annulus:    float or int
                      Radius of inner sky annulus to be passed to disk()

    aperture_radius:
                      float or int
                      Radius of photometry aperture to be passed to disk()


    Returns 
    -------
    None.

    Examples
    --------
    >>> dophot.dophot(photometry, data_cube, outer_annulus, inner_annulus, aperture_radius)


    Revisions
    ---------
    2011-11-06  0.1  Joseph A Conenna- First Draft

    2011-11-30  0.2  Joseph A Conenna- Modified for Transit Project.

  """



def dophot(photometry, data_cube, num_frames, mask_array, outer_annulus, inner_annulus, aperture_radius):
    

    for frame in np.arange(num_frames):
         
        #Multiplies radii by average width for frame.

        aperture_radius  = 4  * photometry[frame][5]
        inner_annulus    = 6  * photometry[frame][5]
        outer_annulus    = 9  * photometry[frame][5]

   
        # Calls apphot for stars in frame f and saves data
        ret = apphot(data_cube[frame], mask_array[frame], photometry[frame][3], photometry[frame][4], outer_annulus, inner_annulus, aperture_radius)
        
        photometry[frame][6] = ret[0]
        photometry[frame][7] = ret[1]
        photometry[frame][8] = ret[2]  

        print(frame) 
