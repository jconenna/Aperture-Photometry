#!/usr/bin/python

###################################################
#Author:     Joseph Conenna                       #
#Professor:  Dr. Joe Harrington                   #
#Class:      AST4765 : Astronomical Data Analysis #
#Due Date:   ~                                    #
#Assignment: Project                              #
###################################################


###########################################
#                Inports                  #
###########################################

import numpy  as np
import pyfits as p 
import ds9
import project_routines as pr
import gaussian as g
import pickle
import matplotlib.pyplot as plt

###########################################
#         Predeclared Variables           #
###########################################

file_directory = '/home/ast5765/data/transit/'
file_prefix    = 'SPITZER_I4_20674048_0'
file_postfix   = '_0000_2_bcd.fits'
first_image    = 187
final_image    = 686
num_images     = final_image - first_image
num_parameters = 10
test_value = 'True4'
save = 'False'


### 4 - See data_collect in project_routines


### 5 - run routine from image 0187 to 0686 

#Photometry table, data cube, number of frames in data cube, and shape of the cube are fetched from data_collect.

photometry, data_cube, num_frames, cube_shape = pr.data_collect(file_directory, file_prefix, file_postfix, first_image, num_images, num_parameters)


# Photometry Parameters:

# 0 - Image Number from 189 - 686 incremented every 64 frames.
# 1 - Time frame was taken in UTC seconds. 
# 2 - median background for frame
# 3 - Center Y Coordinate
# 4 - Center X Coordinate
# 5 - Average Width
# 6 - Stellar Flux
# 7 - Average Sky
# 8 - Number of Bad Pixels


#!#! Sends frames to ds9 in rapid sequence if test_value is True_1.

if test_value == 'True_1': 
   
   d = ds9.ds9()
   
   for frame in np.arange(2500):
   
       d.set_np2arr(data_cube[frame]) 



### 6 - Makes a boolean mask array the same shape as the data array with all
# values set to 1. 


mask_array = np.ones(num_frames * cube_shape[1] * cube_shape[2])

mask_array.shape = ((num_images * cube_shape[0]), cube_shape[1], cube_shape[2])
print(mask_array.shape)


### 7 - Finds the median value in each frame, records this value as the median
# background to the photometry parameter 2, subtracts the value from each 
# frame.

for frame in np.arange(num_frames):

    photometry[frame][2] = np.median(data_cube[frame])
    
    data_cube[frame]     -= photometry[frame][2]



### 8 - See sigma_reject() in project_routines.py


### 9 - See driver() in project_routines.py

# Runs driver and saves resulting mask_array to original mask_array.
"""
mask_array = pr.driver(data_cube, mask_array, num_images, cube_shape)
"""

#!#! If statement if inspection of altered data by new mask is desired (True_2).
if(test_value == "True_2"):

  test_data = np.copy(data_cube)
  test_data *= mask_array
  d = ds9.ds9()
  for frame in np.arange(2500):
      d.set_np2arr(test_data[frame]) 

       
### 10 - Finds approximate location for first star by fitting a 2D Gaussian 
# to the data using a guess position and height at this location.

#PICKLE- Runs routine if save = 'True'.

if(save == 'True'):

  first_frame = 0

  cy     = 14.5
  cx     = 13.5

  # Begins routine to fit Gaussian to a sub_array to find fitted center.

  wid    =  1.0

  # Stores height by finding value at the center coordinates.
  
  ht     =  data_cube[first_frame, cy, cx]  
  
  # Extracts 10 x 10 sub array centered on star.

  sub_y  = 5
  sub_x  = 5      
 
  sub_array  = np.copy(data_cube[first_frame, cy - sub_y: cy + sub_y +1, cx - sub_x: cx + sub_x + 1])
  
       
  # Initializes tuple of data for fitgaussuan.

  guess = ((wid, wid), (sub_y, sub_x), ht)

  # Stores return values from called fitgaussian

  (fw, fc, fh, fe) = g.fitgaussian(sub_array, guess=guess)

  # Stores average width

  photometry[first_frame][5] = np.mean(fw)
  print(np.mean(fw))

  # Adjusts the fitted centre coordinates back to the full array.
  # Stored accordingly.

  photometry[first_frame][3] = fc[0] + cy - sub_y
  photometry[first_frame][4] = fc[1] + cx - sub_x 
  


  # Uses center of previous frame as guess for next frame to fit a 2D Gaussian 
  # to the remaining frames.

  for frame in np.arange(1, num_frames):
    
    
      cy     = photometry[frame - 1][3]
      cx     = photometry[frame - 1][4]

      wid    =  1
    
      # Stores height by finding value at the center coordinates.
  
      ht     =  data_cube[first_frame, cy, cx]  
     
      # Extracts 10 x 10 sub array centered on star.

      sub_y  = 5
      sub_x  = 5     
 
      sub_array  = np.copy(data_cube[frame, cy - sub_y: cy + sub_y + 1, cx - sub_x: cx + sub_x + 1])
       
      # Initializes tuple of data for fitgaussuan.

      guess = ((wid, wid), (sub_y, sub_x), ht)

      # Stores return values from called fitgaussian

      (fw, fc, fh, fe) = g.fitgaussian(sub_array, guess=guess)

      # Stores average width

      photometry[frame][5] = np.average(fw)
      print(np.average(fw))
      
      
      # Adjusts the fitted centre coordinates back to the full array.
      # Stored accordingly.

      photometry[frame][3] = fc[0] + cy - sub_y
      photometry[frame][4] = fc[1] + cx - sub_x 
      
      print(frame, photometry[frame][3],photometry[frame][4])

  # Pickles photometry after execution of above routine.

  output = open('photometry.pkl', 'wb')

  pickle.dump(photometry, output)

  output.close()


#PICKLE- Reads in photometry data from saved pickle from Gaussian fitting routine.

if(save == 'False'):

  input = open('photometry.pkl', 'rb')

  photometry = pickle.load(input)



### 11 The offset from the companion star is approximated to be (-8, -5) from 
# the approximate center. A disk of radius 4 is applied to mask the companion
# star located the offset away from the fitted center of the main star. 

offset = (-8, -5)

#PICKLE- Routine runs if save = 'True'.

if(save == 'True'):
  for frame in np.arange(num_frames):
    
      # Sets paramters for disk()
      radius = 4    
      size   = (32, 32)
      center = (photometry[frame][3] + offset[0], photometry[frame][4] + offset[1])
    
      # Obtains mask array
      companion_mask = pr.disk(radius, center, size)

      # Inverts values in mask so that all False values correspond to the companion star.
      companion_mask -= 1
      companion_mask *= -1
 
      # Multiplies companion mask by corresponding slice of mask_array
      mask_array[frame] *= companion_mask

      print(frame)


  output = open('mask.pkl', 'wb')

  pickle.dump(mask_array, output)
  output.close()



#PICKLE- Reads in mask data from saved pickle.

if(save == 'False'):

  input = open('mask.pkl', 'rb')

  mask_array = pickle.load(input)

#!#! If statement if inspection of altered data by new mask is desired.
if test_value == "True6":
   test_data = np.copy(data_cube)
   test_data *= mask_array
   d = ds9.ds9()
   for frame in np.arange(2000):
       d.set_np2arr(test_data[frame]) 



### 12 Performs Aperture Photometry on the data cube using dophot() and apphot().

#PICKLE- Runs routine if save = 'True'. 

if(save == 'True'):


  ''' Sets variables for photometry aperture radius, two sky annuli, and sub image size as multipliers of the fitted width. The photometry aperture radius, `aperture radius`, should be 3 so that signal for up to 3 sigma is utilized. The inner sky annulus, `inner_annulus`, will be 5 so that is far enough to not get any flux from the star. The outer sky annulus, `outer_annulus`, will be 8 so that there is less uncertainty in the mean and small enough so that the sky doesn't vary or include other stars. The sub array size, `sub_size`, shall be 10 to ensure that there is enough sky centered about the center of the star to utilize. '''



  aperture_radius  = 4
  inner_annulus    = 6
  outer_annulus    = 9
 

  # Does aperture photometry on the data cube. Obtains total flux, average sky
  # and the number of bad pixels in the aperture and stores in photometry.

  pr.dophot(photometry, data_cube, num_frames, mask_array, outer_annulus, inner_annulus, aperture_radius)
  
  # Pickles photometry after execution of above routine.

  output = open('photometry.pkl', 'wb')

  pickle.dump(photometry, output)

  output.close()


#PICKLE- Reads in photometry data from saved pickle from Gaussian fitting routine.

if(save == 'False'):

  input = open('photometry.pkl', 'rb')

  photometry = pickle.load(input)



### 14- 
"""Check photometric outputs by  plotting photometry only for good frames. Plotting the raw data shows several outliers. These outliers are removed from the plot by not plotting any flux over a certain
value which I eyeballed to be 9700. Frames with a background about 7 times the standard deviation of the data from the median for average sky values are rejected. Photometry where the aperture contained a bad pixel are also rejected in the plot. The total of frames rejected is 56. 
"""
# Reads in background data from photometry parameter 7 for each frame.
bkgrnd = np.zeros(num_frames)
for frame in np.arange(num_frames):
    bkgrnd[frame] = photometry[frame][7]

# Calculates the median and standard deviation for the data.

bkmed = np.median(bkgrnd)

bkstd = np.std(bkgrnd)

#Initializes array for integrated flux values.
flux = np.zeros(num_frames)


for frame in np.arange(num_frames):

    # Rejects outliers above 9700
    if(photometry[frame][6] > 9700):
      photometry[frame][8] = -1

    # Rejects data from outside 5 sigma from the median.
    if((np.abs(photometry[frame][7]-bkmed)/bkstd) > 5.):
      photometry[frame][8] = -1
    
    # If the frame has been flagged or has bad pixels, mask as np.nan 
    # so that the data isn't plotted
    if(photometry[frame][8] != 0):
      flux[frame] = np.nan
 
    # Reads in good flux data.
    else:
      flux[frame] = photometry[frame][6]

 
# Plots data.
   
plt.plot(np.arange(num_frames), flux, "c.")
fig = plt.gcf()
fig.set_size_inches(12.5,6.5)
plt.title("Integrated Stellar Flux")
plt.xlabel("Frame number")
plt.ylabel("Flux")
plt.savefig("project_jcon_stellar_flux.png") 



### 15 Bins data into 150 bins. 

flux_avg = np.zeros(150)

for i in np.arange(150):
    flu = []
    for j in np.arange(212):

        # Makes sure not to bin flagged frame data (nan)
        if(np.isfinite(flux[(i*212)+j])):
          flu = np.append(flu,flux[(i*212)+j])
    # Calculates bin average.
    flux_avg[i] = np.mean(flu)

# Plots averages of binned data

plt.clf()
plt.plot(np.arange(150),flux_avg,".")
fig = plt.gcf()
fig.set_size_inches(12.5,6.5)
plt.title("Binned Average Stellar Flux")
plt.xlabel("Bin Number")
plt.ylabel("Average Flux")
plt.savefig("project_jcon_binned_flux.png", orientation='landscape') 

#Saves plot of stellar flux and binned averages on same scale.
plt.clf
plt.plot(np.arange(num_frames), flux, "c.")
plt.plot(np.arange(0,num_frames,213),flux_avg,"ro")
fig = plt.gcf()
fig.set_size_inches(12.5,6.5)
plt.title("Integrated Stellar Flux")
plt.xlabel("Frame number/ bin loc")
plt.ylabel("Flux")
plt.savefig("project_jcon_compare.png", orientation='landscape') 



### 16 - Manually identifies six frames where the following happens,
# using an interactive plot of the previous plot. 

# Data start - frame 00000
data_start     = 0

# First contact (start of transit) - frame 09794
first_contact  = 9794

# Second contact (planet fully on stellar disk) - frame 12800
second_contact = 12800

# Third contact (start of egress) - frame 22160
third_contact  = 22160

# Fourth contact (end of transit) - frame 25000
fourth_contact = 25000

# Data end - frame 31936
data_end       = 31936



### 17 Fits parbola to flux vs. time data outside of transit. 


# Extracts time data

time_out = np.zeros((first_contact - data_start) + (data_end - fourth_contact))

for frame in np.arange(data_start, first_contact):
    time_out[frame] = photometry[frame][1]

for frame in np.arange(data_end - fourth_contact):
    time_out[first_contact + frame] = photometry[fourth_contact + frame][1]


#Extracts flux data

flux_out = np.zeros((first_contact - data_start) + (data_end - fourth_contact))

for frame in np.arange(data_start, first_contact):
    flux_out[frame] = photometry[frame][6]
    
for frame in np.arange(data_end - fourth_contact):
    flux_out[first_contact + frame] = photometry[fourth_contact + frame][6]



# Obtains corrected poly coefficients by minimizing squared error.
corrected = np.polyfit(time_out, flux_out, deg=2)

# Constructs polynomial using corrected coefficients.
parabola = np.poly1d(corrected)


# Divides flux value by corrected value by evaluating the fitted 
# parabola at each corresponding time to normalize the data.

for frame in np.arange(num_frames):
    flux[frame] /= parabola(photometry[frame][1])


# Plots corrected data.   
plt.clf()
plt.plot(np.arange(num_frames), flux, "c.")
fig = plt.gcf()
fig.set_size_inches(12.5,6.5)
plt.title("Corrected And Normalized Flux")
plt.xlabel("Frame number")
plt.ylabel("Flux")
plt.savefig("project_jcon_corrected.png") 

# Saves plot of corrected data and binned averages.

flux_avg = np.zeros(150)

for i in np.arange(150):
    flu = []
    for j in np.arange(212):
        # Makes sure not to bin flagged frame data (nan)
        if(np.isfinite(flux[(i*212)+j])):
          flu = np.append(flu,flux[(i*212)+j])
    flux_avg[i] = np.mean(flu)


#Saves plot of corrected stellar flux and binned averages.
plt.clf
plt.plot(np.arange(num_frames), flux, "c.")
plt.plot(np.arange(0,num_frames,213),flux_avg)
fig = plt.gcf()
fig.set_size_inches(12.5,6.5)
plt.title("Corrected Flux")
plt.xlabel("Frame number/ bin loc")
plt.ylabel("Flux")
plt.savefig("project_jcon_compare_corrected.png", orientation='landscape') 



### 18 Uses flat run of frames 29469 to 29569 avoiding long-period
# ripples to calculate the standard error for the stellar photometry
# level.

standard_error = np.std(flux[29469:29569]) 

# Calculates the signal to noise for thise range of frames.

snr = np.mean(flux[29469:29569]) / standard_error

print(standard_error,snr)

# Obtains standard error of 0.01562 and S/N of 63.932.


### 19 Calculates planet cross-sectional area to stellar cross-sectional area. Ignores bad frames.

""" The area of a star with radius R is pi*R^2 whereas the area of a plent with radius r is pi*r^2. The area of the star is proportional to the mean flux of the star. The area of the planet is proportional to the mean flux of the star minus the mean flux of the star when the planets area is within the star's area. Ap/As = (Fs - Fp) / Fs .
This becomes (r/R)^2 = (Fs - Fp) / Fs. 

r/R = sqrt((Fs - Fp) / Fs) """ 


# Calculates mean flux of star with uncertainty.

flux_star = []

for frame in np.arange(first_contact - data_start):
    # Takes in only good data
    if(np.isfinite(flux[frame])):
      flux_star = np.append(flux_star, flux[frame])

for frame in np.arange(fourth_contact, data_end):
    # Takes in only good data
    if(np.isfinite(flux[frame])):
      flux_star = np.append(flux_star, flux[frame])

# Converts appended list to array
flux_star = np.array(flux_star)

# Creates array or uncertainties by multiplying flux array by standard error.
unc_flux_star = flux_star * standard_error

# Calculates uncertainty of the mean of the star flux.
unc_flux_s_mean = np.sqrt(np.sum(unc_flux_star**2)) / np.sqrt(np.size(flux_star))

# Calculates the mean of the star flux.
flux_s_mean = np.mean(flux_star)


# Calculates mean flux of planet with uncertainty.

flux_planet = []

for frame in np.arange(second_contact, third_contact):
    # Takes in only good data
    if(np.isfinite(flux[frame])):
      flux_planet = np.append(flux_planet, flux[frame])

# Converts list to array
flux_planet = np.array(flux_planet)

# Creates array of uncertainties by multiplying flux array by standard error.
unc_flux_pl = flux_planet * standard_error

# Calculates uncertainty of the mean of planet flux.
unc_flux_pl_mean = np.sqrt(np.sum(unc_flux_pl**2)) / np.sqrt(np.size(flux_planet))

# Calculates the mean of the planet flux.
flux_pl_mean = np.mean(flux_planet)



# Caculates ratio of radius of planet to radius of star with 
# uncertainty.

ratio = flux_s_mean - flux_pl_mean

ratio /= flux_s_mean


# Calculates the uncertainty of this calculation
ratio_unc = np.sqrt(unc_flux_s_mean**2 + unc_flux_pl_mean**2)

ratio_unc = ratio * np.sqrt(ratio_unc**2 + unc_flux_s_mean**2)

# Gets ratio of 0.02454 +/- 0.000658



### 20 Multiplies ratio and uncertainty by radius and uncertainty of the star HD 189733. Values in radius of sun.


radius_pl = ratio * 0.781

radius_pl_unc = radius_pl * np.sqrt(ratio_unc**2 + 0.051**2)


# Radius of planet HD 189733b is 0.0191686 +/- 0.000977 R_Sun.

