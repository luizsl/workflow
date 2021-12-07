# Notes

Notes of activities. Each activity is described along with its application date (YYYY-MM-DD).

## 2021-06-18

### Implementation of notes
I didn't take notes of the previous steps. I think nothing very crucial was lost but from now on I will record the daily progress.

### Implementation of the directory architecture. 

I have spent some time with the organization here, problaby it's going to be usefull for such long-term project. **My future self is going to be happy**. The architecture of the directory is described in the file README.md. I followed some ideias reported in the following articles which were adapted based on the previous research projects I participated. 

[post](https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1000424#s1)
[post](https://towardsdatascience.com/the-gold-standard-of-data-science-project-management-13d68c9e85d6)
[paper](https://aosmith.rbind.io/2018/10/29/an-example-directory-structure/)

## 2021-06-19

### Implementation of standard file for plot configuration

I created a file called fig_conf.mplstyle which contains common configurations for all the plots. Thus it's possible to homogenize somes aspects of figures at once (e.g color, size, typeface) to get a more aesthetic results in publications and also avoid configurations for each plot.

### Signal-to-noise map correction

The previous version of SN map contains an error. The calculations were wrong. The corrected calculations of SNR and the figure preparation will be provided in a pdf document. Although the correction didn't affect the fact that the image of the central region of the galaxy has an SNR greater then 10.

In the previous calculation I divided the integral of the signal by the integral of the noise, but now I realized that a square root of the number of pixels was forgotten.

## 2021-06-21

### Signal-to-noise description

I did a tex draft with the procedure description to compute the SNR. It also contains the example of application for the galaxy ngc 613. Also I made an simple makefile for compiling the tex. I will improve the code of makefile to check dependencis later.

## 2021-06-22

### Final version of script to signal-to-noise
I finished a final version of the script to plot the SNR map. Also I'm implementing a package structure to the scripts this is going to be useful in the future.

### Git and Dropbox
Finally I'm going to add the current work to git and dropbox. The data is restricted and cannot be added to git also it's too big for dropbox. Then it can be shared with Lucimara only by some other means.

## 2021-06-23

### Resampling correction
A correction of the flux conservative resampling procedure was made. It seems that there some problem with the resampling routine within ppxf util where the some calculations show that the flux is not been conserved even with the option for flux conservation being enabled. The results we accomplished are consistent with those obtained with astropy routine. The resampling, rebinning and convolution are going to be very import when applying ppxf for kinematics determination. I'm going to check these procedures very carefully to avoid errors and get reliable results. Following I'm going to describe the implementation of resampling in a pdf.

## 2021-06-24

### Plot
A plot comparing the results of resampling with different methods was made. The performance regarding runtime was measured.

## 2021-06-26

### Resampling
I described the resampling results made with different applications. It was shown that our implementation is the fastest one that still conserves the flux.

## 2021-06-28

### MUSE LSF
The resolving power (R) as function of wavelength of MUSE is available in the [MUSE user manual](https://www.eso.org/sci/facilities/paranal/instruments/muse/doc/ESO-261650_MUSE_User_Manual.pdf). We did the calculation to get the LSF in angstrom and the dispersion in km/s.

### Convolution
The convolution made with ppxf util is very general. It allows us to use a single average LSF, a piecewise convolution as done in ELODIE or even a different LSF for each pixel. We compared these different LSF injections, the results are similar to each other.

## 2021-06-29

### Rebinning
Some tests with rebinning were done. The results depend more on the number of pixels than on the implementation. Here, our goal is just to reproduce the signal without modifying its shape, just changing the points on the x-axis. At this point we have all the tools to treat the spectra to determine the kinematics.

## 2021-07-08

### Resampling and uncertainties
The resampling function works identically to the one that is part of astropy (but ~200x faster). But I'm facing some difficulties with the uncertainties management. I have discarded the employ of the [uncertainties package](https://pythonhosted.org/uncertainties/). There're some discussions about uncertainties handling [here1](https://github.com/astropy/specutils/issues/255) and [here2](https://github.com/astropy/specutils/pull/461).

## 2021-07-09

### Handling uncertainties on resampling
The uncertainties are currently being resampled, at least for now. I'm not sure if this is a correct procedure.

## 2021-08-26

### SpectCube
Now the function for resampling is within an independent package (SpectCube). It will be available on PyPI soon allowing instalation via pip.

## 2021-08-27

### LSF
Introducing function to use LSF as described by equation 8 in Bacon et al. 2017.

## 2021-08-28

### ppxf for the whole MUSE cube
Starting the implementation of scripts to run ppxf with the MUSE cube.

## 2021-08-31

### Preparing model and observations to use on ppxf
Some adaptations were made in the functions to prepare the data in its final form. That is, use all muse data and all the ssp models.

## 2021-09-01

### Muse cube on ppxf
Apparently some bug remains in the code to prepare the muse cube to use on ppxf. The models are prepared. It seems I need to investigate some aspects of the convolution. Currently I using th funcion provided by ppxf to convolve the data.

## 2021-10-12

## Missing annotations

I have missed the annotations of a few days. In summary, I have completed an execution of ppxf with the whole MUSE cube. The output data was saved which allowed to subtract the stellar component. For the nebular emission lines I first attempt of fitting will be made with IFSCube.

## Starting test IFSCube

I have read almost the all IFScube instruction. It seems to be an great tool to accomplish the task of fitting emission line kinematics. I will start testing with a single spectrum. Next, I will test the routine to fit data in cube using a central region of the field of view.

##
