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
I Finished a final version of the script to plot the SNR map. Also I'm implementing a package structure to the scripts this is going to be useful in the future.

### Git and Dropbox
Finally I'm going to add the current work to git and dropbox. The data is restricted and cannot be added to git also it's too big for dropbox. Then it can be shared with Lucimara only by some other means.
