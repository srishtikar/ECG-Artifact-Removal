# ECG-Artifact-Removal
ECG Artifact removal using the adaptive template method.

DESCRIPTION

This repository contains code for ECG artifact removal from EMG data, using a moving average method for R Peak detection.
Thereafter, ECG segments are taken to create an average template for that particular channel of EMG data
As part of the data cleaning process, this average template is subtracted directly from the EMG data at instances of R Peak detection.

INSTALLATION

Once you download these files onto MATLAB, load your EMG data into the 1st line of code, and change the filename variable
to reflect the name of the EMG data. Please note, if the trial contains a load, the word 'Load' must categorically be
mentioned in the file name. On failure to do so, the algorithm will run taking a normal, unloaded trial into consideration.

The file must contain SamplingFrequency and Time. The Data variable contains 38 columns of which 32 are channels and
the rest 6 indicate the start and end of trials, and remain untouched. Modify the code as such to accomodate for
variables of different sizes.

The DSP toolbox must be downloaded for the code to use the iirnotch function. In versions of MATLAB 2025 and newer, 
some variable types must be changed from single to double. Please make the necessary changes in the notch filtering
segment.

This code requires 2 user inputs. Firsly, a histogram with the amplitudes of the derivatives of the signal at each point
will pop up. The user must input, in positive whole numbers only, the minimum outlier value. For example, in a histogram of 
5 bins, if the x-axis digit closest to the third quartile of the first bin is 3000, input 3000. In general, choose a decently 
low enough number so as to account for all the ECG peaks.

Following this, a blue signal plot with pink dots will show up. The pink dots signify an ECG peak. The user must input 
the serial number of the first correctly detected ECG peak. Say the first pink dot occurs at a falsely detected peak, 
while the next one occurs at a true positive. Then, input 2 into the command window.

After this, the R Peak Detection is automatic.


POSSIBLE CHANGES

Contained in this repository are two average templates- one for loaded data sets and the other for unloaded data sets.
In case of a different electrode placement, the shape of the ECG artifact morphs. Thus, the average template holds true 
only for this case. In case the ECG artifact seems to have a different form, then load a suitable average template into 
the code to replace the ones I have given.

My code also contains plots of SNR before and after ECG artifact removal, as well as comparisons of power spectrums.
These are all kept commented. In case they are required, uncomment them. 
