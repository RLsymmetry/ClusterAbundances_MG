"""
This module is the ultimate Fisher analysis module for different modified gravity models constrained by σ8.
It mainly calls the module AgrowthfR_re.py to produce data from the modified gravity models, makes plots against the forcasted errors in σ8, and makes various Fisher analyses.
It inherits most of the functions in the Jupyter notebook Fisher_fR, but is intended to be an importable module. The interactive parts and results will then be run in any separate notebooks importing this module.

Rayne Liu 
06/12/2019
"""

#Import relevant modules
import numpy as np
import matplotlib.pyplot as plt
import pickle as pkl

import camb
from camb import model, initialpower

from matplotlib import rc
plt.rcParams['font.family'] = 'DejaVu Sans'
rc('text', usetex=True)
plt.rcParams.update({'font.size': 21})
plt.rcParams['text.latex.preamble'] = [r'\usepackage{amsmath}'] #for \text command

import AgrowthfR_re


#Introduce errors on sigma8/sigma8_ΛCDM by tSZ galaxy counts forcasted constraint
def serr(z):
    """
    z: an array of the redshifts corresponding to the sigma_8 constraint you wanted. 
    The complete range is Z = np.linspace(0.05, 2.95, 30). The array you enter must be the array that is contained in Z. 
    When this criteria is met, your sigma_8 array is chopped corresponding to your desired redshift ranges.
    """

    f1 = '/Users/RayneLiu/Desktop/Research/Cosmology/Matplotlib/savedS8Fisher_SO-v3-goal-40_grid-default_CMB_all_v1.0_planck_lcdm.pkl' ########Download the f1 data and change to desired directory#######
    paramList,FisherTot = pkl.load(open(f1,'rb'),encoding='latin1') 
    Z = np.arange(0.05, 2.95, 0.10)
    s8errors_tot = np.sqrt(np.diagonal(np.linalg.inv(FisherTot))[14:])
    s8errors_use = []
    for i in range(30):
        for j in range(len(z)):
            if np.allclose(Z[i], z[j], rtol = 0, atol = 1e-15):
                s8errors_use.append(s8errors_tot[i])
    if len(s8errors_use) < len(z):
        raise ValueError('Sorry, we don\'t have sigma_8 data for some of your redshifts')
    return z, np.array(s8errors_use)

###I WOULD LIKE THE DATA READING PROCESS, I.E. THE pkread() functions, modelvalues, etc. to be ALL SHOVELED INTO THE DIFFERENT MODIFIED GRAVITY MODELS. 

###I would like EVERY SUBCLASS OF THEM have the feature of just GIVE ME AN ARRAY OF s8/s8_ΛCDM (as one method inside of the subclass--the class should be able to do more other things for sure), and I would just CALL THE ARRAY DIRECTLY to do the Fisher analysis (also, I would like the array to have ALTERABLE REDSHIFT BINS). 

#Colors used (aligning with the tSZ paper)
colorbar = [(0.18039216, 0.65098039, 0.6627451), (0.65490196, 0, 0), (0.9372549 , 0.70588235, 0.23921569), (0.89803922, 0.84705882, 0.64313725), (0.43529412, 0.36470588, 0.27843137), 'blue']

#Plotting s8/s8_ΛCDM against the given errors
def ratio_s8_mg_err(z, arrays, labels, title = True, save = False):
    """
    Plots sigma_8/sigma_8_ΛCDM, with the given error bars; 
    together with the predicted sigma_8_MG/sigma_8_ΛCDM. 
    Normalized at z = 0 (which is already done in AgrowthfR_re.py).
    
    Parameters:
    
    arrays: a numpy array that contains the various s8/s8_ΛCDM arrays [they must corresponding to the same redshift bins, the 
            default redshift bins being z = np.linspace(0.05, 2.95, 30)].
            Can also accept a single ratio array.
    z:      the redshift corresponding to the arrays.      
    labels: a list of labels corresponding one-to-one to the arrays to put on their respective plots. [Have the same length as   
            arrays, with the same order, format being rc string.]
            
    title: whether to have title.
    
    save:   whether to save the plot as pdf [bool]
    """
    fig = plt.figure(figsize=(9, 9))
    print('The input redshift is:')
    print(z)
    Z = np.linspace(0.05, 2.95, 30)
    #I only use this Z to plot all the error bars. But the individual redshifts of the calculated s8 array can be different!
    plt.errorbar(Z, [1.0 for z in Z], serr(Z)[1], elinewidth = 11.6, alpha = 0.37, color = '#99876e', label = r'$\Lambda\mbox{CDM}$ with forecasted errors')
    plt.axhline(y = 1, ls = '--', linewidth = 1.2, color = (0.54509804, 0.51764706, 0.50196078))
    
    
    if type(arrays[0]) in (int, float, np.int64, np.float64):
        plt.plot(z, arrays, linewidth = 1.2, color = colorbar[0], label = labels)
        plt.fill_between(z, arrays, y2 = 1., color = colorbar[0], alpha = 0.32)
    else:                       
        for i in range(len(arrays)):
            plt.plot(z, arrays[i], linewidth = 1.2, color = colorbar[i], label = labels[i])
            plt.fill_between(z, arrays[i], y2 = 1., color = colorbar[i], alpha = 0.32)
        
    plt.xlabel(r'$z$', fontsize = 21)
    plt.xticks(fontsize=21)
    plt.xticks(fontsize=21)
    plt.title(r'$\sigma_8(z)_{\text{MG}}/\sigma_8(z)_{\Lambda\text{CDM}}$', fontsize = 21)

    #if title == True:
    #    plt.title(r'$\sigma_8(z)_{\mbox{MG}}/\sigma_8(z)_{\Lambda\mbox{CDM}}$, with $\sigma_8(z)_{\Lambda\mbox{CDM}}/\sigma_8(z)_{\Lambda\mbox{CDM}}$ and tSZ forecasted constraints, normalized at $z = 10$')

    plt.legend(loc = 'lower left', fontsize = 21)
    
    if save == True:
        plt.savefig('mgs.pdf', format='pdf', bbox_inches='tight', dpi=1200)
    plt.show()
    

    
###Main Fisher matrix function and confidence ellipses###

def fishermain(derivs, errs):
    """
    Returns: a general n*n Fisher Matrix, given the parameters derivs and errs.

    Parameters:
    
    derivs:  a m*n array of derivatives (general or numerical)--m being the number of observables, with 
             b being the subscript (i.e. one row corresponds to one b only, with m rows in total); 
             n being the number of parameters, with i & j being the subscripts (i.e. one row walks
             through 1-n).
    
    errs:    a 1*m array of measurement uncertainties. Does not have to be serr().
    """
    #Construct a frame for the Fisher Matrix
    m = len(derivs)
    n = len(derivs[0])
    F = np.zeros((n, n))
    #Slightly process the error list
    invsq = np.divide(1, errs ** 2)
    #print(invsq)
    
    #Construct the matrix body term by term
    for i in range(n):
        for j in range(n):
            #Tried to use einsum directly from derivs but failed to find good ones. Give up...and construct  
            #a new array first.
            newlist = np.empty(m)
            for b in range(m):
                newlist[b] = derivs[b][i] * derivs[b][j]
            #print(newlist)
            F[i][j] = np.einsum('b, b', newlist, invsq)
            
    return F

#The α value used in plotting ellipses
alphas = [1.52, 2.48]


#The Dan Coe tutorial method of plotting the confidence ellipse
def coeellipse(Cov, pars, labelpars, blockvalues, linewidth = 1.6):
    """
    The Dan Coe method of drawing the confidence ellipse from a given covariance matrix.
    Code reference from the Princeton summer school tutorial.
    Did not multiply by the alpha parameter. Can alter the code if it is needed.
    The plot is currently still labeled by f_R0 because we are currently doing it. Can change and generalize later.
    
    Cov:        the corresponding covariance matrix.
    pars:       the numpy array of parameters. Currently this ellipse function only accepts two parameters.
    labelpars:  the list of labels to put on the axes denoting the parameters respectively.
    blockvalues: the error of pars[0] after blocking pars[1], and likewise for pars[1]. Length-2 numpy array.
    """
    from matplotlib.patches import Ellipse

    eigval,eigvec = np.linalg.eigh(Cov)

    theta = np.degrees(np.arctan2(eigvec[1,0], eigvec[0,0]))
    a,b = np.sqrt(eigval)

    fig,ax = plt.subplots(1, 1, figsize = (11, 11))
    #ax.axis('equal')
    ax.plot(pars[0], pars[1], marker='X', linewidth = linewidth * 3, zorder=10, linestyle='none', color='k', label='Canonical value')
    
    #One-sigma & two-sigma confidence level ellipse
    for i in range(len(alphas)):
        ax.add_patch(Ellipse((pars[0], pars[1]), width = a * 2 * alphas[i], height = b * 2 * alphas[i], angle = theta, 
                                 fill = False, linewidth = 2.5 * linewidth/alphas[i], color = colorbar[i], label = str(i + 1) + r'$\sigma$ Confidence Ellipse'))


        plt.axvline(x = pars[0] + np.sqrt(Cov[0][0]) * alphas[i], ls = '--', color = colorbar[i + 2], linewidth = linewidth, label = str(i + 1) + r'$\sigma$ $x + \alpha \sigma_x$ (' + labelpars[0] +')')
        plt.axvline(x = pars[0] - np.sqrt(Cov[0][0]) * alphas[i], ls = '--', color = colorbar[i + 2], linewidth = linewidth, label = str(i + 1) + r'$\sigma$ $x - \alpha \sigma_x$')
        plt.axhline(y = pars[1] + np.sqrt(Cov[1][1]) * alphas[i], ls = '--', color = colorbar[i + 3], linewidth = linewidth, label = str(i + 1) + r'$\sigma$ $y + \alpha \sigma_y$ (' + labelpars[1] + ')')
        plt.axhline(y = pars[1] - np.sqrt(Cov[1][1]) * alphas[i], ls = '--', color = colorbar[i + 3], linewidth = linewidth, label = str(i + 1) + r'$\sigma$ $y - \alpha \sigma_y$')


    ax.set_xlabel(labelpars[0])
    ax.set_ylabel(labelpars[1])
    ax.legend(loc='upper left')
    plt.savefig('coemethodf_R0=1e' + str(pars[0]) + 'n=' + str(pars[1]) + '.pdf', format='pdf', bbox_inches='tight', dpi=1200)
    fig.tight_layout()

#The Cholesky decomposition of plotting the confidence ellipse
def choleskyellipse(Cov, pars, labelpars, blockvalues, linewidth = 1.6):
    """
    Draws the confidence ellipse using Cholesky decomposition of the covariance matrix.
    
    Cov:       the corresponding covariance matrix.
    pars:      the numpy array of parameters. Currently this ellipse function only accepts two parameters.
    labelpars: the list of labels to put on the axes denoting the parameters respectively.
    blockvalues: the error of pars[0] after blocking pars[1], and likewise for pars[1]. Length-2 numpy array.
    """
    #Taking the Cholesky decomposition, where L is a lower triangular
    L_fR = np.linalg.cholesky(Cov)
    #print(L_fR)
    #print(np.dot(L_fR, L_fR.T.conj()))

    #Creating a unit circle of vectors
    t = np.linspace(0, 2 * np.pi, 100000)
    fcirc = np.array([np.cos(t), np.sin(t)])
    #print(fcirc)

    #Making the linear transformation
    meanblock = np.array([np.repeat(pars[0], 100000), np.repeat(pars[1], 100000)])
    #print(trans)

    #Plotting the ellipse
    fig,ax = plt.subplots(1, 1, figsize = (11, 11))
    #ax.axis('equal')
    ax.plot(pars[0], pars[1], marker='X', linewidth = linewidth * 3, zorder=10, linestyle='none', color='k', label='Canonical value')
    for i in range(len(alphas)):
        trans = np.dot(L_fR, fcirc) * alphas[i] + meanblock
        ax.plot(trans[0], trans[1], color = colorbar[i], linewidth = 2.5 * linewidth/alphas[i], label = str(i + 1) + r'$\sigma$ confidence ellipse')

        plt.axvline(x = pars[0] + np.sqrt(Cov[0][0]) * alphas[i], ls = '--', color = colorbar[i + 2], linewidth = linewidth, label = str(i + 1) + r'$\sigma$ $x + \alpha \sigma_x$ (' + labelpars[0] +')')
        plt.axvline(x = pars[0] - np.sqrt(Cov[0][0]) * alphas[i], ls = '--', color = colorbar[i + 2], linewidth = linewidth, label = str(i + 1) + r'$\sigma$ $x - \alpha \sigma_x$')
        plt.axhline(y = pars[1] + np.sqrt(Cov[1][1]) * alphas[i], ls = '--', color = colorbar[i + 3], linewidth = linewidth, label = str(i + 1) + r'$\sigma$ $y + \alpha \sigma_y$ (' + labelpars[1] + ')')
        plt.axhline(y = pars[1] - np.sqrt(Cov[1][1]) * alphas[i], ls = '--', color = colorbar[i + 3], linewidth = linewidth, label = str(i + 1) + r'$\sigma$ $y - \alpha \sigma_y$')
    #plt.xscale('log')
    plt.xlabel(labelpars[0])
    plt.ylabel(labelpars[1])
    plt.legend(loc='upper left')
    plt.savefig('choleskymethodf_R0=1e' + str(pars[0]) + 'n=' + str(pars[1]) + '.pdf', format='pdf', bbox_inches='tight', dpi=1200)
    fig.tight_layout()


    
###Fisher analysis section###
#The convergence tests are put inside of the classes, because each one is different and not very convenient to merge.

#Small sections of different scenarios in the Fisher analysis
def fisher1d(var, derivs):
    """
    Does the Fisher analysis for the 1-D array of imput derivatives. Inherited parameters.
    """
    derivs1d = np.array([derivs]).T
    Fish1d = fishermain(derivs1d, serr(var[0])[1])
    Cov1d = np.linalg.inv(Fish1d)
    scalarinv = np.divide(1, Fish1d[0][0])

    print('The chosen derivative array is:')
    print(derivs1d)
    print('The resulting Fisher matrix(scalar) is:')
    print(Fish1d)
    print('The covariance matrix(scalar) is:')
    print(Cov1d)
    print('Checking the inverse of the Fisher scalar value:')
    print(scalarinv)
    print('The error is thus:')
    print(np.sqrt(scalarinv))
    return Fish1d, np.sqrt(scalarinv)

    
def fisher2d(var, pars, derivs, printables, labels):
    """
    Does the Fisher analysis for the 2-D array of imput derivatives. Inherited parameters.
    """
    print('We are now looking at the redshifts:')
    print(var[0])
    derivs2d = derivs.T
    print('The chosen derivative matrix is:')
    print(derivs2d)
    Fish2d = fishermain(derivs2d, serr(var[0])[1])
    print('The resulting Fisher matrix is:')
    print(Fish2d)
    Cov2d = np.linalg.pinv(Fish2d)
    print('The covariance matrix is:')
    print(Cov2d)
    cov2ds = [np.sqrt(Cov2d[0][0]), np.sqrt(Cov2d[1][1])]
    print('The marginalized error on ' + printables[0] + ' is:')
    print(cov2ds[0])
    print('The marginalized error on ' + printables[1] + ' is:')
    print(cov2ds[1])
    
    c_11 = np.sqrt(np.divide(1, Fish2d[0][0]))
    print('The error on ' + printables[0] + ' after blocking ' + printables[1] + ' is:')
    print(c_11)
    c_22 = np.sqrt(np.divide(1, Fish2d[1][1]))
    print('The error on ' + printables[1] + ' after blocking ' + printables[0] + ' is:')
    print(c_22)
    print('The confidence ellipse, Dan Coe tutorial version (blue) vs. Cholesky version (grey):')
    coeellipse(Cov2d, pars, labels, np.array([c_11, c_22]))
    choleskyellipse(Cov2d, pars, labels, np.array([c_11, c_22]))
    
    #return the marginalized error and blocked error for use
    return Fish2d, cov2ds, np.array([c_11, c_22])
    
    
def singularfisher(var, derivs, printables):
    """
    Does the Fisher analysis for the scenario where one of the parameters in the length-2 pars results in 
    all-zero partial derivatives. Inherited parameters, see the function fisheranalysis.
    """
    if abs(np.max(derivs[0])) < 1e-15:
        print('Partial derivatives over ' + printables[0] + ' is all-zero, now doing a singular Fisher analysis about ' + printables[1])
        print(derivs[0])
        print('The other derivative array:')
        print(derivs[1])
        return fisher1d(var, derivs[1])

    if abs(np.max(derivs[1])) < 1e-15:
        print('Partial derivatives over ' + printables[1] + ' is all-zero, now doing a singular Fisher analysis about ' + printables[0])
        print(derivs[1])
        print('The other derivative array:')
        print(derivs[0])
        return fisher1d(var, derivs[0])
        
        
#Overall Fisher analysis of a model with a certain set of given parameters; has to take into account of the parameter dimensions.
def fisheranalysis(model, var, pars, derivs):
    """
    ###### It looks like I put in changeable z arrays in TOO EARLY. The serr() doesn't have changeable z arrays...just use this very carefully right now and we may need and be able to set this later. ######
    ##New edit on Oct.7 2020: now the changeable z arrays come into use!! We want to be able to chop the serr() and give the fisher matrix constraint on any chunk of z between what's given.
    
    Returns: the Fisher matrix, the covariance matrix, the Fisher blocking result, and the confidence ellipse with
    both the Dan Coe method and the Cholesky method.
    
    Parameters:
    var:        the numpy array consisting of the redshift z and scale wave number k used.
    pars:       the numpy array of parameters. Also can be a single parameter.
    derivs:     the numpy array consisting of the corresponding arrays of chosen derivatives wrt the parameters, respectively.
                It should be noted that even with 1 parameter, derivs should be an array of the array of derivatives, not a
                single array of derivatives. This is taken care of in the final complete analysis function.
    labels:     the list of labels corresponding to the pars (must be in the same order).
    printables: the label to put in print statements. Lists of strings like 'f_R0', 'n', etc.
    """
    if isinstance(model, AgrowthfR_re.HuSawicki):
        if abs(np.max(derivs[0])) < 1e-15 or abs(np.max(derivs[1])) < 1e-15:
            return singularfisher(var, derivs, model.get_printables())
        else:
            #Check whether that gives me the same constraints as what I get before, and restore the original fiducial
            #fr0 = 10.0**(pars[0])
            return fisher2d(var, pars, derivs, model.get_printables(), model.get_axeslabels())
    elif isinstance(model, AgrowthfR_re.DGP):   
        return fisher1d(var, derivs)
    else:
        print('Sorry, this function cannot accept your current dimension yet. We\'re working!')

    
###Assembling the analysis together into one call###

def modelanalysis(model, var, labels, pars = None, save = False):
    """
    The ultimate σ8 analysis of a given MG model. Enter the model, the relevant parameters (with the order aligned!)
    
    var: a numpy array consisting of the arrays of redshift(first), wavenumber k.
    labels are the ones that you put inside of the error bar plotting and is never None!
    """
    #Sets the parameters in the corresponding model. 
    #It's hard to do it here (the parameter names are specific for each model!) but we can pass the pars down to the model
    #class and do it there!
    if isinstance(model, AgrowthfR_re.LCDM):
        print('No free parameters to set in LCDM model')
        svalues = AgrowthfR_re.Cosmosground().sigma8_ratio(model, var)
    else:
        model.setpars(pars)
    
    #Gets the sigma_8 ratio arrays and plots the result
    if isinstance(model, AgrowthfR_re.HuSawicki):
        svalues = AgrowthfR_re.Cosmosground().sigma8_ratio(model, var[0], k = var[1])
    elif isinstance(model, AgrowthfR_re.DGP):
        svalues = AgrowthfR_re.Cosmosground().sigma8_ratio(model, var[0])
    
    print('The sigma8/sigma8_LCDM ratio plot of the model is:')
    ratio_s8_mg_err(var[0], svalues, labels, save)
    
    #print(svalues)
    
    #Does the derivative convergence test
    if isinstance(model, AgrowthfR_re.LCDM):
        print('Currently no Fisher analysis to be done for this model')
    else:
        if isinstance(model, AgrowthfR_re.HuSawicki):
            derivs = model.testderiv(pars[0], pars[1], var[0], var[1], save)
        elif isinstance(model, AgrowthfR_re.DGP):
            derivs = model.testderiv(pars, var[0], save)          
        #Does the proceeding Fisher analysis
        return fisheranalysis(model, var, pars, derivs)
        
#For nDGP currently (and perhaps other 1-parameter models) doing a SNR plot for an array of parameters
def plot1param(model, pararray, z, save = False):
    """
    Does a simple Fisher analysis for 1-parameter models like the nDGP model and plots the error-parameter relation.
    pararray: a numpy array of the different values of the single parameter of the 1-parameter model.
    """
    errors = np.zeros(pararray.shape)
    for i in range(len(pararray)):
        derivs = model.testderiv(pararray[i], z, show = False)
        errors[i] = fisher1d(derivs, ret = True)
    
    fig = plt.figure(figsize=(12, 8.8))
    plt.plot(pararray, np.divide(pararray, errors), linewidth = 1.7)
    plt.xlabel('Parameter fiducial values chosen')
    plt.ylabel('Fiducial values over Errors')
    plt.title(r'Signal-to-noise vs. Fiducial Value, $z$ from ' + str(z[0]) + ' to ' + str(z[-1]) + ' with ' + str(len(z)) + ' datapoints')
    if save == True:
        plt.savefig('nDGP_SNR.pdf', format='pdf', bbox_inches='tight', dpi=1200)
    plt.show()