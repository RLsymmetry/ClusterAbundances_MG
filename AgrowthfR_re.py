"""
This importable module translates the 1st-order solution of the mass growth factor to the matter growth differential equation in the different modified gravity models, and also does the σ_8 calculation. The detailed explanations and derivations are placed inside of an ipython notebook with the same name.

It is possible and encouraged to extend this module for higher order solutions. In order to be better utilized and revised in later work, I wrote a class with basic cosmology and its subclasses each corresponding to a modified gravity model. 

Initially there are three subclasses, representing the ΛCDM model, the Hu-Sawicki fR model, and the nDGP model respectively. Adding more future theoretical model subclasses is welcome and anticipated.

Rayne Liu 06/12/2019
"""

#Import necessary modules
import numpy as np
from scipy.misc import derivative
from scipy import integrate
from scipy import special
import matplotlib.pyplot as plt
import camb

from matplotlib import rc
plt.rcParams['font.family'] = 'DejaVu Sans'
rc('text', usetex=True)

#Change to the desired directory, if needed to export .txt files
#import os
#os.chdir('/Users/liuchangchun/Desktop/Research/Cosmology/Growth \
#Calculation/Altered D(a) Codes') 



class Cosmosground(object):
    """
    Background cosmology that should be universal in all models, regardless of the theoretical model we're examining.
    """
    
    def get_amin(self):
        """
        Returns: the minimum scale factor when the diviation between the ΛCDM model and the fR model began to take place, 
        i.e. when we start to solve for the modified growth differential equation.  

        Before this time, the growth factor is given by the integral term--reference: Dodelson cosmology pp.207 Eqn.(7.77).
        """
        return self._amin
    
    def set_amin(self, value):
        """
        Sets the amin scale factor to value (although it looks like we don't generally change it).
        
        value: [0 <= value <= 1]
        """
        assert 0 <= value <= 1, 'Invalid scale factor value'
        self._amin = value
        
    def get_amax(self):
        """
        Returns: the max scale factor we are looking at, in general the scale factor today. 
        """
        return self._amax
    
    def set_amax(self, value):
        """
        Sets the amax scale factor to value, in general the scale factor today.
        
        value: [0 <= value <= 1]
        """
        assert 0 <= value <= 1, 'Invalid scale factor value'
        self._amax = value
        
    def get_c(self):
        """
        Returns: the speed of light we're using in this cosmology. 
        """
        return self._c
    
    def set_c(self, value):
        """
        Sets the speed of light to value, depending on the normalization we are using (in general not suggested to change).
        """
        self._c = value
        
    def get_H0(self):
        """
        Returns: the Hubble constant (at a = 1). 
        """
        return self._H0
    
    
    def set_H0(self, value):
        """
        Sets the Hubble constant (at a = 1). 
        This also simultaneously changes self._Om. For this reason, there is no separate set_Om method.
        """
        self._H0 = value
        self._Om = (self._omch2 + self._ombh2)/((self._H0/100) ** 2)
    
    
    def get_omch2(self):
        """
        Returns: the energy density to critical of the cold dark matter, scaled by /h^2 (h = H0/100). 
        """
        return self._omch2
    

    def set_omch2(self, value):
        """
        Sets the energy density to critical of the cold dark matter, scaled by /h^2 (h = H0/100). 
        This also simultaneously changes self._Om. For this reason, there is no separate set_Om method.
        """
        self._omch2 = value
        self._Om = (self._omch2 + self._ombh2)/((self._H0/100) ** 2)

        
    def get_ombh2(self):
        """
        Returns: the energy density to critical of the baryons, scaled by /h^2 (h = H0/100). 
        """
        return self._ombh2
    
    
    def set_ombh2(self, value):
        """
        Sets the energy density to critical of the cold dark matter, scaled by /h^2 (h = H0/100). 
        This also simultaneously changes self._Om. For this reason, there is no separate set_Om method.
        """
        self._ombh2 = value
        self._Om = (self._omch2 + self._ombh2)/((self._H0/100) ** 2)
    
    
    def get_Om(self):
        """
        Returns: the current value of Om, i.e. the normalized non-relativistic matter density parameter. 
        """
        return self._Om
    
    def get_ns(self):
        """
        Returns: the current value of ns, i.e. the scalar spectral index n_s of the CMB. 
        """
        return self._ns
    
    def set_ns(self, value):
        """
        Sets ns to value. 
        """
        assert type(value) == float or int
        self._ns = value
    
    
    def get_As(self):
        """
        Returns: the current value of As, i.e. the comoving curvature power A_s of the CMB. 
        """
        return self._As
    
    def set_As(self, value):
        """
        Sets As to value. 
        """
        assert type(value) == float or int
        self._As = value
   
    
    '''
    def get_P0(self):
        """
        Returns: the matter power spectrum at redshift z = 0. Using the data directly from CAMB, taking 500 kh values.
        """
        return self._P0
    '''
    
    def __init__(self):
        """
        Initializes the basic constant cosmological quantities.
        """
        #(Change to desired background cosmology if needed. )
        self._amin = 0.002  
        self._amax = 1  
        self._c = 2997
        self._H0 = 67.0 #H0 = 67.0, i.e. h = 0.67. However, this module itself works on normalized H0 = 1, and 67.0 is mainly for CAMB.)
        self._omch2 = 0.1194
        self._ombh2 = 0.022
        self._Om = (self._omch2 + self._ombh2)/((self._H0/100) ** 2) #+ 0.06/(94 * (0.67 ** 2)) 
        #Om denotes the normalized density factor of all non-relativistic matter.)
        #(For Wmap9: Om=0.281)     
        #(For Planck: Om=0.3089)
        #####Mind when you set Om, you have to set the corresponding density factor in camb as well!)
        #Because of this, set_Om is disabled; you can only set_omch2 or set_ombh2.
        self._ns = 0.96
        self._As = 2.2e-9
        #From Nick's tSZ galaxy count paper, ombh2 = 0.022, omch2 = 0.1194
        
        #self._Onu = 
        #(Gave up on this nu--this doesn't change the final results by any amount and it caused a bag of troubles in the calculation!)
        #self._P0 = self._p(np.array([0.0]))[2][0]
        #Matter power spectrum when z = 0

     

    #Pertinent quantities to appear in the DiffEqn.s
    def _adot(self, a):
        """
        The time derivative of the growth factor. 
        From the ΛCDM intro paper: H^2/H0^2 = O_m/a^3+O_Λ---
        (From Georgios: we are now well in the matter-dominated era, so O_r is ignored; 
        also, according to Georgios, his code is working on everything normalized by H0^2, i.e. let H0^2 = 1. 
        If want to use it in a different context later, maybe can add a way to set it?)
        """
        return np.sqrt(self._Om/a + (1 - self._Om) * (a ** 2))


    def _derivadot(self, a):
        """
        The derivative of adot with respect to a. 
        This is a term that appears in the solution process of the 2nd-order differential equation of the growth function
        shown below. Since mathematica can do abstract derivative but python cannot, but luckily this derivative can be done 
        analytically, so I here just put in the analytical expression by hand.
        """
        #verified correct by putting 5 different a's into mathematica and comparing.
        numerator = - (self._Om) + 2 * (1 - self._Om) * (a ** 3) 
        denominator = 2 * np.sqrt((a ** 3) * (self._Om) + (1 - self._Om) * (a ** 6))
        return numerator/denominator
    

    def _itgrand(self, a):
        """
        Returns: the integrand in the intInitial.
        """
        return 1/(self._adot(a) ** 3)

    def _intInitial(self):
        """
        Returns: the helper integral to give the inital conditions of the growth factor DiffEq. 
        Reference: Dodelson pp. 207 equation (7.77).

        Parameters:
        adot & a are both real.
        """
        #assert type(adot) in (int, float, np.int64, np.float64), 'adot is not a real number'
        #assert type(a) in (int, float, np.int64, np.float64), 'a is not a real number'
        #(This doesn't seem to work but maybe I don't need to assert real values after all)

        intInitial = integrate.quad(self._itgrand, 0, self._amin)
        return intInitial[0]

    #Hubble stuff
    def H(self, a):
        """
        Returns: the Hubble parameter.
        """
        return self._adot(a)/a

    def _derivHa(self, a0):
        """
        Returns: the derivative of the Hubble parameter H over the scale factor a, at value a0. 
        Doesn't seem to have explicit physical meanings by now, but appears in the process of solving the DiffEq.
        """
        return derivative(self.H, a0, dx = 1e-8)
    
    #Initial conditions of the growth factor
    def _D1(self):
        """
        Returns: the initial value of the growth factor. This is the same for all gravity models and has no k-dependence.
        """
        return 2.5 * self._Om * self.H(self._amin) * self._intInitial()
    
    def _D1deriv(self):
        """
        Returns: the initial value of the derivative of the growth factor over the scale factor a.
        """
        return 2.5 * self._Om * (self._derivHa(self._amin) * self._intInitial() + \
                                 self.H(self._amin)/(self._adot(self._amin)) ** 3)
    
    #The σ8 section
    #I directly copied sigma8_LCDM here (it was another module to import originally), 
    #since after the shrinking down of things it reduces to only two functions.
    #No need to import anymore.
    #Sets CAMB initial conditions and gets P(k) array directly from CAMB
    def _p(self, z, w0 = -1, wa = 0, npoints = 513):
        """
        Returns: Three arrays representing the mass power spectrum as a function of
        kh in different redshift bins, produced by the camb module, given the parameters w0 and wa.

        Reference: camb instruction example notebook codes adapted.
        Parameters: 
        z: the redshift. Must be an array.
        w0, wa: parameters in the equation of state in dark energy.
        model: if calculating the p(k) of a particular model, use the model's parameters. 
               Else use the background parameters.
        """
        #print('In p(k), the parameters are:')
        #print(self._H0)
        #print(self._omch2)
        #print(self._ombh2)
        parms = camb.CAMBparams()
        parms.set_cosmology(H0 = self._H0, omch2 = self._omch2, ombh2 = self._ombh2, tau = 0.06, mnu = 0.06) #mnu = 0.06
        #From Nick's paper  #omch2 – physical density in cold dark matter
        parms.InitPower.set_params(ns = self._ns, As = self._As) #Also from Nick's paper
        #(Didn't put this into the attributes since it is not used frequently -- actually just once here
        parms.set_matter_power(redshifts = z, kmax = 12.0)
        parms.set_dark_energy(w = w0, wa = wa, dark_energy_model = 'ppf') #This is the remnant of my Fisher training section, 
                                                                         #but might be useful later
        #Linear spectra - where the σ8 analysis lives
        parms.NonLinear = camb.model.NonLinear_none
        results = camb.get_results(parms)
        kh, z, P = results.get_matter_power_spectrum(minkh=1e-4, maxkh=10, npoints = npoints)
        s8_camb = results.get_sigma8()
        return kh, z, P, s8_camb


    #Actually calculates sigma8 with redshift bins, using Simpson's rule. 
    def _sigma8_dyn(self, z, w0 = -1, wa = 0, pkarray = None, npoints = 513):
        """
        Sigma_8 as a function of redshift z calculated from the built-in
        camb-calculated P(k). 

        It looks like Simpson's rule does not really require linear spacing!!! 
        Simpson is good enough at least better than trapz!!
        I was soooo dumb................

        Parameter: 
        z          [a numpy array of redshifts, for each entry 0 <= z[i] <= 1100]
        model      [if model is None, use the background parameters. If it is not, use the model parameters.]
        """
        #print('In sigma8_dyn, the Om used is:')
        #print(self._Om)
        if pkarray != None:
            kh, z, P = pkarray
        else:
            kh, z, P = self._p(z, w0, wa, npoints = npoints)[:-1] #Default value of power spectrum is obtained from CAMB
        itgd = P * (3 * special.spherical_jn(1, kh * 8)/(kh * 8)) ** 2 * (kh ** 2)
        integ = integrate.simps(itgd, x = kh)
        integ = np.sqrt(integ/(2 * (np.pi) ** 2))
        return z, integ    
    
    #Calls sigma_8 calculations built in CAMB (currently only available in ΛCDM model) with a given z to see whether I 
    #calculated sigma_8 right.
    def sigma8_check(self, z):
        """
        Calls _sigma8_dyn and then the CAMB built-in sigma8, and make comparisons. Have to make sure that the fundamental 
        cosmological parameters and P(k), etc. are the same thing.
        z: the redshift array that we look at.      
        """
        sigma8_own = self._sigma8_dyn(z)[1]
        sigma8_camb = np.flip(self._p(z)[3])
        check_ratio = sigma8_own/sigma8_camb
        print('The calculated sigma_8 is:')
        print(sigma8_own)
        print('The sigma_8 from CAMB is:')
        print(sigma8_camb)
        print('The ratio of the former agains the latter is:')
        print(check_ratio)
        
        

   
    #The function to produce the array of kh, z, P, for k-dependent growth factor only
    def _Pk(self, z, k, data):
        """
        Returns: the desired specific kh, z, P array extracted from data in models that feature k-dependent growth factors.

        Parameters: 
        data    [the data to read in, a numpy array]
        z & k must be complying with what were used to obtain the data
        model: the model to calculate the p(k). This matters because we might compare the p(k) calculated in some model, 
        say, Hu-Sawicki, with different cosmology parameters, with the p(k) in standard Cosmosground cosmology.
        """
        z = z
        kh = k
        D = data.T
        #print('D is')
        #print(D.shape)
        #print(D)
        P0 = self._p(np.array([0.0]), npoints = len(kh))[2][0]#P_{ΛCDM}(k,z=0), used data directly from CAMB,
                                                         #used default value for 500 k numberpoints
        #print('P0 is')
        #print(P0)
        D0 = LCDM().solvegrowth(np.linspace(0, 3, 30))[0]#D_{ΛCDM}(k,z=0), solved from the LCDM subclass; 
                                                         #used 30 redshift bins (actually can use less)
        #This is fine because D0 from the DiffEqn (0 redshift instead of high redshifts) does not deviate much
        #from that in CAMB.
        #print('D0 is')
        #print(D0)

        for i in range(0, len(D)):
            #P_{MG}(k,z) =  P_{ΛCDM}(k,z=0)*(D_{MG}(k,z)/D_{ΛCDM}(z=0))^2, here we directly alter on the basis of D.
            for j in range(0, len(D[i])):
                Pij = P0[j] * ((D[i][j]/D0) ** 2)
                D[i][j] = Pij

        return kh, z, D   #This actually returns the P(k) but since we altered D directly it just takes the name of D
    
    def sigma8_model_ratio(self, model, z, k = None):
        """
        Returns: the normalized ratio σ8_{model}/σ8_{ΛCDM} under different models. 
        Also returns the σ8_{model} itself for later use.
        
        Here, the σ8_{ΛCDM} comes from CAMB data calculated at the beginning, 
        instead of from the solution in the LCDM model class.
        Always normalized at z = 10. We assume that the range of redshift that we study will never reach 10,
        hence we first directly append the 10 to the tail of z without looking at different scenarios.
        
        model: the class corresponds to the model we're looking at.
        z: the numpy array of redshifts that we have chosen.
        """
        #print('In sigma8_model_ratio, the Om used is:')
        #print(self._Om)
        
        #Getting the sigma8_lcdm-z relation at redshift ten
        z_aug = np.append(z, np.array([10]))
        s8lcdm = self._sigma8_dyn(z_aug)[1]
        
        #Different algorithm with respect to different k dependence
        if model.kdependence == 0:   
            #Solve the growth factor as it is
            D = model.solvegrowth(z_aug)
            #D/D[0] is normalizing, since the growth factor in any model is 1 at redshift 0.
            D_norm = D/D[0]
            #However, here when we normalize the s8 ratio wrt z = 10, 
            #s8_MG(z = 10) = s8_LCDM(z = 10). Hence s8_MG(z = 0) = s8_MG(z = 10)/D_norm(z = 10).
            #We then multiply s8_MG(z = 0) by D_norm.
            s8d = (s8lcdm[-1]/D_norm[-1]) * D_norm
            #And then take the ratio over s8lcdm. This should already be 1 at redshift 10,
            #hence the s8 shoud already be a normalized one, which I can directly use.
            ratio = s8d/s8lcdm
            fratio = np.delete(ratio, -1)
            fs8 = np.delete(s8d, -1)
            return fratio, fs8
        elif model.kdependence == 1:
            pkarray = self._Pk(z_aug, k, model.solvegrowth(z_aug, k))
            #Calculate the s8_MG via the given power spectrum.
            s8model = self._sigma8_dyn(z_aug, pkarray = pkarray, npoints = len(k))[1]
            #I want the output to contain a normalized s8model
            norms8model = s8model * s8lcdm[-1]/s8model[-1]
            #Make the ratio. Now it should already be normalized.
            nratio = norms8model/s8lcdm
            #print(nratio)
            fratio = np.delete(nratio, -1)
            fs8 = np.delete(norms8model, -1)
            return fratio, fs8
        
    def sigma8_ratio(self, model, z, k = None):
        """
        Returns: the ratio from sigma8_model_ratio alone. (Remember to check whether it's normalized or not)
        (because I'm too lazy to change everything I've written before...sorry)
        """
        return self.sigma8_model_ratio(model, z, k)[0]
    
    def sigma8_values(self, model, z, k = None):
        """
        Returns: the value from sigma8_model_ratio alone. (Remember to check whether it's normalized or not)
        (I never expected that this "save for later use" will actually save my life someday)
        """
        #print('Now the ombh2, omch2, Om are:')
        #print(self._ombh2)
        #print(self._omch2)
        #print(self._Om)
        return self.sigma8_model_ratio(model, z, k)[1]
    
    def sigma8_lcdm(self, z, k = None):
        """
        Returns: the sigma_8 wrt z calculated from ΛCDM model from CAMB.
        """
        return self._sigma8_dyn(z)[1]
    
       
    #The Linder gamma check section
    def _Om_M(self, a):
        """
        The energy density relative to critical as a function of a.
        From Linder: Om_M(a) = (Om * a^{-3})/(H/H0)^2
        Parameter:
        a                 [the scale factor, 0 <= a <= 1]
        """
        #print('Here the Om parameter is:')
        #print(self._Om)
        #Here I try altering the Hubble parameter by hand for DGP.
        #numerator = self._Om/(a ** 3)
        #denominator = (np.sqrt(self._Om/(a ** 3) + 1/(4 * 5 ** 2)) - 1/(2 * 5)) ** 2
        #return numerator/denominator
        return self._Om * a/(self._Om * a + (1 - self._Om) * (a ** 4))

    def _itgD_from_g(self, a, gamma):
        """
        The integrand of D_from_g(a).
        gamma: growth function parameter
        """
        return (((self._Om_M(a)) ** gamma) - 1)/a

    def _D_from_g(self, a, gamma):
        """
        The growth function D(a) obtained from the fitting function g(a).
        gamma: growth function paramter
        """
        intgrl = integrate.quad(self._itgD_from_g, 1e-9, a, args = (gamma))
        #Warning: if I really do it from 0 to a the integral will diverge, which gives
        #me really bad results!
        return a * np.exp(intgrl[0])
        #return a * np.exp(np.divide((0.315 ** gamma) * (a ** (-3 * gamma)), (-3 * gamma)) - np.log(a))

    def Linder_compare(self, model, gamma, z, k = None, save = False):
        """
        Plots the comparison of the Linder fitting function with the growth function,
        and the residual function (the ratio function -1)in the same panel, 
        when gamma is the given gamma, model is determined.   
        z: the redshift array.
        save: whether to save the plot as a pdf file.
        """
        #print('In Linder fitting function, the parameters are:')
        #print(self._H0)
        #print(self._omch2)
        #print(self._ombh2)
        
        fig = plt.figure(figsize = (8, 12.95))
        Z = z
        if k == None:
            k = np.logspace(-4, 1, 513)

        #The ratio of s8 and the normalized growth function
        s8lcdm0 = self._sigma8_dyn(np.array([0]))[1][0]
        #s8lcdm = self._sigma8_dyn(z)[1]
        #if isinstance(model, LCDM):
        #    s8modelratio = s8lcdm/s8lcdm0
        #else:
        s8modelratio = self.sigma8_model_ratio(model, Z, k)[1]/s8lcdm0
        
        D_1 = self._D_from_g(1, gamma)
        Dnorm = np.array([self._D_from_g(np.divide(1, 1 + z), gamma)  for z in Z])/ D_1
        
        plt.subplot(2, 1, 1)
        plt.plot(Z, s8modelratio, label = r'$\sigma_8$ numerical integration version')

        plt.plot(Z, Dnorm, '--', label = r'$g(a)$ parametrized version')


        #plt.yscale('log')

        plt.xlabel('z')

        plt.legend()
        plt.title('Comparison of $D(a)$ from $\sigma_8$ numerical integration and that from $g(a)$, with the residual')
        plt.ylabel('Direct comparison')

        plt.subplot(2, 1, 2)
        plt.plot(Z, s8modelratio / Dnorm - 1)
        plt.xlabel('z')
        plt.ylabel(r'The residual, i.e. ratio of $D(a)$: $\sigma_8$ result over $g(a)$ result minus 1')

        if save == True:
            plt.savefig('Linder_comparison.pdf', format='pdf', bbox_inches='tight', dpi=1200)
        plt.show()



          
            
            
class LCDM(Cosmosground):
    """
    A class representing the structure growth in ΛCDM model. 
    Currently solves the growth differential equation and gives σ8 values, but can have more features.
    """

    
    def __init__(self):
        """
        Initializes the ΛCDM model parameters.
        """
        super().__init__()
        #The dependence of the growth factor on the wavenumber k. 0 denotes independent, 1 denotes dependent.
        self.kdependence = 0
        
        
    #Defines and solves the growth differential equation
    def _growthlcdm(self, vec, a):
        """
        Returns: the 2nd order ode of the growth factor in the ΛCDM model in terms of a 1st order vector ode.
        
        vec: the vector consisting of D and the 1st derivative of D over a.
        """
        D, Delta = vec
        dDelta_da_1 = - ((self._derivadot(a) + 2 * self.H(a)) * Delta)/self._adot(a)
        dDelta_da_2 = 1.5 * self._Om * D/((self._adot(a) ** 2) * (a ** 3))
        dvec_da = [Delta, dDelta_da_1 + dDelta_da_2]
        return dvec_da
    
    def solvegrowth(self, z):
        """
        Returns: an array consisting of the growth factor solutions corresponding to the redshift values z, 
                 together with the redshift z to be passed down to the σ8 solver.
                 (scipy.integrate.odeint itself returns a second array: the derivative of the growth factor over a, 
                 which is discarded here because we are studying things with respect to z, the derivative gets a little tricky.
                 If the derivative array is somehow needed later on, we can come back and alter this function.)
                 
        z:  redshifts  [a number or numpy array, z >=0, if it is an array it has to be sorted in an ascending order
                        (or else there would be trouble with odeint).]
        """
        #Assertions
        if type(z) in (int, float, np.int64, np.float64):
            assert z >= 0, 'Redshift must be greater than zero'
        elif type(z) == np.ndarray:
            assert np.all(z[:-1] <= z[1:]) and z[0] >= 0, 'Redshift must be sorted in an ascending order to solve the ODE'
        else:
            raise TypeError('Redshift must be a number')
        
        #Gives the initial conditions
        vec0 = [self._D1(), self._D1deriv()]

        #Gives the scale factor array and adds the initial condition, rearranging in an ascending order
        a = np.append(1/(1 + z), np.array([self._amin]))
        a = np.flip(a, 0)

        #Solves the differential equation and deletes the initial condition solution afterwards, rearranging back to z
        sollcdm = integrate.odeint(self._growthlcdm, vec0, a, rtol = 10 ** (-13), atol = 0).T[0]
        sollcdm = np.delete(sollcdm, 0)
        sollcdm = np.flip(sollcdm, 0)
        
        return sollcdm
    
    #The growth factor at redshift zero
    def _Dz0(self):
        """
        Returns: the growth factor at redshift zero. Just a fixed number; it will basically remain unchanged once you 
        calculate it using enough number of redshift points.
        """
        z = np.linspace(0, 3, 30)
        return self.solvegrowth(z)[0]
    
    
    
    
class HuSawicki(Cosmosground):
    """
    A class representing the structure growth in the Hu-Sawicki fR model. 
    Currently solves the growth differential equation and gives σ8 values, but can have more features.
    """
    
    #Getters and setters
    def get_fr0exp(self):
        """
        Returns: the exponential of the fr0 parameter.
        """
        return self._fr0exp
    
    def get_fr0(self):
        """
        Returns: the fr0 parameter.
        """
        return self._fr0
    
    def set_fr0exp(self, value):
        """
        Sets fr0exp to value, and then subsequently set fr0 to 10^(value).
        Stackoverflow says that I don't really need a manual assertion.
        """
        self._fr0exp = value
        if value == 0:
            self._fr0 = value
        else:
            self._fr0 = 10.0**(self._fr0exp)
            
    def set_fr0(self, value):
        """
        Sets fr0exp to value, and then subsequently set fr0 to log_10(value).
        In order to do derivatives I have to keep both switches available...sigh
        """
        self._fr0 = value
        if value == 0:
            self._fr0exp = value
        else:
            self._fr0exp = np.log10(self._fr0)
        
    def get_n(self):
        """
        Returns: the n parameter in the Hu-Sawicki model.
        """
        return self._n
    
    def set_n(self, value):
        """
        Sets n to value.
        """
        self._n = value
                
    #A general parameter setting function that can be used by modules like Fisher_MG_final, etc.
    def setpars(self, pars):
        """
        Sets the parameters to pars. The assertions already taken care of by individual setters.
        pars: parameters (array-like).
        """
        self.set_fr0exp(pars[0])
        self.set_n(pars[1])
        
    def get_axeslabels(self):
        """
        Returns: the labels to put in the axes especially in drawing the confidence ellipse.
        """
        return self._axeslabels
    
    def set_axeslabels(self, words):
        """
        Sets the axeslabels to value. Must be a list of two strings.
        """
        assert type(words) == list and len(words) == 2, 'Labels must be a length-2 list of strings'
        assert type(words[0]) == str and type(words[0]) == str, 'Labels must be a length-2 list of strings'
        self._axeslabels = words
        
    def get_printables(self):
        """
        Returns: the names to put in the fisher analysis print statements.
        """
        return self._printables
    
    def set_printables(self, words):
        """
        Sets the printables to value. Must be a list of two strings.
        """
        assert type(words) == list and len(words) == 2, 'Labels must be a length-2 list of strings'
        assert type(words[0]) == str and type(words[0]) == str, 'Labels must be a length-2 list of strings'
        self._printables = words
        
   
    def __init__(self):
        """
        Initializes the HS model parameters, as well as various labels to put inside of the Fisher analysis.
        """
        super().__init__()
        self.kdependence = 1
        #Hu-Sawicki parameters
        self._n = 1
        self._fr0exp = -6
        self._fr0 = 10**(self._fr0exp)
        self._axeslabels = [r'$f_{R0}$ exponent', r'$n$']
        self._printables = ['f_R0 (exponent)', 'n']

    #Defining mass terms and effective g factor in fR
    def _m(self, a):
        '''
        The mass term \mu in the effective g factor in the growth differential equation in fR model.
        '''
        #if self._fr0 == 0:
        #    np.seterr(divide='ignore', invalid='ignore') #Get rid of the Runtime error since the final results are still fine
        return (1/self._c) * np.sqrt(1/((self._n + 1) * self._fr0)) * np.sqrt((self._Om + 4 * (1 - self._Om)) ** (- self._n - 1)) * np.sqrt((self._Om/(a ** 3) + 4 * (1 - self._Om)) ** (2 + self._n))
    
    def _gefffR(self, a, k):
        """
        The effective term corresponding to modified gravity to be added into the growth function equation.
        The full derivation of this term involves a deep understanding of the modified gravity model...I will learn it
        eventually!!
        
        k: the wavenumber in the k space of the matter power spectrum.
        a: the scale factor.
        """
        if self._fr0 == 0:
            return 0
        else:
            return k ** 2/(3 * (k ** 2 + (a * self._m(a)) ** 2)) 
    
    #Defines and solves the differential equation

    #Below we use the odeint in scipy.
    #Primary conclusion is that solve_ivp is much slower than odeint for a problem like what I have right now. Maybe it's because my problem is still a small problem......instead of a relatively large ode problem. According to this post https://github.com/scipy/scipy/issues/8257, direct RK45 is also gonna be slower......I'll keep odeint for now.
    def _growthfR(self, vecfr, a, k):
        """
        Returns: the 2nd order ode of the growth factor in the fR model in terms of a 1st order vector ode.
        The original mathematica code:
        Eqn[k_, a_] = D[p[a], a, a] + (adot*Ddot + 2 (adot^2)/a) D[p[a], a]/(adot^2) == 
        (1.5*Om /(a^3))  p[a] (1 + geff[k, a])/(adot^2).
        vecfr: the vector consisting of D and the 1st derivative of D over a.
        """
        Dmg, Deltamg = vecfr
        dDeltamg_da_1 = - (self._derivadot(a) + 2 * self.H(a)) * Deltamg/self._adot(a)
        dDeltamg_da_2 = (1.5 * self._Om * (1 + self._gefffR(a, k)) * Dmg)/((self._adot(a) ** 2) * (a ** 3))
        dvecfr_da = [Deltamg, dDeltamg_da_1 + dDeltamg_da_2]
        return dvecfr_da


    def solvegrowth(self, z, k):
        """
        Returns: the solution to the _growthfR given k and z.
        k & z: numpy arrays.
        """
        #Assertions
        if type(z) in (int, float, np.int64, np.float64):
            assert z >= 0, 'Redshift must be greater than zero'
        elif type(z) == np.ndarray:
            assert np.all(z[:-1] <= z[1:]) and z[0] >= 0, 'Redshift must be sorted in an ascending order to solve the ODE'
        else:
            raise TypeError('Redshift must be a number')
        
        #print('fr0 is now')
        #print(self._fr0)
        #Gives the initial conditions
        vec0 = [self._D1(), self._D1deriv()]
        
        #Gives the scale factor array and adds the initial condition, rearranging in an ascending order
        a = np.append(1/(1 + z), np.array([self._amin]))
        a = np.flip(a, 0)

        #Solves the differential equation
        #This section is trying to use vectorize.
        
        sol = np.zeros((len(k) , len(a) - 1))
        for i in range(len(k)):
            solorig = integrate.odeint(self._growthfR, vec0, a, args = (k[i], ), rtol = 10 ** (-13), atol = 0).T[0]
            sol[i] = np.flip(np.delete(solorig, 0), 0)
        return sol
    
    
    #Convergence tests and productions of numerical derivatives  
    
    #This function is calculating dr/d(log10(f_R0))
    def _frtestderiv(self, fr0_exp, n, z, k, save):
        """
        Tests the convergence of the numerical derivative partial f_R in the Hu-Sawicki fR model. Inherited paramters.
        fr0_exp: the fiducial value of the exponential of f_R0. f_R0 = 0 if the input is 'NegInfty'.
        """
        self.set_n(n)        

        if fr0_exp == 0:
            print('The fiducial is now 0, we can only test partial derivative over f_R0 itself')
            #Numerical derivative with half stepsize 1e-9 
            #(This stepsize is so that it should be distinguishable with 1e-7 which is a value of f_R0.
            
            #Won't consider the possibility of mixing up with 1e0 = 1, since in the literature it is restricted
            #that f_{R0} < 1e-2.
            
            #Don't even know whether this is right since this is still varying f_R0 itself, while I can't do varying
            #in the exponent when f_R0 = 0. But currently we don't really use this part, so might deal with it later.

            #This part employs a modified higher-order derivative at the boundary. 
            #$f'(x) = \frac{18f(x + h) - 9f(x + 2h) + 2f(x + 3h) - 11f(x)}{6h}$
            steps = np.array([9e-9, 1e-8])
            sfder = []
            for i in range(len(steps)):
                self.set_fr0(0)
                sf0 = self.sigma8_ratio(self, z, k)
                self.set_fr0(steps[i])
                sf1 = self.sigma8_ratio(self, z, k)
                self.set_fr0(2*steps[i])
                sf2 = self.sigma8_ratio(self, z, k)
                self.set_fr0(3*steps[i])
                sf3 = self.sigma8_ratio(self, z, k)

                sfder.append((18*sf1 - 9*sf2 + 2*sf3 - 11*sf0)/(6*steps[i]))
        
            
        else:
            print('Testing partial derivative over log10(f_R0)')

            steps = np.array([0.1, 0.05])
            sfder = []
            for i in range(len(steps)): 
                #Numerical derivative with half stepsizes on the exponential. This part employs the five-point stencil method,
                #in order to find out what's going on with the scipy higher order derivative.
                self.set_fr0exp(fr0_exp - 2*steps[i])
                sf_2 = self.sigma8_ratio(self, z, k)
                self.set_fr0exp(fr0_exp - steps[i])
                sf_1 = self.sigma8_ratio(self, z, k)
                self.set_fr0exp(fr0_exp + steps[i])
                sf1 = self.sigma8_ratio(self, z, k)
                self.set_fr0exp(fr0_exp + 2*steps[i])
                sf2 = self.sigma8_ratio(self, z, k)
                sfder.append((sf_2 - 8*sf_1 + 8*sf1 - sf2)/(12*steps[i]))
                
        
        if np.allclose(sfder[1], sfder[0], rtol = 1e-03, atol = 0):
            print('Partial derivative over log10(f_R0) converges nicely, proceeding to partial derivative over n')
            #For the Hu-Sawicki model in particular, we do a log-correction (in order to be more precise...although I don't
            #know how powerful that would be for our case
            return np.array(sfder)
        else:
            print((sfder[1] - sfder[0])/sfder[1])
            raise RuntimeError('Partial derivative convergence over f_R failed, please check your model')


    #Checking the chain rule
    #This function uses the same differences in the ratio of sigma_8, but the stepsize in the denominator is different, i.e.
    #this function is directly implementing dr/df_R0.
    def _frtestderiv2(self, fr0_exp, n, z, k, save):
        """
        Tests the convergence of the numerical derivative partial f_R in the Hu-Sawicki fR model. Inherited paramters.
        fr0_exp: the fiducial value of the exponential of f_R0. f_R0 = 0 if the input is 'NegInfty'.
        """
        self.set_n(n)
        print('Testing partial derivative (2nd) over f_R0')
        #Numerical derivative with half stepsize 1e-8 on f_R0 itself
        self.set_fr0exp(fr0_exp)
        print('Now f_R0 is')
        print(self.get_fr0())
        self.set_fr0exp(fr0_exp + 0.02)
        fr11 = self.get_fr0()
        sf11 = self.sigma8_ratio(self, z, k)
        self.set_fr0exp(fr0_exp - 0.02)
        fr12 = self.get_fr0()
        sf12 = self.sigma8_ratio(self, z, k)
        #Test if it's the set_fr0exp that was the problem
        sfder1 = (sf12-sf11)/(fr12-fr11)
        #sfder1 = (sf12-sf11)/(-0.04)

        #Numerical derivative with half stepsize 0.01 on the exponential
        self.set_fr0exp(fr0_exp + 0.01)
        fr21 = self.get_fr0()
        sf21 = self.sigma8_ratio(self, z, k)
        self.set_fr0exp(fr0_exp - 0.01)
        fr22 = self.get_fr0()
        sf22 = self.sigma8_ratio(self, z, k)
        sfder2 = (sf22-sf21)/(fr22-fr21)
        #sfder2 = (sf22-sf21)/(-0.02)

        #return np.array([sfder4, sfder3, sfder0, sfder1, sfder2])

        if np.allclose(sfder1, sfder2, rtol = 1e-03, atol = 0):
            print('Partial derivative over f_R converges nicely, proceeding to partial derivative over n')
            return np.array([sfder1, sfder2])
        else:
            print((sfder1 - sfder2)/sfder2)
            raise RuntimeError('Partial derivative convergence over f_R failed, please check your model')

        
    #n numerical derivatives
    def _ntestderiv(self, fr0_exp, n, z, k, save):
        """
        Tests the convergence of the numerical derivative partial n in the Hu-Sawicki fR model. Inherited parameters.
        """
        if fr0_exp == 0:
            self.set_fr0(0)
        else:
            self.set_fr0exp(fr0_exp)
        print('Testing partial derivative over n')

        steps = np.array([0.1, 0.05])
        snder = []
        for i in range(len(steps)): 
            self.set_n(n - 2*steps[i])
            sn_2 = self.sigma8_ratio(self, z, k)
            self.set_n(n - steps[i])
            sn_1 = self.sigma8_ratio(self, z, k)
            self.set_n(n + steps[i])
            sn1 = self.sigma8_ratio(self, z, k)
            self.set_n(n + 2*steps[i])
            sn2 = self.sigma8_ratio(self, z, k)
            snder.append((sn_2 - 8*sn_1 + 8*sn1 - sn2)/(12*steps[i]))

        '''
        return np.array([snder0, snder3, snder1, snder2])
        '''
        if np.allclose(snder[0], snder[1], rtol = 1e-02, atol = 0):
            print('Partial derivative over n converges nicely')
            return np.array(snder)
        elif (np.max(snder[0]) < 1e-15 and np.max(snder[1]) < 1e-15):
            print('Warning: when f_R0 = 0, partial derivative over n is essentially zero, the derivative plot may be showing calculation noise')
            return np.array(snder)
        else:
            print((snder[0] - snder[1])/snder[1])
            raise RuntimeError('Partial derivative convergence over n failed, please check your model')
        
        
    #Main callable
    def testderiv(self, fr0_exp, n, z, k, save = False):
        """
        Tests the convergence of the numerical derivative partial f_R in the Hu-Sawicki fR model, and the numerical 
        derivative partial n.
        Inherits the parameters in the model.

        Criterion of convergence: 
        For all the data points taken, the difference between the two numerical derivatives 
        lies within 0.1% of the 1st array of derivatives taken (the larger step derivative arrays).

        If the convergence is reached, the function also returns the 2 respective numerical derivatives for later use, 
        with f_R derivatives first, choosing the derivative arrays with the smaller steps.

        Parameters:
        p: the negative power of the parameter fR0.
        n: the n factor inside of the fR0 model.
        z: numpy array, redshift bins.
        save: whether to save the convergence test pdf. If True, then save.
        """
        
        #fR numerical derivatives
        dfr = self._frtestderiv(fr0_exp, n, z, k, save)
        #print('dfr is')
        #print(dfr)
        #print(type(dfr))
        
        #n numerical derivatives
        dn = self._ntestderiv(fr0_exp, n, z, k, save)
        #print('dn is')
        #print(dn)
        #print(type(dn))
        
        if type(dfr) in (np.ndarray, tuple, list) and type(dn) in (np.ndarray, tuple, list):
            
            fig = plt.figure(figsize=(8, 12))
            plt.subplot(2, 1, 1)
            
            #plt.scatter(z, dfr[0], c = 'orange', s = 2, label = '$f_{R0}$ exponent derivatives, half-step $1$')
            #plt.scatter(z, dfr[1], c = 'pink', s = 1, label = '$f_{R0}$ exponent derivatives, half-step $0.5$')
            #plt.scatter(z, dfr[2], c = 'blue', s = 1, label = '$f_{R0}$ exponent derivatives, half-step $0.1$')
            if fr0_exp == 0:
                plt.scatter(z, dfr[0], c = 'green', s = 0.5, label = '$f_{R0}$ derivatives, half-step 9e-9')
                plt.scatter(z, dfr[1], c = 'red', s = 0.1, label = '$f_{R0}$ derivatives, half-step 1e-8')
            else :
                plt.scatter(z, dfr[0], c = 'green', s = 0.5, label = '$f_{R0}$ exponent derivatives, half-step $0.1$')
                plt.scatter(z, dfr[1], c = 'red', s = 0.1, label = '$f_{R0}$ exponent derivatives, half-step $0.05$')
            plt.xlabel('z')
            plt.legend()
            
            plt.subplot(2, 1, 2)
            
            #plt.scatter(z, dn[0], c = 'orange', s = 2, label = '$n$ derivatives, half-step $100$ percent')
            #plt.scatter(z, dn[1], c = 'pink', s = 1, label = '$n$ derivatives, half-step $50$ percent')
            
            plt.scatter(z, dn[0], c = 'blue', s = 0.5, label = '$n$ derivatives, half-step $0.1$')
            plt.scatter(z, dn[1], c = 'black', s = 0.1, label = '$n$ derivatives, half-step $0.05$')
            plt.xlabel('z')
            plt.legend()
            
            if save == True:
                plt.savefig('f_R0_exp' + str(fr0_exp) + 'n=' + str(n) + 'derivtest.pdf', format='pdf', bbox_inches='tight', dpi=1200)
            plt.show() 
           
            #print(dfr[1], dn[1])
            return np.array([dfr[-1], dn[-1]])
        else:
            raise RuntimeError('Error in the convergence test for partial derivatives')
    
        
        
        

     
    
class DGP(Cosmosground):
    """
    A class representing the structure growth in the DGP model. 
    It includes the sDGP and nDGP model because the only difference is the β term which is off by a sign.
    Currently solves the growth differential equation and gives σ8 values, but can have more features.
    """
    
    #Getters and setters 
    def get_rc(self):
        """
        Returns: the rc value in this model.
        """
        return self._rc
    
    def set_rc(self, value):
        """
        Sets rc to value. 
        value: a number, in general from 1 to 4.
        """
        assert type(value) in (int, float, np.int64, np.float64), 'value must be a number'
        self._rc = value
        self._n = value*self.H(1)
        
    #A general parameter setting function that can be used by modules like Fisher_MG_final, etc.
    def setpars(self, pars):
        """
        Sets the parameters to pars. The assertions already taken care of by individual setters.
        pars: an array of numbers, the 1st being the branch, the 2nd being the rc value.
        """
        #self.set_branch(pars[0])
        self.set_rc(pars)
     
    
    def __init__(self):
        """
        Initializes the DGP model parameters.
        """
        super().__init__()
        self.kdependence = 0
        #DGP parameters
        #self._branch = 1
        self._rc = 3
        self._n = self._rc*self.H(1)
    
    #Defining mass terms and effective g factor in DGP
    def _geffDGP(self, a0):
        """
        The corresponding geff parameter for the DGP model. This time it only depends on rc.
        Due to the limits of np.derivative as represented in derivHa(a0) (it can only take the derivative at one point),
        in order for the a's to not mix up with each other, the argument here is noted as a0.
        Code from mathematica:
        betaDGP[a_] = 1 + 2 n H (1 + D[H, a] adot/(3 H H)); (+ for nDGP, - for sDGP)
        beta[a_] = Sqrt[1/(6 betaDGP[a])];
        geff[a_] = 2 (beta[a_])^2;
        (Note that when you include sDGP there's the sign problem so you have to do the sqrt
        """
        betaDGP = 1 + 2 * self._n * self.H(a0) * (1 + (self._derivHa(a0) * self._adot(a0))/(3 * (self.H(a0)) ** 2))
        geff = 1/(3 * betaDGP)
        return geff
    
    #Defines and solves the differential equation
    def _growthDGP(self, vecn, a):
        """
        Returns: the 2nd order ode of the growth factor in the DGP model in terms of a 1st order vector ode.
        The original mathematica code:
        Eqn[a_] = 
        D[p[a], a, a] + (adot*Ddot + 2 (adot^2)/a) D[p[a], a]/(adot^2) == (1.5*Om /(a^3))  p[a] (1 + geff[a])/(adot^2).
        vecn: the vector consisting of D and the 1st derivative of D over a.
        """
        Dn, Deltan = vecn
        dDeltan_da_1 = - ((self._derivadot(a) + 2 * self.H(a)) * Deltan)/self._adot(a)
        dDeltan_da_2 = (1.5 * self._Om * (1 + self._geffDGP(a)) * Dn)/((self._adot(a) ** 2) * (a ** 3))
        dvecn_da = [Deltan, dDeltan_da_1 + dDeltan_da_2]
        return dvecn_da

    def solvegrowth(self, z):
        """
        Returns: the solution to _growthDGP given z.
        z: numpy arrays.
        """
        #Assertions
        if type(z) in (int, float, np.int64, np.float64):
            assert z >= 0, 'Redshift must be greater than zero'
        elif type(z) == np.ndarray:
            assert np.all(z[:-1] <= z[1:]) and z[0] >= 0, 'Redshift must be sorted in an ascending order to solve the ODE'
        else:
            raise TypeError('Redshift must be a number')
        
        #Gives the initial conditions
        vec0 = [self._D1(), self._D1deriv()]
        
        #Gives the scale factor array and adds the initial condition, rearranging in an ascending order
        a = np.append(1/(1 + z), np.array([self._amin]))
        a = np.flip(a, 0)

        #Solves the differential equation
        sol = integrate.odeint(self._growthDGP, vec0, a, rtol = 10 ** (-13), atol = 0).T[0]
        sol = np.flip(np.delete(sol, 0), 0)
        return sol
    
    #The growth factor at redshift zero
    def _Dz0(self):
        """
        Returns: the growth factor at redshift zero. Just a fixed number; it will basically remain unchanged once you 
        calculate it using enough number of redshift points.
        """
        z = np.linspace(0, 3, 30)
        return self.solvegrowth(z)[0]
    
    
    #Convergence tests and productions of numerical derivatives
    #r_c partial derivatives
    def _rctestderiv(self, n, z, save):
        """
        Tests the convergence of the numerical derivative partial n (n = H0 * rc here) in the DGP model, but is called by rctestderiv in order to distinguish between the _ntestderiv in f(R) (Also trying to avoid confusion in other modules that already called this function). Inherited paramters.
        """
        print('Testing partial derivative over n')
        steps = np.array([0.1, 0.05])
        srder = []
        H0 = self.H(1)
        rc = n/H0
        for i in range(len(steps)): 
            self.set_rc(rc - 2*steps[i])
            sr_2 = self.sigma8_ratio(self, z)
            self.set_rc(rc - steps[i])
            sr_1 = self.sigma8_ratio(self, z)
            self.set_rc(rc + steps[i])
            sr1 = self.sigma8_ratio(self, z)
            self.set_rc(rc + 2*steps[i])
            sr2 = self.sigma8_ratio(self, z)
            #steps in rc * H0 is steps in n
            srder.append((sr_2 - 8*sr_1 + 8*sr1 - sr2)/(12*steps[i]*H0))

        if np.allclose(srder[0], srder[1], rtol = 5e-03, atol = 0):
            print('Partial derivative over n converges nicely')
            return np.array(srder)
        else:
            print(srder[1] - srder[0])
            raise RuntimeError('Partial derivative convergence over n failed, please check your model')
        
        
    #Main callable
    def testderiv(self, n, z, show = True, save = False):
        """
        Tests the convergence of the numerical derivative partial rc in the DGP modified gravity model.
        Inherits the parameters in the model.

        Criterion of convergence: 
        For all the data points taken, the difference between the two numerical derivatives 
        lies within 0.1% of the 1st array of derivatives taken (the larger step derivative arrays).

        If the convergence is reached, the function also returns the numerical derivatives for later use.

        Parameters:
        rc: the rc factor inside of the nDGP model.
        z: numpy array, redshift bins.
        show: whether to show the plot in the notebook. If True, then show.
        save: whether to save the convergence test pdf. If True, then save.
        """
        
        #numerical derivatives
        dn = self._rctestderiv(n, z, save)
        
        if type(dn) in (np.ndarray, list, tuple):
            if show == True:               
                fig = plt.figure(figsize=(8, 5.6))
                plt.scatter(z, dn[0], c = 'yellow', s = 0.1, label = '$r_c$ derivatives, step $10^{-4}$')
                plt.scatter(z, dn[1], c = 'cyan', s = 0.1, label = '$r_c$ derivatives, step $10^{-6}$')
                plt.legend()
                plt.xlabel('z')

                if save == True:
                    plt.savefig('DGPn=' + str(rc*Cm.H(1)) + 'derivtest.pdf', format='pdf', bbox_inches='tight', dpi=1200)
                plt.show() 

            return dn[1]
