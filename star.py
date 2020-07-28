import numpy as np
import pkg_resources
import pandas

import astropy.units as u
import astropy.constants as c
import astropy.io.fits as fits

def bpgs_list(spectype=None,verbose=False):
    '''
    Returns pandas dataframe with the list of the files, the name and the type of the star

    '''
    fname = pkg_resources.resource_filename('exoscene', 'bpgs/bpgs_readme.csv')
    dat = pandas.read_csv(fname)
    if spectype is not None: dat = dat[dat['Type']==spectype]
    if verbose: print(dat)
    return dat

def bpgs_spectype_to_photonrate(spectype,Vmag,minlam,maxlam):
    '''
    Parameters
    ----------
    spectype: string
        String representing the spectral type of the star
    Vmag: float
        V magnitude of star
    minlam: float
        Minimum wavelength of the band in nm    
    maxlam: float
        Maximum wavelength of the band in nm    
    
    Returns
    -------
    val: Quantity
        Photons/second/m2 coming from the star within the band
    '''
    dat = bpgs_list(verbose=False)
    subset = dat[dat['Type']==spectype]
    if len(subset) > 0:
        specnum = dat[dat['Type']==spectype].index[0]+1
        fname = pkg_resources.resource_filename('exoscene', '/bpgs/bpgs_' + str(specnum)+ '.fits')
    
        return bpgsfile_to_photonrate(fname,Vmag,minlam,maxlam)
    else:
        print('No corresponding spectral type in database, check exoscene/bpgs/bpgs_readme.csv')


def bpgs_to_photonrate(specnum,Vmag,minlam,maxlam):
    '''
    Parameters
    ----------
    specnum: int
        Number of spectrum file from bpgs folder (in crispy/Input/bpgs folder)
    Vmag: float
        V magnitude of star
    minlam: float
        Minimum wavelength of the band in nm    
    maxlam: float
        Maximum wavelength of the band in nm    
    
    Returns
    -------
    val: Quantity
        Photons/second/m2 coming from the star within the band
    '''
    fname = pkg_resources.resource_filename('exoscene', '/bpgs/bpgs_' + str(specnum)+ '.fits')
    return bpgsfile_to_photonrate(fname,Vmag,minlam,maxlam)

def bpgsfile_to_photonrate(filename,Vmag,minlam,maxlam):
    '''
    Parameters
    ----------
    filename: string
        Text file corresponding to the bpgs specturm in pysynphot catalog
    Vmag: float
        V magnitude of star
    minlam: float
        Minimum wavelength of the band in nm    
    maxlam: float
        Maximum wavelength of the band in nm    
    
    Returns
    -------
    val: Quantity
        Photons/second/m2 coming from the star within the band
    '''
    wavel = np.arange(minlam,maxlam)
    star = input_star(filename,Vmag,wavel)
    return (np.sum(star)*u.nm).to(u.photon/u.m**2/u.s)
    

def input_star(filename,Vmag,wavel):
    '''
    Parameters
    ----------
    filename: string
        Text file corresponding to the bpgs specturm in pysynphot catalog
    Vmag: float
        V magnitude of star
    wavel: array
        Array of desired wavelengths in nanometers
        
    Returns
    -------
    val: Quantity
        Photons/second/m2/nm coming from the star for each input wavelength bin
    '''
    fopen = fits.open(filename)
    f = np.array(fopen[1].data)
    # files are in erg s-1 cm-2 Ang-1
    flam = u.erg/u.s/u.cm**2/u.Angstrom
    fac = flam.to(u.W/u.m**2/u.nm)
    dat = f['FLUX']*fac*10**(-0.4 * Vmag)
    wav = f['WAVELENGTH']/10.
    func = scipy.interpolate.interp1d(wav,dat,bounds_error=False,fill_value=0.0)
    flux = func(wavel)*u.W/u.m**2/u.nm
    Eph = (c.c*c.h/(wavel*u.nm)/u.photon).to(u.J/u.photon)
    return (flux/Eph).to(u.photon/u.s/u.m**2/u.nm)

def calc_contrast(wavelist, distance, radius, filename, albedo=None):
    '''
    Function calcContrast

    Returns the flux ratio Fplanet/Fstar at given wavelengths.

    Parameters
    ----------
    wavelist :   1D array, list
            Array of wavelengths in nm at which the contrast is computed.
    distance : float
            Distance in AU between the planet and the star
    radius: float
            Radius of planet in units of Jupiter radii
    filename: string
            Two-column file with first column as the wavelength in microns, second column is the geometrical albedo
    albedo: float
            If None, then the albedo is given by the contents of the text file. If not None, the geometrical albedo given
            in the text file is normalized to have its maximum within the wavelist range to be albedo.

    Returns
    -------
    vals : 1D array
            Array of flux ratio at the desired wavelengths.

    '''

    spectrum = np.loadtxt(filename)
    spec_func = scipy.interpolate.interp1d(spectrum[:, 0] * 1000., spectrum[:, 1])
    vals = spec_func(wavelist)
    if albedo is not None:
        vals /= np.amax(vals)
        vals *= albedo

    vals = vals * (radius * c.R_jup.to(u.m) / (distance * u.AU).to(u.m))**2
    return vals
