##Python 2 Version

import numpy as np
import astropy.units as u
import astropy.constants as c
import astropy.analytic_functions as af
try:
    from astropy.io import fits
except BaseException:
    import pyfits as fits
from scipy.interpolate import interp1d
import pkg_resources

def adjust_krist_header(cube, lamc, pixsize=None):
    '''
    Force a different central wavelength, assuming that everything is wavelength-independent

    Parameters
    ----------

    cube: 3D float array
        Input cube from J. Krist
    lamc: float
        Central wavelength to override in nm
    pixsize: float
        Pixel scale at central wavelength to override (in lambda/D)

    '''
    oldlam = cube.header['LAM_C']
    cube.header['LAM_C'] = lamc / 1000.
    cube.header['LAM_MIN'] *= lamc / 1000. / oldlam
    cube.header['LAM_MAX'] *= lamc / 1000. / oldlam

    if pixsize is not None:
        cube.header['PIXSIZE'] = pixsize


def convert_krist_cube(cubeshape, lamlist, star_T, star_Vmag, tel_area):
    '''

    Function convert_krist_cube

    This function calculates the number of photons per second
    entering the WFIRST obscured aperture, given the star `Vmag` and its temperature
    for each slice of the input cube
    This was only tested with John Krist's cubes and his normalization.

    Parameters
    ----------
    cubeshape: tuple
        Indicates the shape of the cube that needs to be multiplied
    lamlist: 1D array
        Wavelength array in nm corresponding to the cube slices
    star_T: float
        Stellar temperature in K
    star_Vmag: float
        Stellar magnitude in V band
    tel_area: units.m**2
        Area of telescope in units of units.m**2

    Returns
    -------
    newcube: 3D array
        Cube that multiplies an input normalized cube from John Krist to turn it into units
        of photons/s/nm/pixel. Each slice from the product cube subsequently needs to be multiplied by the
        bandwidth of each slice to determine the photons/s/pixel incident on the telescope

    '''

    # We need to determine the coefficient of proportionality between a blackbody source and the
    # actualy flux received (depends on size of the star, distance, etc)
    # define Vband
    lambda_cent = 550 * u.nanometer

    # this is the flux density per steradian (specific intensity) you would
    # expect from Vband
    flux_bb_F550 = af.blackbody_lambda(
        lambda_cent, star_T).to(
        u.Watt / u.m**2 / u.um / u.sr)

    # this is the actual flux density received in Vband
    Vband_zero_pt = (3636 * u.Jansky).to(u.Watt / u.m**2 / u.Hertz)
    Vband_zero_pt *= (c.c / lambda_cent**2)
    flux_star_Vband = Vband_zero_pt * 10**(-0.4 * star_Vmag)

    # the ratio is coefficient we seek; this will multiply a blackbody function to yield flux densities
    # at all wavelengths
    ratio_star = (flux_star_Vband / flux_bb_F550)

    # this is the ratio which we want to multiply phot_Uma_Vband for the other bands
    #print("Ratio of blackbodies is %f" % ratio_Uma)

    # Now convert each slice to photons per second per square meter
    dlam = lamlist[1] - lamlist[0]
    newcube = np.zeros(cubeshape) * u.photon / u.s / u.m**2 / u.nm
    for i in range(len(lamlist)):
        E_ph = (
            c.h *
            c.c /
            lamlist[i] /
            u.photon).to(
            u.J /
            u.photon)  # photon energy at middle frequency
        BBlam = af.blackbody_lambda(
            lamlist[i], star_T).to(
            u.Watt / u.m**2 / u.nm / u.sr)
        flux = (
            BBlam *
            ratio_star).to(
            u.W /
            u.m**2 /
            u.nm)  # this is Watts per m2 per nm
        photon_flux = flux / E_ph  # This is in Photons per second per m2 per nm
        # add value to entire cube since we will multiply Krist's cube
        newcube[i, :, :] += photon_flux.to(u.photon / u.s / u.m**2 / u.nm)

    # multiply by the number of wavelengths since this is the way J. Krist
    # normalizes his cubes
    newcube *= tel_area * len(lamlist)
    # note that this is still per unit nanometer of bandwidth, so it still
    # needs to be multipled by dlam in the IFS.
    return newcube

import pandas as pd

def bpgs_list(spectype=None,verbose=False):
    '''
    Returns pandas dataframe with the list of the files, the name and the type of the star

    '''
    fname = pkg_resources.resource_filename('crispy', 'Inputs') + '/bpgs/bpgs_readme.csv'
    dat = pd.read_csv(fname)
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
    if len(subset>0):
        specnum = dat[dat['Type']==spectype].index[0]+1
        fname = pkg_resources.resource_filename('crispy', 'Inputs') + '/bpgs/bpgs_' + str(specnum)+ '.fits'
    
        return bpgsfile_to_photonrate(fname,Vmag,minlam,maxlam)
    else:
        print('No corresponding spectral type in database, check crispy/Input/bpgs/bpgs_readme.csv')


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
    fname = pkg_resources.resource_filename('crispy', 'Inputs') + '/bpgs/bpgs_' + str(specnum)+ '.fits'
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
        http://www.stsci.edu/hst/observatory/crds/astronomical_catalogs.html
    Vmag: float
        V magnitude of star
    wavel: array
        Array of desired wavelengths in microns
        
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
    func = interp1d(wav,dat,bounds_error=False,fill_value=0.0)
    flux = func(wavel)*u.W/u.m**2/u.nm
    Eph = (c.c*c.h/(wavel*u.nm)/u.photon).to(u.J/u.photon)
    return (flux/Eph).to(u.photon/u.s/u.m**2/u.nm)


def haystacks_to_photons(haystacks_hdu):
    '''
    Function haystacks_to_photons

    This function converts a Haystacks hdu in Jy/pixels to photons/s/nm/m^2/pixel
    and returns a full cube

    Parameters
    ----------
    cube: haystacks_hdu
        Haystacks HDU

    Returns
    -------
    hc: ndarray
        Converted cube in ph/s/um/m2
    lamlist: wavelength array

    '''

    # last extension is the list of wavelengths
    NEXT = haystacks_hdu[0].header['N_EXT']
    lamlist = haystacks_hdu[NEXT + 1].data * u.um
    lamcube = c.c / lamlist[:, np.newaxis, np.newaxis]**2

    # allocate memory
    hc = np.zeros(
        (NEXT,
         haystacks_hdu[1].data.shape[0],
         haystacks_hdu[1].data.shape[1]),
        dtype=np.float32) * u.Jy
    for i in range(NEXT):
        hc[i] = haystacks_hdu[i + 1].data * u.Jy

    # convert cube
    hc = hc.to(u.Watt / u.m**2 / u.Hertz)
    hc *= lamcube
    hc = hc.to(u.W / u.m**2 / u.um)

    # photon energy
    Eph = (c.h *
           c.c /
           lamlist[:, np.newaxis, np.newaxis] /
           u.photon).to(u.J /
                        u.photon)

    # convert to photon
    return (hc / Eph).to(u.photon / u.s / u.m**2 / u.nm), lamlist


def Jy_to_photons(cube_Jy, wavlist):
    '''
    Parameters
    ----------
        cube_Jy: 3D datacube in Jy
        wavlist: 1D array with wavelengths in microns
    Returns
    -------
        hc: 3D haystacks cube in photons/m2/um/s

    '''
    if isinstance(wavlist, u.Quantity):
        lamlist = wavlist.to(u.um)
    else:
        lamlist = wavlist * u.um
    lamcube = c.c / lamlist[:, np.newaxis, np.newaxis]**2

    hc = cube_Jy * u.Jansky

    hc = hc.to(u.Watt / u.m**2 / u.Hertz)
    hc *= lamcube
    hc = hc.to(u.W / u.m**2 / u.um)

    # photon energy
    Eph = (c.h *
           c.c /
           lamlist[:, np.newaxis, np.newaxis] /
           u.photon).to(u.J /
                        u.photon)
    hc = (hc / Eph).to(u.photon / u.s / u.m**2 / u.um)

    return hc


def zodi_cube(
        krist_cube,
        area_per_pixel,
        absmag,
        Vstarmag,
        zodi_surfmag,
        exozodi_surfmag,
        distAU,
        t_zodi):
    '''
    (obsolete)

    '''
    cube = krist_cube.copy()
    # *cube.shape[1]*cube.shape[2] # now we consider the cube as a real spectral datacube instead of a multiplication cube
    cube /= cube.shape[0]
    # each pixel now represents the number of photons per pixel per slice if
    # it was uniform across the field
    Msun = 4.83
    # where area_per_pixel has to be in square arcsec
    zodicube = cube * t_zodi * area_per_pixel * \
        10**(-0.4 * (zodi_surfmag - Vstarmag))
    # where area_per_pixel has to be in square arcsec
    exozodicube = cube * t_zodi * area_per_pixel * \
        10**(-0.4 * (exozodi_surfmag + absmag - Msun - Vstarmag)) / distAU**2
    return zodicube + exozodicube


def calc_contrast_Bijan(wavelist,
                        # default values are for 47 Uma c
                        albedo=0.28,  # in the continuum; use albedo=0 to use native albedo files from Cahoy et al
                        radius=1.27,  # in Jupiter radius
                        dist=3.6,  # in AU - note that it is different from the distance keyword below because there simply isn't a Cahoy spectrum for 3.6AU
                        # this is just to load some spectrum
                        planet_type='Jupiter',
                        abundance=1,
                        distance=5,
                        phase=90,
                        folder='/Users/mrizzo/Science/Haystacks/Cahoy_Spectra/albedo_spectra/'):
    '''
    Function calc_contrast_Bijan (obsolete)

    '''
    if folder is None:
        vals = np.ones(len(wavelist))
    else:
        filename = folder + planet_type + '_' + \
            str(abundance) + 'x_' + str(distance) + 'AU_' + str(phase) + 'deg.dat'
        spectrum = np.loadtxt(filename)
        spec_func = interp1d(spectrum[:, 0] * 1000., spectrum[:, 1])
        vals = spec_func(wavelist)
        if albedo != 0:
            vals /= np.amax(vals)
    if albedo != 0:
        vals *= albedo
    vals *= (radius * c.R_jup.to(u.m) / (dist * u.AU).to(u.m))**2
    return vals


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
    spec_func = interp1d(spectrum[:, 0] * 1000., spectrum[:, 1])
    vals = spec_func(wavelist)
    if albedo is not None:
        vals /= np.amax(vals)
        vals *= albedo

    vals = vals * (radius * c.R_jup.to(u.m) / (distance * u.AU).to(u.m))**2
    return vals
