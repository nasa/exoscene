import numpy as np
import scipy.interpolate
import pkg_resources
import pandas

import astropy.coordinates
import astropy.units as u
import astropy.constants as c
import astropy.io.fits as fits

def get_bulk_astrom_offset(t_ref, t_obs, mu_RA, mu_Dec, plx, coord_icrs):
    """
    Compute the astrometric offset of a star due to proper motion and parallax,
    for a geocentric observer. This is an early work in progress - so far it only
    accounts for proper motion.
    
    Parameters
    ----------
    t_ref - astropy quantity, date
        Reference epoch
    t_obs - astropy quantity, date
        Observing epoch
    mu_RA - astropy quantity, physical units angle / time
        Proper motion of star in EW direction
    mu_Dec - astropy quantity, physical units angle / time
        Proper motion of star in NS direction
    plx - astropy quantity, physical units angle
        Annual parallax of star
    ec_lon - Ecliptic longitude of star 
    ec_lat - Ecliptic latitude of star
    
    Returns
    -------
    offset_RA - astropy quantity, angle
        Bulk astrometric offset in EW direction w.r.t. reference epoch
    offset_Dec - astropy quantity, angle
        Bulk astrometric offset in NS direction w.r.t. reference epoch
    """
    
    pm_offset_RA = (mu_RA * (t_obs - t_ref)).to(u.milliarcsecond)
    pm_offset_Dec = (mu_Dec * (t_obs - t_ref)).to(u.milliarcsecond)
    
    coord_icrs_wpm = astropy.coordinates.SkyCoord(
            ra=coord_icrs.ra + pm_offset_RA / np.cos(coord_icrs.dec),
            dec=coord_icrs.dec + pm_offset_Dec,
            unit='deg', frame='icrs')
    
    coord_ec_wpm = coord_icrs_wpm.transform_to(
            astropy.coordinates.BarycentricTrueEcliptic)
    elon = coord_ec_wpm.lon
    elat = coord_ec_wpm.lat

    phi_orb = 2 * np.pi * (t_obs.to(u.year).value % 1)
    # d_L2Sun = (1.496e11 + 1.492e9) * u.meter
    d_L2Sun = (1 * u.AU) + (1.5 * 10**6) * u.kilometer
    d_star = (1 * u.AU / np.tan(plx)).to(u.pc)
    x_tel = d_L2Sun * np.cos(phi_orb)
    y_tel = d_L2Sun * np.sin(phi_orb)
    z_tel = 0 # Assuming L2 orbit on Ecliptical plane

    # Vector joining the star and the telescope
    x_star_tel = d_star * np.cos(elat) * np.cos(elon) - x_tel
    y_star_tel = d_star * np.cos(elat) * np.sin(elon) - y_tel
    z_star_tel = d_star * np.sin(elat)

    # Normalized position vector
    d_star_tel = np.sqrt(  x_star_tel * x_star_tel 
                         + y_star_tel * y_star_tel 
                         + z_star_tel * z_star_tel)

    x_nrm = x_star_tel / d_star_tel
    y_nrm = y_star_tel / d_star_tel
    z_nrm = z_star_tel / d_star_tel

    # Deriving the new Ecliptic coordinates
    elat_wplx = np.arcsin(z_nrm)
    elon_wplx = np.arctan2(y_nrm, x_nrm)
    #print("elon, elat = {:.4f}, {:.4f}".format(np.rad2deg(elon_wplx), np.rad2deg(elat_wplx)))
    
    coord_ec_wplx = astropy.coordinates.BarycentricTrueEcliptic(
            lon = elon_wplx, lat = elat_wplx)

    coord_icrs_wplx = coord_ec_wplx.transform_to(
            astropy.coordinates.ICRS)
    
    offset_RA, offset_Dec = coord_icrs.spherical_offsets_to(coord_icrs_wplx)
    
    return offset_RA.to(u.mas), offset_Dec.to(u.mas)

def bpgs_list(spectype=None,verbose=False):
    '''
    Returns pandas dataframe with the list of the files, the name and the type of the star

    '''
    fname = pkg_resources.resource_filename('exoscene', 'data/bpgs/bpgs_readme.csv')
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
        fname = pkg_resources.resource_filename('exoscene', 'data/bpgs/bpgs_' + str(specnum)+ '.fits')
    
        return bpgsfile_to_photonrate(fname,Vmag,minlam,maxlam)
    else:
        print('No corresponding spectral type in database, check exoscene/data/bpgs/bpgs_readme.csv')


def bpgs_to_photonrate(specnum,Vmag,minlam,maxlam):
    '''
    Parameters
    ----------
    specnum: int
        Number of spectrum file from bpgs folder (exoscene/data/bpgs/)
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
    fname = pkg_resources.resource_filename('exoscene', 'data/bpgs/bpgs_' + str(specnum)+ '.fits')
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

    if len(wavel) == 1:
        delta_lambda = (maxlam - minlam) * u.nm
    else:
        delta_lambda = 1 * u.nm
    return (np.sum(star) * delta_lambda).to(u.photon / u.m**2 / u.s)
    
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
