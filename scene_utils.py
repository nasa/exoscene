from numpy import sin,cos,sqrt,pi
import numpy as np
import numpy.testing
import scipy.ndimage
import scipy.interpolate
import skimage.transform
import os
import pkg_resources
import pandas
import csv

import matplotlib.pyplot as plt

import astropy.units as u
import astropy.constants as c
import astropy.io.fits as fits
import astropy.table

class planet:
    def __init__(self, name, ipac_df=None, dist=None, P=None,
                 a=None, inc=None, ecc=None,
                 longnode=None, tperi=None, argperi=None,
                 radius=None, albedo_wavelens=None, albedo_vals=None,
                 mplan=None, mstar=None):
        self.name = name
        
        if type(ipac_df) is pandas.core.frame.DataFrame:
            row = ipac_df[ipac_df['pl_name'] == name]
    
            self.dist = row['st_dist'].item() * u.parsec
            self.P = row['pl_orbper'].item() * u.day
            self.a = row['pl_orbsmax'].item() * u.AU
            self.ecc = row['pl_orbeccen'].item()
            self.tperi = row['pl_orbtper'].item() * u.day
            self.argperi = row['pl_orblper'].item() * u.degree
        else:
            self.dist = dist
            self.P = P
            self.a = a
            self.ecc = ecc
            self.tperi = tperi
            self.argperi = argperi
            
        if inc != None:
            self.inc = inc
        else:
            self.inc = 0 * u.degree
            
        if longnode != None:
            self.longnode = longnode
        else:
            self.longnode = 0 * u.degree
            
        if albedo_wavelens != None:
            self.albedo_wavelens = albedo_wavelens
            self.albedo_vals = albedo_vals
        else:
            self.albedo_wavelens = [575*u.nanometer]
            self.albedo_vals = [0.3]
            
        if radius != None:
            self.radius = radius
        else:
            self.radius = 1 * u.R_jup
            
        if mstar != None:
            self.mstar = mstar
        else:
            self.mstar = 1 * u.Msun

        if mplan != None:
            self.mplan = mplan
        else:
            self.mplan = 1 * u.Mjup
            
        if self.P == None:
            self.P = np.sqrt( 4*np.pi**2 / (c.G * (self.mstar + self.mplan) ) * \
                              (self.a)**3 ).to(u.year)

        if self.argperi == None:
            self.argperi = 0 * u.deg

        if self.tperi == None:
            self.tperi = 0 * u.year
            
    def compute_ephem(self, tmax, tstep):
        ts = np.arange(0, tmax.value + tstep.to(tmax.unit).value, tstep.to(tmax.unit).value) * tmax.unit
        
        Nt = len(ts)
        delx = np.zeros(Nt) * u.milliarcsecond
        dely = np.zeros(Nt) * u.milliarcsecond
        phasefunc = np.zeros(Nt)
        orad = np.zeros(Nt) * u.AU
        
        plnt_obs_vec = np.array([0.,0., self.dist.to(u.AU).value]).T
        
        for tt, epoch in enumerate(ts):      
            posvel = np.array(cartesian(a=self.a.to(u.AU).value,
                                        ecc=self.ecc, 
                                        incl=self.inc.to(u.radian).value, 
                                        longnode=self.longnode.to(u.radian).value, 
                                        argperi=self.argperi.to(u.radian).value, 
                                        meananom=0,
                                        mstar=self.mstar, mplan=self.mplan,
                                        tperi=self.tperi.to(u.year),
                                        epoch=epoch.to(u.year),
                                        inc_obs=0))
            pl_r = np.sqrt(posvel[0]**2+posvel[1]**2+posvel[2]**2) * u.AU
            cos_obs = np.dot(plnt_obs_vec, posvel[:3] / (pl_r * np.sqrt(np.sum(plnt_obs_vec**2))))
            # Beta is the planet phase, which is 180 degrees out of phase from the scattering angle.
            beta = pi - np.arccos(cos_obs)
            # Lambert phase function
            lambert_pf = (sin(beta) + (pi - beta) * cos(beta)) / pi
            phasefunc[tt] = lambert_pf
            orad[tt] = pl_r
            delx[tt] = np.arctan2(posvel[0]*u.AU, self.dist).to(u.milliarcsecond)
            dely[tt] = np.arctan2(posvel[1]*u.AU, self.dist).to(u.milliarcsecond)
        
        return ts, delx, dely, phasefunc, orad
            
    def set_phase_curve(self, name, phasecurve_df, lambdac, inc, fsed, Rp=None, inc_orbit=None):
        # TODO: incorporate multiple wavelengths
        rows = phasecurve_df[phasecurve_df['Name'] == name]
        
        dmag_col_name = 'dMag_{:03d}C_{:03d}NM_I{:02d}'.format(int(100*fsed), int(lambdac.value), int(inc.value))
        pPhi_col_name = 'pPhi_{:03d}C_{:03d}NM_I{:02d}'.format(int(100*fsed), int(lambdac.value), int(inc.value))
        
        self.dmag = rows[dmag_col_name].values
        self.pPhi = rows[pPhi_col_name].values
        
        self.M = rows['M'].values * u.radian
        self.t = self.M * self.P / (2*np.pi* u.radian)
        
        if Rp == None:
            self.Rp = 1.0 * u.R_jup
        else:
            self.Rp = Rp
            
        self.fsed = fsed
        
        if inc_orbit == None:
            self.inc_orbit = inc
        else:
            self.inc_orbit = inc_orbit
            
def approx_E(e,M):

    error = 1.e-8
    if M<pi:
        E = M+e/2
    else:
        E = M-e/2
    ratio = 1
    while abs(ratio) > error:
        ratio = (E-e*sin(E) - M)/(1.0-e*cos(E)) #new ratio
        E = E - ratio  #updated value of E
    return E

def cartesian(a, 
              ecc, 
              incl, 
              longnode, 
              argperi, 
              meananom,
              mstar,
              mplan,
              tperi = 0.0,
              epoch = 0.0,
              inc_obs=None,
             ):
    pio180 = pi/180.
    # Modified June 2019 to enable the application of absolute times for 
    # time of periastron and epoch.
    #
    # Modified Sep 2018 to allow for variable star mass.
    
    #GM = 4. * pi * pi
    #period = a**1.5
    period = np.sqrt( 4*pi**2 / (c.G * (mstar + mplan) ) * (a*u.AU)**3 ).to(u.year)
    # Compute orbital periods and astrometric signatures
    # pPeriod = np.sqrt( 4*np.pi**2 / (astropy.constants.G * (sMass + pMass) ) * pSMA**3 ).to(u.year)
    mean_anomaly = meananom + 2.*pi * (epoch - tperi).value / period.value
    
    # first, compute ecc. anom
    if not isinstance(mean_anomaly,float):
        E = np.array([approx_E(e,m) for (e,m) in zip(ecc,mean_anomaly)]) 
    else:
        E = approx_E(ecc,mean_anomaly)
    cosE = cos(E)
    sinE = sin(E)

    # compute unrotated positions and velocities
    foo = sqrt(1.0 - ecc*ecc)
#    meanmotion = sqrt(GM/(a*a*a))
    meanmotion = 2*pi/period.value
    x = a * (cosE - ecc)
    y = foo * (a * sinE)
    z = np.zeros_like(y)
    denom = 1. / (1.0 - ecc * cosE)
    xd = (-a * meanmotion * sinE) * denom
    yd = foo * (a * meanmotion * cosE * denom)
    zd = np.zeros_like(yd)

    # rotate by argument of periastron in orbit plane
    cosw = cos(argperi)
    sinw = sin(argperi)
    xp = x * cosw - y * sinw
    yp = x * sinw + y * cosw
    zp = z
    xdp = xd * cosw - yd * sinw
    ydp = xd * sinw + yd * cosw
    zdp = zd

    #rotate by inclination about x axis
    cosi = cos(incl)
    sini = sin(incl)
    x = xp
    y = yp * cosi - zp * sini
    z = yp * sini + zp * cosi
    xd = xdp
    yd = ydp * cosi - zdp * sini
    zd = ydp * sini + zdp * cosi

    #rotate by longitude of node about z axis
    cosnode = cos(longnode)
    sinnode = sin(longnode)
    xf = x * cosnode - y * sinnode
    yf = x * sinnode + y * cosnode
    zf = z
    vx = xd * cosnode - yd * sinnode
    vy = xd * sinnode + yd * cosnode
    vz = zd
    x = xf
    y = yf
    z = zf
    
    if inc_obs is not None:
        cos_i = cos(inc_obs * pio180)
        sin_i = sin(inc_obs * pio180)
        y = yf*cos_i - zf*sin_i
        z = yf*sin_i + zf*cos_i
        vyf = vy.copy()
        vzf = vz.copy()
        vy = vyf*cos_i - vzf*sin_i
        vz = vyf*sin_i + vzf*cos_i
    
    return x,y,z,vx,vy,vz

def write_ephem_table(planet, tstep, tspan, table_fname):
    tseries, delx, dely, phasefunc, orad = planet.compute_ephem(tspan, tstep)
   
    ephem_table = astropy.table.QTable(data = [tseries, delx, dely, phasefunc, orad],
                                               names = ['t (years)', 'delta x (mas)', 'delta y (mas)',
                                               'phase', 'r (AU)'])
    
    for (wavel, albedo) in zip(planet.albedo_wavelens, planet.albedo_vals):
        fluxratio = phasefunc * albedo * (planet.radius.to(u.AU) / orad)**2
        col_name = 'fluxratio_{:d}'.format(int(wavel.to(u.nm).value))
        ephem_table[col_name] = fluxratio
    
    ephem_table.write(table_fname, format='ascii.csv', overwrite=True)

def planet_cube(imgsize, res, planetlist, epoch=0.0*u.year, inc_obs=0.0*u.deg):
    '''
    Parameters
    ----------
    
    imgsize: int
        Size of image (assumed square)
    res: float
        Size of pixel in AU
    planetlist: list
        List of planet objects (see example below)
    dist_pc: float
        Distance to the system in pc
    epoch: float
        Epoch in years (default 0.0, this parameter is redundant with the mean anomaly parameter of a planet)
    inc_obs: float
        Inclination at which the system is seen. 
    
    Returns
    -------
    
    out: dict
        Dictionary with the system's image, the phase function
    '''
    img = np.zeros((imgsize,imgsize))
    c_img = imgsize//2
    
    # vector to the star
    plnt_obs_vec = np.array([0.,0., planetlist[0].dist.to(u.AU).value]).T

    phasefunclist = []
    contrastlist = []
    coordlist = []
    delxlist = []
    delylist = []
    
    for planet in planetlist:
        posvel = np.array(cartesian(a=planet.a.to(u.AU).value,
                                    ecc=planet.ecc, 
                                    incl=planet.inc.to(u.radian).value, 
                                    longnode=planet.longnode.to(u.radian).value, 
                                    argperi=planet.argperi.to(u.radian).value, 
                                    meananom=0,
                                    mstar=planet.mstar, mplan=planet.mplan,
                                    tperi=planet.tperi.to(u.year),
                                    epoch=epoch.to(u.year),
                                    inc_obs=inc_obs.to(u.radian).value))
        pl_r = np.sqrt(posvel[0]**2+posvel[1]**2+posvel[2]**2) * u.AU
        cos_obs = np.dot(plnt_obs_vec, posvel[:3] / (pl_r * np.sqrt(np.sum(plnt_obs_vec**2))))
        # Beta is the planet phase, which is 180 degrees out of phase from the scattering angle.
        beta = pi - np.arccos(cos_obs)
        # Lambert phase function
        lambert_pf = (sin(beta) + (pi - beta) * cos(beta)) / pi
        phasefunclist.append(lambert_pf)
        delxlist.append( np.arctan2(posvel[0]*u.AU, planet.dist).to(u.milliarcsecond) )
        delylist.append( np.arctan2(posvel[1]*u.AU, planet.dist).to(u.milliarcsecond) )
    
        plpix = np.zeros(2)
        plpix[0] = np.round(posvel[0]*u.AU/res)+c_img
        plpix[1] = np.round(posvel[1]*u.AU/res)+c_img
        coords = (int(plpix[0]),int(plpix[1]))
        coordlist.append(coords)
        if coords[0]<img.shape[0] and coords[0]>=0 and coords[1]<img.shape[1] and coords[1]>=0:
            contrast = lambert_pf*planet.albedo_vals[0]*(planet.radius.to(u.AU)/pl_r)**2
            img[coords] += contrast
            contrastlist.append(contrast)
    return {'img':img,
            'phasefunclist':phasefunclist,
            'contrastlist':contrastlist,
            'coordlist':coordlist,
            'delxlist':delxlist, 'delylist':delylist,
            'planetlist':planetlist}

def get_hires_psf_at_xy(offax_psfs, offax_offsets_as,
                        inner_offax_psfs, inner_offax_offsets_as,
                        pixscale_as, delx_as, dely_as, cx,
                        roll_angle=26.):
    '''
    Subroutine to shift and rotate off-axis PSF model
    to arbitrary array position
    '''

    r_as = np.sqrt(delx_as**2 + dely_as**2)
    theta = np.rad2deg(np.arctan2(dely_as, delx_as))
    
    if r_as <= np.max(inner_offax_offsets_as): # Use fine offset off-axis PSF model
        oi = np.argmin(np.abs(r_as - inner_offax_offsets_as)) # index of closest radial offset
        dr_as = r_as - inner_offax_offsets_as[oi] # radial shift needed in arcseconds
        dr_p = dr_as / pixscale_as # radial shift needed in pixels
        shift_offset_psf = scipy.ndimage.interpolation.shift(inner_offax_psfs[oi], (0, dr_p),
                                                             order = 1, prefilter=False,
                                                             mode='constant', cval=0)
    else: # Use public off-axis PSF model
        oi = np.argmin(np.abs(r_as - offax_offsets_as)) # index of closest radial offset
        dr_as = r_as - offax_offsets_as[oi] # radial shift needed in arcseconds
        dr_p = dr_as / pixscale_as # radial shift needed in pixels    
        shift_offset_psf = scipy.ndimage.interpolation.shift(offax_psfs[oi], (0, dr_p),
                                                             order = 1, prefilter=False,
                                                             mode='constant', cval=0)

    rot_shift_offset_psf = skimage.transform.rotate(shift_offset_psf,
                                                    angle = -theta,
                                                    order = 1, resize = False,
                                                    center=(cx, cx))
            
    rot_shift_offset_psf_roll = skimage.transform.rotate(shift_offset_psf,
                                                         angle = -(theta + roll_angle),
                                                         order = 1, resize = False,
                                                         center=(cx, cx))
            
    return rot_shift_offset_psf, rot_shift_offset_psf_roll

def xy_to_psf_odd(x, y, quad_cube):  # for odd image array width
    cx = quad_cube.shape[-1] // 2
    hw = (quad_cube.shape[-1] // 2 + 1)
    if x >= cx and y >= cx:  # in first quadrant
        s = (y - cx) * hw + (x - cx)
        return quad_cube[s]
    elif x < cx and y >= cx:  # second quadrant
        s = (y - cx) * hw + (cx - x)
        return quad_cube[s, :, ::-1]
    elif x < cx and y < cx:  # third quadrant
        s = (cx - y) * hw + (cx - x)
        return quad_cube[s, ::-1, ::-1]
    else:                 # fourth quadrant
        s = (cx - y) * hw + (x - cx)
    return quad_cube[s, ::-1, :]

def xy_to_psf_even(x, y, quad_cube):  # for even image array width
    cx = quad_cube.shape[-1] // 2 - 0.5
    cxi = quad_cube.shape[-1] // 2
    hw = quad_cube.shape[-1] // 2
    if x > cx and y > cx:  # in first quadrant
        s = (y - cxi) * hw + (x - cxi)
        return quad_cube[s]
    elif x < cx and y > cx:  # second quadrant
        s = (y - cxi) * hw + (cxi - 1 - x)
        return quad_cube[s, :, ::-1]
    elif x < cx and y < cx:  # third quadrant
        s = (cxi - 1 - y) * hw + (cxi - 1 - x)
        return quad_cube[s, ::-1, ::-1]
    else:                 # fourth quadrant
        s = (cxi - 1 - y) * hw + (x - cxi)
        return quad_cube[s, ::-1, :]

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
    # (1*u.AU) + (1.5 * 10**6) * u.kilometer
    d_L2Sun = (1.496e11 + 1.492e9) * u.meter
    d_star = (1 * u.AU / np.sin(plx)).to(u.pc)
    x_tel = d_L2Sun * np.cos( phi_orb )
    y_tel = d_L2Sun * np.sin( phi_orb )
    z_tel = 0 # Assuming L2 orbit on Ecliptical plane

    # Vector joining the star and the telescope
    x_star_tel = d_star * np.cos( elat ) * np.cos( elon ) - x_tel
    y_star_tel = d_star * np.cos( elat ) * np.sin( elon ) - y_tel
    z_star_tel = d_star * np.sin( elat )

    # Normalized position vector
    d_star_tel = sqrt( x_star_tel * x_star_tel 
                     + y_star_tel * y_star_tel 
                     + z_star_tel * z_star_tel )

    x_nrm = x_star_tel / d_star_tel
    y_nrm = y_star_tel / d_star_tel
    z_nrm = z_star_tel / d_star_tel

    # Deriving the new Ecliptic coordinates
    elat_wplx = np.arcsin( z_nrm )
    elon_wplx = np.arctan2( y_nrm, x_nrm )
    #print("elon, elat = {:.4f}, {:.4f}".format(np.rad2deg(elon_wplx), np.rad2deg(elat_wplx)))
    
    coord_ec_wplx = astropy.coordinates.BarycentricTrueEcliptic(
            lon = elon_wplx, lat = elat_wplx)

    coord_icrs_wplx = coord_ec_wplx.transform_to(
            astropy.coordinates.ICRS)
    
    offset_RA, offset_Dec = coord_icrs.spherical_offsets_to(coord_icrs_wplx)
    
    return offset_RA.to(u.mas), offset_Dec.to(u.mas)

def Jy_to_photons(cube_Jy, wavel):
    '''
    Parameters
    ----------
        cube_Jy: 3D haystacks cube in Jy
        wavel: 1D array of wavelengths in astropy length units
    Returns
    -------
        haycube: 3D haystacks cube in photons/m2/um/s
   
    '''
    if isinstance(wavel, u.quantity.Quantity):
        lamlist = wavel.to(u.nm)
    else: # Assume input wavelengths are in microns
        lamlist = (wavel * u.um).to(u.nm)
    lamcube = c.c/lamlist[:,np.newaxis,np.newaxis]**2
    
    haycube = cube_Jy * u.Jansky 
    
    haycube = haycube.to(u.Watt/u.m**2/u.Hertz)
    haycube *= lamcube
    haycube = haycube.to(u.W/u.m**2/u.nm)
    
    # photon energy
    Eph = (c.h*c.c/lamlist[:,np.newaxis,np.newaxis]/u.photon).to(u.J/u.photon)
    haycube = (haycube/Eph).to(u.photon/u.s/u.m**2/u.um)
    
    return haycube

def resample_image_array(img, img_pixscale, img_xcoord, img_ycoord,
                         det_pixscale, det_width, binfac=9, conserve='sum'):
    """
    Resample an image plane array to the resolution of a detector, cropped to a 
    specified width.

    Parameters
    ----------
    img : numpy.ndarray
        Input image plane array, can be complex-valued
    img_pixscale : float
        Pixel scale of image plane array, in sky angle units
    img_xcoord : astropy.units.quantity.Quantity of dimension angle
        Horizontal array coordinate vector w.r.t. image center
    img_ycoord : astropy.units.quantity.Quantity of dimension angle
        Vertical array coordinate vector w.r.t. image center
    det_pixscale: astropy.units.quantity.Quantity of dimension angle
        Pixel scale of detector, in sky angle units
    det_width: astropy.units.quantity.Quantity of dimension angle
        Detector array width, in sky angle units
    binfac: numpy.int
        Whole number ratio of upsampled resolution to 
        binned detector resolution, default is 10, minimum is 1
    conserve: string
        'sum' (default) - Conserve the sum of input array values integrated 
                          over the detector array (for quantities proportional
                          to intensity).
        'sumofsq' - Conserve the sum of the square of input array values 
                    integrated over the detector array (for quantities 
                    proportional to electric field).

    Returns
    -------
    det : numpy.ndarray
        Square detector array of odd width, 
        with image centered on the middle pixel
    det_xcoord : astropy.units.quantity.Quantity of dimension angle
        Horizontal array coordinate vector w.r.t. center
    det_ycoord : astropy.units.quantity.Quantity of dimension angle
        Vertical array coordinate vector w.r.t. center
    """

    # Set boundaries of the image region cropped to detector array
    crop_col_beg = np.greater_equal(img_xcoord, -det_width / 2.).nonzero()[0].min()
    crop_col_end = np.less_equal(img_xcoord, det_width / 2.).nonzero()[0].max()
    crop_row_beg = np.greater_equal(img_ycoord, -det_width / 2.).nonzero()[0].min()
    crop_row_end = np.less_equal(img_ycoord, det_width / 2.).nonzero()[0].max()
    
    crop_img = img[crop_row_beg:crop_row_end+1, crop_col_beg:crop_col_end+1]
    crop_xcoord = img_xcoord[crop_col_beg:crop_col_end+1]
    crop_ycoord = img_ycoord[crop_row_beg:crop_row_end+1]
    
    # Upsample the input image array in preparation for detector pixel integration
    upsamp_scalefac = binfac * img_pixscale.value / det_pixscale.value
    upsamp_pixscale = img_pixscale / upsamp_scalefac
    
    N_upsamp_lef = int(-crop_xcoord[0] // upsamp_pixscale)
    N_upsamp_rig = int(crop_xcoord[-1] // upsamp_pixscale)
    N_upsamp_bot = int(-crop_ycoord[0] // upsamp_pixscale)
    N_upsamp_top = int(crop_ycoord[-1] // upsamp_pixscale)
    upsamp_xcoord = np.linspace(-N_upsamp_lef + 0.5,
                                 N_upsamp_rig - 0.5,
                                 N_upsamp_lef + N_upsamp_rig) * upsamp_pixscale
    upsamp_ycoord = np.linspace(-N_upsamp_bot + 0.5,
                                 N_upsamp_top - 0.5,
                                 N_upsamp_bot + N_upsamp_top) * upsamp_pixscale

    realpart_interp_func = scipy.interpolate.interp2d(
            crop_xcoord.value, crop_ycoord.value, crop_img.real, kind="linear")                              
    upsamp_realpart = realpart_interp_func(upsamp_xcoord.value,
                                           upsamp_ycoord.value)
    if np.iscomplexobj(crop_img):
        imagpart_interp_func = scipy.interpolate.interp2d(
                crop_xcoord.value, crop_ycoord.value, crop_img.imag, kind="linear")
        upsamp_imagpart = imagpart_interp_func(upsamp_xcoord.value,
                                               upsamp_ycoord.value)
        upsamp_array = upsamp_realpart + 1j*upsamp_imagpart
    else:
        upsamp_array = upsamp_realpart
        
    if conserve == 'sum':
        conserve_fac = 1.0 / upsamp_scalefac**2
    elif conserve == 'sumofsq':
        conserve_fac = np.sqrt( np.sum(np.abs(crop_img)**2) 
                                / np.sum(np.abs(upsamp_array)**2) )
    upsamp_array = upsamp_array * conserve_fac
    
    # Trim the upsampled array dimensions to an {odd number} x {odd number} array
    # of {binfac} x {binfac} detector pixel tiles.
    num_pix = upsamp_array.shape[0] // binfac
    if num_pix % 2 == 1:
        num_odd_pix = num_pix
    else:
        num_odd_pix = num_pix - 1
    trim_rows = upsamp_array.shape[0] % (binfac * num_odd_pix)
    trim_cols = upsamp_array.shape[1] % (binfac * num_odd_pix)
    trim_top = trim_rows // 2
    trim_bot = trim_rows - trim_top
    trim_rig = trim_cols // 2
    trim_lef = trim_cols - trim_rig
    np.testing.assert_equal(
            trim_lef, trim_rig,
            err_msg = "Combination of upsampled array size ({:d}), bin factor ({:d}x) cannot preserve center of input array.".format(
            upsamp_array.shape[0], binfac))
    
    trim_upsamp_img = upsamp_array[trim_bot:(upsamp_array.shape[0]-trim_top),
                                   trim_lef:(upsamp_array.shape[1]-trim_rig)]
    trim_upsamp_xcoord = upsamp_xcoord[trim_lef:(upsamp_array.shape[1]-trim_rig)]
    trim_upsamp_ycoord = upsamp_ycoord[trim_bot:(upsamp_array.shape[1]-trim_top)]
    
    # Pixel-integrated downsample by whole number bin factor
    det_img = trim_upsamp_img.reshape(
            trim_upsamp_img.shape[0] // binfac, binfac,
            trim_upsamp_img.shape[1] // binfac, binfac).sum(axis = 1).sum(axis = 2)
    if binfac % 2 == 1:
        det_xcoord = trim_upsamp_xcoord[binfac // 2 :: binfac]
        det_ycoord = trim_upsamp_ycoord[binfac // 2 :: binfac]
    else:
        det_xcoord = trim_upsamp_xcoord[binfac // 2 :: binfac] - upsamp_pixscale / 2
        det_ycoord = trim_upsamp_xcoord[binfac // 2 :: binfac] - upsamp_pixscale / 2

    if conserve == 'sum':
        conserve_fac = 1.
    elif conserve == 'sumofsq':
        conserve_fac = np.sqrt( np.sum(np.abs(trim_upsamp_img)** 2)
                                / np.sum(np.abs(det_img)**2) )
    det_img = det_img * conserve_fac
    
    return det_img, det_xcoord, det_ycoord

def normintens_to_countrate(ni_map, star_photrate, collecting_area,
                            coron_thrupt_peakpixel, optloss = 0.5,
                            qe = 0.9, ifsloss = 1.0):
    """
    Convert a normalized intensity array to an array of photoelectron count 
    rates, based on parameters for stellar irradiance, telescope collecting 
    area, coronagragh mask throughput, optical loss factor, and detector 
    quantum efficiency.
    
    Parameters
    ----------
    ni_map : numpy.ndarray
        Input array of normalized intensity values, normalized to the detector
        pixel integrated peak of an unocculted PSF.
    star_photrate : astropy.units.quantity.Quantity, of physical dimensions 
        energy / time / area. This is the bandpass-integrated irradiance of 
        the target star; suggested units are photons / second / meter^2
    collecting_area : astropy.units.quantity.Quantity, of physical dimensions 
        area. This is the collecting area of the obscured telelescope aperture.
    coron_thrupt_peakpixel: numpy.float
        Throughput from telescope aperture to the detector-pixel-inegrated peak 
        of an unocculted PSF.
    optloss : numpy.float
        Cumulative attenuation due to mirror reflections and transmissive optics
    qe : numpy.float
        Detector quantum efficiency; can also include losses due to readout 
        efficiencies and cosmic rays
    ifsloss : numpy.float
        signal attenuation factor due to re-imaging onto spectrograph detector;
        for no IFS leave at the default value of 1.0.
        
    Returns
    -------
    countrate_map : numpy.ndarray of astropy.units.quantity.Quantity
        Photoelectron count rate map, units photons / sec
    """
    countrate_map = (ni_map * star_photrate * collecting_area * 
                     coron_thrupt_peakpixel * optloss * qe * ifsloss)
    return countrate_map

def get_detector_exposure(countrate_map, total_inttime, read_inttime,
                          dark_cur, read_noise, return_read_cube=False):
    """
    Generate a simple simulated detector exposure.

    Parameters
    ----------
    countrate_map : astropy.units.quantity.Quantity numpy.ndarray
        2-D array of detector photoelectron count rates, units photons / sec
    total_inttime : astropy.units.quantity.Quantity 
        Total integration time of exposure
    read_inttime : astropy.units.quantity.Quantity 
        Individual detector read integration time
    dark_cur : astropy.units.quantity.Quantity 
        Dark current, units photons / sec
    read_noise : astropy.units.quantity.Quantity 
        Read noise, units photons
    return_read_cube : bool
        If True, return a cube containing the sequence of simulated detector 
        integrations. Default value is False.

    Returns
    -------
    detector_exposure : astropy.units.quantity.Quantity numpy.ndarray
        2-D array detector exposure, units photons
    N_reads : numpy.int
        Number of detector integrations in read sequence
    read_cube : astropy.units.quantity.Quantity numpy.ndarray
        (Optional) Cube containing detector integration sequence,
        units photons
     """
    
    N_reads = int(total_inttime.to(read_inttime.unit) // read_inttime)
    read_expectation = ((countrate_map + dark_cur) * read_inttime).to(u.photon)
    
    exposure_array = np.zeros(countrate_map.shape) * u.photon
    if return_read_cube:
        read_cube = np.zeros((N_reads, countrate_map.shape[0],
                              countrate_map.shape[1])) * u.photon
    
    for ii in range(N_reads):
        read_array = ( np.random.poisson(read_expectation.value)
                     + np.random.normal(scale = read_noise.value,
                                        size = read_expectation.shape)) 
        exposure_array += read_array * u.photon
        
        if return_read_cube:
            read_cube[ii] = read_array
    
    if return_read_cube:
        return exposure_array, N_reads, read_cube
    else:
        return exposure_array, N_reads

def bpgs_list(spectype=None,verbose=False):
    '''
    Returns pandas dataframe with the list of the files, the name and the type of the star

    '''
    #fname = pkg_resources.resource_filename('crispy', 'Inputs') + '/bpgs/bpgs_readme.csv'
    fname = pkg_resources.resource_filename('scene_utils', 'bpgs/bpgs_readme.csv')
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
