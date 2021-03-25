from numpy import sin, cos, sqrt, pi
import numpy as np
import scipy.interpolate
import astropy.units as u
import astropy.constants as c
import astropy.table
import pandas

class Planet:
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
            self.albedo_wavelens = [575 * u.nanometer]
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
            
        if self.P == None and self.a == None:
            raise ValueError('User must define either a period or sma for the planet.')        
        elif self.P == None:
            self.P = np.sqrt( 4 * np.pi**2 / (c.G * (self.mstar + self.mplan) ) * \
                              (self.a)**3 ).to(u.year)
        elif self.a == None:
            self.a = np.cbrt((c.G * (self.mstar + self.mplan) * self.P**2) \
                             / (4 * np.pi**2) ).to(u.AU)
        if self.argperi == None:
            self.argperi = 0 * u.deg

        if self.tperi == None:
            self.tperi = 0 * u.year
            
    def compute_ephem(self, tarray = None, tbeg = 0 * u.year, tend = None, tstep = None):
        if tarray == None:
            ts = np.arange(tbeg.to(tend.unit).value,
                           tend.value + tstep.to(tend.unit).value,
                           tstep.to(tend.unit).value) * tend.unit
        else:
            ts = tarray
        
        Nt = len(ts)
        delx = np.zeros(Nt) * u.milliarcsecond
        dely = np.zeros(Nt) * u.milliarcsecond
        beta = np.zeros(Nt)
        phasefunc = np.zeros(Nt)
        orad = np.zeros(Nt) * u.AU
        
        dist = self.dist.to(u.AU)
        
        for tt, epoch in enumerate(ts):      
            posvel = np.array(cartesian(a = self.a.to(u.AU).value,
                                        ecc = self.ecc, 
                                        incl = self.inc.to(u.radian).value, 
                                        longnode = self.longnode.to(u.radian).value, 
                                        argperi = self.argperi.to(u.radian).value, 
                                        meananom = 0,
                                        mstar = self.mstar, 
                                        mplan = self.mplan,
                                        tperi = self.tperi.to(u.year),
                                        epoch = epoch.to(u.year),
                                        inc_obs = 0))
            pl_r = np.sqrt(posvel[0]**2 + posvel[1]**2 + posvel[2]**2) * u.AU
            # The angle between the planet-observer and star-planet vectors
            # is obtained from the vector law of cosines:
            plnt_obs_vec = np.array([-posvel[0], -posvel[1], dist.value - posvel[2]])
            plnt_obs_dist = np.sqrt(np.sum(plnt_obs_vec**2)) * u.AU
            cos_obs = np.dot(plnt_obs_vec, posvel[:3]) / (pl_r.value * plnt_obs_dist.value)
            # Beta is the planet phase angle, which is the supplement of the angle between the 
            # star-planet and planet-observer vectors.
            beta[tt] = pi - np.arccos(cos_obs)
            # Lambert phase function
            lambert_pf = (sin(beta[tt]) + (pi - beta[tt]) * cos(beta[tt])) / pi
            phasefunc[tt] = lambert_pf
            orad[tt] = pl_r
            delx[tt] = np.arctan2(posvel[0]*u.AU, self.dist).to(u.milliarcsecond)
            dely[tt] = np.arctan2(posvel[1]*u.AU, self.dist).to(u.milliarcsecond)
        
        return ts, delx, dely, beta, phasefunc, orad
            
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
    # Modified July 2020 to correct coordinate convention for angles of orbital elements.
    #
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
    foo = sqrt(1.0 - ecc * ecc)
    meanmotion = 2 * pi / period.value
    x = a * (cosE - ecc)
    y = foo * (a * sinE)
    z = np.zeros_like(y)
    denom = 1. / (1.0 - ecc * cosE)
    xd = (-a * meanmotion * sinE) * denom
    yd = foo * (a * meanmotion * cosE * denom)
    zd = np.zeros_like(yd)

    # rotate by argument of periastron in orbit plane
    cosw = cos(argperi - pi)
    sinw = sin(argperi - pi)
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
    cosnode = cos(longnode - pi / 2)
    sinnode = sin(longnode - pi / 2)
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

def write_ephem_table(planet, table_fname, tarray = None, tbeg = 0 * u.year, tend = None, tstep = None):
    tseries, delx, dely, beta, phasefunc, orad = planet.compute_ephem(tarray, tbeg, tend, tstep)
   
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
    img = np.zeros((imgsize, imgsize))
    c_img = imgsize // 2
    
    # vector to the star
    dist = planetlist[0].dist.to(u.AU)

    phasefunclist = []
    contrastlist = []
    coordlist = []
    delxlist = []
    delylist = []
    
    for planet in planetlist:
        posvel = np.array(cartesian(a = planet.a.to(u.AU).value,
                                    ecc = planet.ecc, 
                                    incl = planet.inc.to(u.radian).value, 
                                    longnode = planet.longnode.to(u.radian).value, 
                                    argperi = planet.argperi.to(u.radian).value, 
                                    meananom = 0,
                                    mstar = planet.mstar, mplan=planet.mplan,
                                    tperi = planet.tperi.to(u.year),
                                    epoch = epoch.to(u.year),
                                    inc_obs = inc_obs.to(u.radian).value))
        pl_r = np.sqrt(posvel[0]**2 + posvel[1]**2 + posvel[2]**2) * u.AU
        # The angle between the planet-observer and star-planet vectors
        # is obtained from the vector law of cosines:
        plnt_obs_vec = np.array([-posvel[0], -posvel[1], dist.value - posvel[2]])
        plnt_obs_dist = np.sqrt(np.sum(plnt_obs_vec**2)) * u.AU
        cos_obs = np.dot(plnt_obs_vec, posvel[:3]) / (pl_r.value * plnt_obs_dist.value)
        # Beta is the planet phase angle, which is the supplement of the angle between the 
        # star-planet and planet-observer vectors.
        beta = pi - np.arccos(cos_obs)
        # Lambert phase function
        lambert_pf = (sin(beta) + (pi - beta) * cos(beta)) / pi
        phasefunclist.append(lambert_pf)
        delxlist.append( np.arctan2(posvel[0]*u.AU, planet.dist).to(u.milliarcsecond) )
        delylist.append( np.arctan2(posvel[1]*u.AU, planet.dist).to(u.milliarcsecond) )
    
        plpix = np.zeros(2)
        plpix[0] = np.round(posvel[1]*u.AU / res) + c_img
        plpix[1] = np.round(posvel[0]*u.AU / res) + c_img
        coords = (int(plpix[0]), int(plpix[1]))
        coordlist.append(coords)
        if (coords[0] < img.shape[0] and coords[0] >= 0 and 
            coords[1] < img.shape[1] and coords[1] >= 0):
            contrast = (lambert_pf * planet.albedo_vals[0] * 
                        (planet.radius.to(u.AU) / pl_r)**2)
            img[coords] += contrast
            contrastlist.append(contrast)
    return {'img':img,
            'phasefunclist':phasefunclist,
            'contrastlist':contrastlist,
            'coordlist':coordlist,
            'delxlist':delxlist, 'delylist':delylist,
            'planetlist':planetlist}

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
