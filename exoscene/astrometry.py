import numpy as np

import astropy.coordinates
import astropy.units as u
import astropy.constants as c

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
