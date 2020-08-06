import numpy as np
import numpy.testing
import scipy.ndimage
import scipy.interpolate
import skimage.transform

import astropy.units as u
import astropy.constants as c

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

def get_hires_psf_at_xy_os6(offax_psfs, offax_offsets_as,
                            inner_offax_psfs, inner_offax_offsets_as,
                            pixscale_as, delx_as, dely_as, cx,
                            roll_angle = 26.):
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

def get_hires_psf_at_xy_os9(offax_psfs, offsets_as, angles,
                            pixscale_as, delx_as, dely_as, cx):
    '''
    Subroutine to shift and rotate off-axis PSF model
    to arbitrary array position
    '''
    r_as = np.sqrt(delx_as**2 + dely_as**2)
    theta = np.rad2deg(np.arctan2(dely_as, delx_as))

    oi = np.argmin(np.abs(r_as - offsets_as))
    dr_as = r_as - offsets_as[oi] # radial shift needed in arcseconds
    dr_p = dr_as / pixscale_as # radial shift needed in pixels    

    if theta >= 0 and theta < 90: # in first quadrant
        theta_q = theta
    elif theta >= 90 and theta < 180: # second quadrant
        theta_q = 180 - theta
    elif theta >= 180 and theta < 270: # third quadrant
        theta_q = theta - 180
    else: # fourth quadrant
        theta_q = 360 - theta
    ai = np.argmin(np.abs(theta_q - angles))
    dtheta = theta_q - angles[ai]
    dx_p = dr_p * np.cos(np.deg2rad(theta_q))
    dy_p = dr_p * np.sin(np.deg2rad(theta_q))

    rot_psf = skimage.transform.rotate(
            offax_psfs[ai, oi],
            angle = -dtheta,
            order = 1, resize = False,
            center = (cx, cx))
    shift_rot_psf = scipy.ndimage.interpolation.shift(
            rot_psf, (dy_p, dx_p),
            order = 1, prefilter = False,
            mode = 'constant', cval = 0)

    if theta >= 0 and theta < 90: # in first quadrant
        reflect_psf = shift_rot_psf
    elif theta >= 90 and theta < 180: # second quadrant
        reflect_psf = shift_rot_psf[:, ::-1]
    elif theta >= 180 and theta < 270: # third quadrant
        reflect_psf = shift_rot_psf[::-1, ::-1]
    else: # fourth quadrant
        reflect_psf = shift_rot_psf[::-1, :]

    return reflect_psf

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
