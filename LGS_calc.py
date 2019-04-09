#########################################
# Author: Jennifer Lumbres (contact: jlumbres@optics.arizona.edu)
# Last edit: 2018/07/01
# functions for LGS

#########################################
# PACKAGE IMPORT
#########################################
#load modules
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from astropy import units as u
from astropy.io import fits
from  matplotlib.colors import LogNorm
import scipy.ndimage
import h5py
#POPPY
import poppy
from poppy.poppy_core import PlaneType
# MagAO-X functions
import magaoxFunctions as mf

#########################################
# FUNCTION DEFINITIONS
#########################################

# Function: surfFITS
# Description: Initiates a FITS file to add to optical system.
# Input Parameters:
#   file_loc    - string    - path location of FITS file
#   optic_type  - string    - Declare if the file is OPD or Transmission type ('opd' or 'trans')
#   opdunit     - string    - OPD units of FITS file. For some reason, BUNIT header card gives errors.
#   name        - string    - descriptive name for optic. Useful for phase description.
# Output Parameters:
#   optic_surf  - FITSOpticalElement    - Returns FITSOpticalElement to use as surface mapping file.
# Sequence of Initializing:
#   - Call in FITS file
#   - Typecast FITS data to float type (workaround to get POPPY to accept FITS data)
#   - Determine optic type to choose how to build FITSOpticalElement
#   - Return FITSOpticalElement object
def surfFITS(file_loc, optic_type, opdunit, name):
    optic_fits = fits.open(file_loc)
    optic_fits[0].data = np.float_(optic_fits[0].data) # typecasting for POPPY workaround
    if optic_type == 'opd':
        optic_surf = poppy.FITSOpticalElement(name = name, opd=optic_fits, opdunits = opdunit)
    else:
        optic_surf = poppy.FITSOpticalElement(name = name, transmission=optic_fits)
    return optic_surf

# Function: writeOPDfile
# Description: Writes OPD mask to FITS file, WILL OVERRIDE OLD FILE IF fileloc IS REUSED
# Input Parameters:
#    opd_surf_data   - OPD surface data
#    pixelscl        - pixel scale on m/pix
#    fileloc         - file string location for vAPP OPD mask FITS file
# Output:
#    none (just does the thing)
def writeOPDfile(opd_surf_data, pixelscl, fileloc):
    writeOPD = fits.PrimaryHDU(data=opd_surf_data)
    writeOPD.header.set('PUPLSCAL', pixelscl)
    writeOPD.header.comments['PUPLSCAL'] = 'pixel scale [m/pix]'
    writeOPD.header.set('BUNIT', 'meters')
    writeOPD.header.comments['BUNIT'] = 'opd units'
    writeOPD.writeto(fileloc, clobber=True)

# Function: writeTRANSfile
# Description: Writes transmission mask to FITS file, WILL OVERRIDE OLD FILE IF fileloc IS REUSED
# Input Parameters:
#    trans_data      - transmission data
#    pixelscl        - pixel scale on m/pix
#    fileloc         - file string location for vAPP transmission mask FITS file
# Output:
#    none (just does the thing)
def writeTRANSfile(trans_data, pixelscl, fileloc):
    writetrans = fits.PrimaryHDU(data=trans_data)
    writetrans.header.set('PUPLSCAL', pixelscl)
    writetrans.header.comments['PUPLSCAL'] = 'pixel scale [m/pix]'
    writetrans.writeto(fileloc, clobber=True)

# Function: makeRxCSV
# Desription: Get the system prescription from CSV file
# FYI: This has some hardcoded numbers in it, but just follow the specs on the CSV file.
# Input parameters:
#    csv_file   - CSV file location
# Output parameters:
#    sys_rx     - system prescription into a workable array format
def makeRxCSV(csv_file):
    sys_rx=np.genfromtxt(csv_file, delimiter=',', dtype="i2,U19,U10,f8,f8,f8,U90,U90,U10,U10,f8,U10,", skip_header=13,names=True)
    print('CSV file name: %s' % csv_file)
    print('The names of the headers are:')
    print(sys_rx.dtype.names)
    return sys_rx


# Function: sellmeierIndexRefraction
# Description: Calculates the index of refraction for N-BK7 using the Sellmeier Equation, as index of refraction depends on wavelength
# Input Parameters:
#   wavelength      - float     - wavelength of light [meters]
# Ouput Parameters:
#   refIndex        - float     - Index of refraction [unitless]
def sellmeierIndexRefraction(wavelength):
    # convert wavelength into microns, as dictated by Sellmeier Equation
    wavelength_um = wavelength.value * 10**6 # converts meters to microns
    refIndex = np.sqrt(1 + ( 1.03961212*(wavelength_um**2) / ( (wavelength_um**2) - 0.00600069867 )) + ( 0.231792344*(wavelength_um**2) / ( (wavelength_um**2) - 0.0200179144 )) + ( 1.01046945*(wavelength_um**2) / ( (wavelength_um**2) - 103.560653 )) )
    return refIndex


# Function: calcDefocusWFE
# Description: Calculates Peak-to-Valley wavefront error of LGS based on telescope diameter and LGS range.
# Input Parameters:
#   D_telescope     - float     - Diameter of telescope, [meters]
#   LGS_range       - integer   - distance of LGS from telescope, units of x10,000 KM
# Ouput Parameters:
#   LGS_defocus_WFE - float     - Wavefront error value, [meters]
def calcDefocusWFE(D_telescope, LGS_range):
    defocus_WFE = LGS_range - np.sqrt(LGS_range**2 - (D_telescope/2)**2)
    return defocus_WFE

# Function: calcDefocusWaves
# Description: Calculates number of defocus waves based on telescope diameter, LGS range and wavelength
# Input Parameters:
#   D_telescope     - integer   - Diameter of telescope, [meters]
#   LGS_range       - integer   - distance of LGS from telescope, units of x10,000 KM
#   LGS_wavelength  - integer   - LGS wavelength, [meters]
# Output Parameters:
#   defocus_waves   - float     - number of waves of defocus, [unitless], a020
def calcDefocusWaves(D_telescope, LGS_range, LGS_wavelength):
    defocus_waves = calcDefocusWFE(D_telescope, LGS_range)/LGS_wavelength
    return defocus_waves

# Function: calcDefocusShift
# Description: Calculates number of defocus waves based on telescope diameter, LGS range and wavelength
# Input Parameters:
#   f_num               - integer   - F/# of testbed = f_oap / D_beam [unitless]
#   testbed_wavelength  - integer   - testbed LGS wavelength [meters]
#   defocus_waves       - float     - number of defocus waves produced by LGS cubesat
# Output Parameters:
#   defocus_shift       - float     - how far to move source closer to f_oap [meters]
def calcDefocusShift(f_num, testbed_wavelength, defocus_waves):
    dz = 8 * (f_num**2) * defocus_waves * testbed_wavelength
    return dz

# Function: calcTiltWaves
# Description: Calculates number of tilt waves based on LGS separation, telescope diameter, and wavelength
# Input Parameters:
#   separation      - float     - LGS separation [arcsec]
#   D_telescope     - integer   - Diameter of telescope, [meters]
#   LGS_wavelength  - integer   - LGS wavelength, [meters]
# Output Parameters:
#   tilt_waves      - float     - number of waves of tilt, [unitless]
def calcTiltWaves(separation, D_telescope, LGS_wavelength):
    arcsec2rad = 206265 * u.arcsec / u.rad
    tilt_waves = np.tan(separation/arcsec2rad)*(D_telescope/2)/LGS_wavelength
    return tilt_waves


# Function: calcPlateScale
# Description: Calculates plate scale of the testbed setup
# Input Parameters:
#   beam_diam   - float     - diameter of beam [m]
#   f_num       - integer   - focal length / EP diameter [unitless]
# Output Parameters:
#   plate_scale - float     - platescale of testbed [arcsec/m]
def calcPlateScale(beam_diam, f_num):
    arcsec2rad = 206265 * u.arcsec
    plate_scale = arcsec2rad/(beam_diam*f_num)
    return plate_scale

# Function: calcZernikeNormCoeff
# Description: Calculates Zernike normalization coefficient
# Source material: http://www.telescope-optics.net/zernike_aberrations.htm
# Input Parameters:
#   Zernike_name    - string    - name of Zernike
# Output Parameters:
#   norm_coeff  - normalization coefficient based on name of Zernike
def calcZernikeNormCoeff(Zernike_name):
    if Zernike_name == 'piston':
        norm_coeff = 1;
    else:
        if Zernike_name == 'tip':
            n = 1; m = 1;
        elif Zernike_name == 'tilt':
            n = 1; m = -1; 
        elif Zernike_name == 'defocus':
            n = 2; m = 0;
        elif Zernike_name == 'oastig':
            n = 2; m = 2;
        elif Zernike_name == 'vastig':
            n = 2; m = -2;
        elif Zernike_name == 'vcoma':
            n = 3; m = -1;
        elif Zernike_name == 'hcoma':
            n = 3; m = 1;
        elif Zernike_name == 'vtrefoil':
            n = 3; m = -3;
        elif Zernike_name == 'otrefoil':
            n = 3; m = 3;
        elif Zernike_name == 'spherical':
            n = 4; m = 0;
        else:
            print('Bad Zernike name, please try again')
            n = 0;
            
        # normalization based on POPPY Documentation
        # Source: https://poppy-optics.readthedocs.io/en/stable/wfe.html#zernike-normalization
        if m==0:
            norm_coeff = np.sqrt(n+1);
        else:
            norm_coeff = np.sqrt(2)*np.sqrt(n+1);
    return norm_coeff;


# Function: ZWFEcoeff_LGS
# Description: Calculates the tilt and defocus components for ZWFE coefficients for LGS unit
# Input parameters:
#   space_parms   - dictionary of LGS cubesat parameters in space
#   wavelength    - wavelength of the LGS unit
#   irisAO_radius - radius of IrisAO segmented DM
# Output:
#   LGS_coeff_sequence - array sequence of ZWFE coefficients as LGS stand-in
def ZWFEcoeff_LGS(space_parms, wavelen, irisAO_radius):
    # calculate number of waves of error produced by LGS
    defocus_waves = calcDefocusWaves(space_parms['LUVOIR_diam'], space_parms['LGS_dist'], wavelen)
    tilt_waves = calcTiltWaves(space_parms['LGS_sep'], irisAO_radius*2, wavelen)
    
    # calculate Zernike coefficients
    # The 2 factor is to work in waves PV instead of center-to-peak
    tilt_coeff = tilt_waves * wavelen / (2 * calcZernikeNormCoeff('tilt'))
    defocus_coeff = defocus_waves * wavelen/ (2 * calcZernikeNormCoeff('defocus'))
    
    # coefficients are in Noll index order, starting from piston.
    LGS_coeff_sequence = [0, 0, tilt_coeff.value, defocus_coeff.value]
    
    return LGS_coeff_sequence

# Function: csvFresnel
# Description: Builds FresnelOpticalSystem from a prescription CSV file passed in
# Input parameters:
#    rx_csv      - system prescription
#    res         - resolution
#    oversamp    - oversampling convention used in PROPER
#    break_plane - plane to break building (so either focal plane or ZWFS image plane)
#    souce_ZWFE  - numerical array of WFE coefficients. All 0's if on-axis target star; LGS will have tilt and defocus components
#    irisAOstatus - boolean value determining if hexDM present. If not, then treats like a perfect flat mirror.
# Output:
#    sys_build   - FresnelOpticalSystem object with all optics built into it
def csvFresnel(rx_csv, samp, oversamp, axis, break_plane, source_ZWFE_coeff, irisAOmap, irisAOstatus):
    EP_radius=rx_csv['Radius_m'][2]*u.m # Element [2] is IrisAO because Element [0] is the diverging source, [1] is OAP1
    irisAO_radius=rx_csv['Radius_m'][6]*u.m
    
    sys_build = poppy.FresnelOpticalSystem(pupil_diameter=2*EP_radius, npix=samp, beam_ratio=oversamp)
    #sys_build = poppy.OpticalSystem(pupil_diameter=2*EP_radius, npix=samp, beam_ratio=oversamp)

    # Entrance Aperture
    sys_build.add_optic(poppy.CircularAperture(radius=EP_radius))
    
    # Apply if off-axis LGS is used
    if axis == 'LGS':
        src_aberr = poppy.ZernikeWFE(radius=irisAO_radius.value, coefficients = source_ZWFE_coeff);
        sys_build.add_optic(src_aberr)
    # if the on-axis target is used, then a source aberration will not be applied (hence on-axis and at infinity) 

    # Build LGS optical system from CSV file to the break plane
    for n_optic,optic in enumerate(rx_csv): # n_optic: count, optic: value
        #print('n_optic = ', n_optic)

        dz = optic['Distance_m'] * u.m # Propagation distance from the previous optic (n_optic-1)
        fl = optic['Focal_Length_m'] * u.m # Focal length of the current optic (n_optic)

        #print('Check PSD file for %s: %s' % (optic['Name'], optic['surf_PSD']))
        # if PSD file present
        if optic['surf_PSD_filename'] != 'none':
            # make a string insertion for the file location
            surf_file_loc = optic['surf_PSD_folder'] + optic['surf_PSD_filename'] + '.fits'
            # call surfFITS to send out surface map
            optic_surface = mf.surfFITS(file_loc = surf_file_loc, optic_type = optic['optic_type'], opdunit = optic['OPD_unit'], name = optic['Name']+' surface')
            # Add generated surface map to optical system
            sys_build.add_optic(optic_surface,distance=dz)

            if fl != 0: # powered optic with PSD file present
                sys_build.add_optic(poppy.QuadraticLens(fl,name=optic['Name'])) 
                # no distance; surface comes first
                sys_build.add_optic(poppy.CircularAperture(radius=optic['Radius_m']*u.m, name=optic['Name']+" aperture"))

            elif optic['Type'] != 'pupil': # non-powered optic but has PSD present that is NOT the pupil
                sys_build.add_optic(poppy.CircularAperture(radius=optic['Radius_m']*u.m, name=optic['Name']+" aperture"))

        # if no PSD file present (DM, focal plane, testing optical surface)
        else:
            #print('Enter no PSD file condition')
            # if powered optic is being tested
            if fl !=0: 
                #print('Enter powered optic condition')
                sys_build.add_optic(poppy.QuadraticLens(fl,name=optic['Name']), distance=dz)
                if n_optic > 0:
                    sys_build.add_optic(poppy.CircularAperture(radius=optic['Radius_m']*u.m, name=optic['Name']+" aperture"))
            
            # if building IrisAO segmented DM
            elif optic['Name'] == 'IrisAO':
                #print('Enter build IrisAO map')
                if irisAOstatus == True:
                    #print('IrisAO present')
                    #sys_build.add_optic(poppy.MultiHexagonAperture(name='IrisAO DM', rings=3, side=7e-4, gap=7e-6, center=True), distance=dz)
                    sys_build.add_optic(irisAOmap)
                else:
                    #print('IrisAO not present')
                    sys_build.add_optic(poppy.ScalarTransmission(planetype=PlaneType.intermediate, name=optic['Name']),distance=dz)
                    
                sys_build.add_optic(poppy.CircularAperture(radius=optic['Radius_m']*u.m, name=optic['Name']+" aperture"))
            
            # for DM, flat mirrors
            elif optic['Type'] == 'mirror' or optic['Type'] == 'DM':
                #print('Enter mirror or DM conditon')
                sys_build.add_optic(poppy.ScalarTransmission(planetype=PlaneType.intermediate, name=optic['Name']),distance=dz)
                sys_build.add_optic(poppy.CircularAperture(radius=optic['Radius_m']*u.m, name=optic['Name']+" aperture"))

            else: # for focal plane, science plane, lyot plane
                #print('Enter focal plane conditon')
                if optic['Type'] == 'fplane':
                    # Apply focal plane correction distance
                    dz = optic['Distance_m'] * u.m + optic['Correction_m']*u.m
                
                sys_build.add_optic(poppy.ScalarTransmission(planetype=PlaneType.intermediate, name=optic['Name']),distance=dz)
                

        # if the most recent optic studied was the break plane, break out of loop.
        if optic['Name'] == break_plane: 
            #print('Finish building FresnelOpticalSystem at %s' % break_plane)
            break
        
    return sys_build


# Function: csvFresnel
# Description: Builds FresnelOpticalSystem from a prescription CSV file passed in
# Input parameters:
#    rx_csv      - system prescription
#    res         - resolution
#    oversamp    - oversampling convention used in PROPER
#    break_plane - plane to break building (so either focal plane or ZWFS image plane)
#    souce_ZWFE  - numerical array of WFE coefficients. All 0's if on-axis target star; LGS will have tilt and defocus components
# Output:
#    sys_build   - FresnelOpticalSystem object with all optics built into it
def csvFresnel2(rx_csv, samp, oversamp, axis, break_plane, source_ZWFE_coeff, irisAOstatus):
    irisAO_radius=rx_csv['Radius_m'][2]*u.m # Element [2] is IrisAO because Element [0] is the diverging source, [1] is OAP1
    
    sys_build = poppy.FresnelOpticalSystem(pupil_diameter=2*irisAO_radius, npix=samp, beam_ratio=oversamp)

    # Entrance Aperture
    sys_build.add_optic(poppy.CircularAperture(radius=irisAO_radius))
    
    # Apply if off-axis LGS is used
    if axis == 'LGS':
        src_aberr = poppy.ZernikeWFE(radius=irisAO_radius.value, coefficients = source_ZWFE_coeff);
        sys_build.add_optic(src_aberr)

    # Build MagAO-X optical system from CSV file to the Lyot plane
    for n_optic,optic in enumerate(rx_csv): # n_optic: count, optic: value
        #print('n_optic = ', n_optic)

        dz = optic['Distance_m'] * u.m # Propagation distance from the previous optic (n_optic-1)
        fl = optic['Focal_Length_m'] * u.m # Focal length of the current optic (n_optic)

        #print('Check PSD file for %s: %s' % (optic['Name'], optic['surf_PSD']))
        # if PSD file present
        if optic['surf_PSD_filename'] != 'none':
            # make a string insertion for the file location
            surf_file_loc = optic['surf_PSD_folder'] + optic['surf_PSD_filename'] + '.fits'
            # call surfFITS to send out surface map
            optic_surface = mf.surfFITS(file_loc = surf_file_loc, optic_type = optic['optic_type'], opdunit = optic['OPD_unit'], name = optic['Name']+' surface')
            # Add generated surface map to optical system
            sys_build.add_optic(optic_surface,distance=dz)

            if fl != 0: # powered optic with PSD file present
                sys_build.add_optic(poppy.QuadraticLens(fl,name=optic['Name'])) 
                # no distance; surface comes first
                sys_build.add_optic(poppy.CircularAperture(radius=optic['Radius_m']*u.m, name=optic['Name']+" aperture"))

            elif optic['Type'] != 'pupil': # non-powered optic but has PSD present that is NOT the pupil
                sys_build.add_optic(poppy.CircularAperture(radius=optic['Radius_m']*u.m, name=optic['Name']+" aperture"))

        # if no PSD file present (DM, focal plane, testing optical surface)
        else:
            #print('Enter no PSD file condition')
            # if powered optic is being tested
            if fl !=0: 
                #print('Enter powered optic condition')
                sys_build.add_optic(poppy.QuadraticLens(fl,name=optic['Name']), distance=dz)
                if n_optic > 0:
                    sys_build.add_optic(poppy.CircularAperture(radius=optic['Radius_m']*u.m, name=optic['Name']+" aperture"))
            
            # if building IrisAO segmented DM
            elif optic['Name'] == 'IrisAO-trans':
                #print('Enter build IrisAO map')
                if irisAOstatus == True:
                    #print('IrisAO present')
                    sys_build.add_optic(poppy.MultiHexagonAperture(name='IrisAO DM', rings=3, side=7e-4,
                                        gap=7e-6, center=True), distance=dz)
                else:
                    #print('IrisAO not present')
                    sys_build.add_optic(poppy.ScalarTransmission(planetype=PlaneType.intermediate, name=optic['Name']),distance=dz)
                    
                sys_build.add_optic(poppy.CircularAperture(radius=optic['Radius_m']*u.m, name=optic['Name']+" aperture"))
            
            elif optic['Name'] == 'IrisAO-opd':
                #print('Enter build IrisAO Zernike map')
                if axis == 'LGS' and irisAOstatus == True:
                    #print('Aberration present on Primary Mirror')
                    # pass in the OPD file for on-axis aberration
                    # call surfFITS to send out surface map
                    surf_file_loc = 'onAxisDMAberration.fits'
                    optic_surface = mf.surfFITS(file_loc = surf_file_loc, optic_type = 'opd', opdunit = 'meters', name = 'DM aberration')
                    # Add generated surface map to optical system
                    sys_build.add_optic(optic_surface,distance=dz)
                    
                else:
                    #print('Aberration not present on Primary Mirror')
                    sys_build.add_optic(poppy.ScalarTransmission(planetype=PlaneType.intermediate, name=optic['Name']),distance=dz)
                    
                sys_build.add_optic(poppy.CircularAperture(radius=optic['Radius_m']*u.m, name=optic['Name']+" aperture"))
            
            # for DM, flat mirrors
            elif optic['Type'] == 'mirror' or optic['Type'] == 'DM':
                #print('Enter mirror or DM conditon')
                sys_build.add_optic(poppy.ScalarTransmission(planetype=PlaneType.intermediate, name=optic['Name']),distance=dz)
                sys_build.add_optic(poppy.CircularAperture(radius=optic['Radius_m']*u.m, name=optic['Name']+" aperture"))

            else: # for focal plane, science plane, lyot plane
                #print('Enter focal plane conditon')
                if optic['Type'] == 'fplane':
                    # Apply focal plane correction distance
                    dz = optic['Distance_m'] * u.m + optic['Correction_m']*u.m
                
                sys_build.add_optic(poppy.ScalarTransmission(planetype=PlaneType.intermediate, name=optic['Name']),distance=dz)
                

        # if the most recent optic studied was the break plane, break out of loop.
        if optic['Name'] == break_plane: 
            #print('Finish building FresnelOpticalSystem at %s' % break_plane)
            break
        
    return sys_build

# Function: csvFresnel
# Description: Builds FresnelOpticalSystem from a prescription CSV file passed in
# Input parameters:
#    rx_csv      - system prescription
#    res         - resolution
#    oversamp    - oversampling convention used in PROPER
#    break_plane - plane to break building (so either focal plane or ZWFS image plane)
#    souce_ZWFE  - numerical array of WFE coefficients. All 0's if on-axis target star; LGS will have tilt and defocus components
#    DM_ZWFE_ceoff - numerical array of WFE coefficients representing DM shape. On-axis will see an aberrated DM.
# Output:
#    sys_build   - FresnelOpticalSystem object with all optics built into it
def csvFresnel3(rx_csv, samp, oversamp, axis, break_plane, source_ZWFE_coeff, irisAOstatus, DM_ZWFE_coeff):
    irisAO_radius=rx_csv['Radius_m'][2]*u.m # Element [2] is IrisAO because Element [0] is the diverging source, [1] is OAP1
    
    
    sys_build = poppy.FresnelOpticalSystem(pupil_diameter=2*irisAO_radius, npix=samp, beam_ratio=oversamp)

    # Entrance Aperture
    sys_build.add_optic(poppy.CircularAperture(radius=irisAO_radius))
    
    # Apply if off-axis LGS is used
    if axis == 'LGS':
        src_aberr = poppy.ZernikeWFE(radius=irisAO_radius.value, coefficients = source_ZWFE_coeff);
        sys_build.add_optic(src_aberr)

    # Build MagAO-X optical system from CSV file to the Lyot plane
    for n_optic,optic in enumerate(rx_csv): # n_optic: count, optic: value
        #print('n_optic = ', n_optic)

        dz = optic['Distance_m'] * u.m # Propagation distance from the previous optic (n_optic-1)
        fl = optic['Focal_Length_m'] * u.m # Focal length of the current optic (n_optic)

        #print('Check PSD file for %s: %s' % (optic['Name'], optic['surf_PSD']))
        # if PSD file present
        if optic['surf_PSD_filename'] != 'none':
            # make a string insertion for the file location
            surf_file_loc = optic['surf_PSD_folder'] + optic['surf_PSD_filename'] + '.fits'
            # call surfFITS to send out surface map
            optic_surface = mf.surfFITS(file_loc = surf_file_loc, optic_type = optic['optic_type'], opdunit = optic['OPD_unit'], name = optic['Name']+' surface')
            # Add generated surface map to optical system
            sys_build.add_optic(optic_surface,distance=dz)

            if fl != 0: # powered optic with PSD file present
                sys_build.add_optic(poppy.QuadraticLens(fl,name=optic['Name'])) 
                # no distance; surface comes first
                sys_build.add_optic(poppy.CircularAperture(radius=optic['Radius_m']*u.m, name=optic['Name']+" aperture"))

            elif optic['Type'] != 'pupil': # non-powered optic but has PSD present that is NOT the pupil
                sys_build.add_optic(poppy.CircularAperture(radius=optic['Radius_m']*u.m, name=optic['Name']+" aperture"))

        # if no PSD file present (DM, focal plane, testing optical surface)
        else:
            #print('Enter no PSD file condition')
            # if powered optic is being tested
            if fl !=0: 
                #print('Enter powered optic condition')
                sys_build.add_optic(poppy.QuadraticLens(fl,name=optic['Name']), distance=dz)
                if n_optic > 0:
                    sys_build.add_optic(poppy.CircularAperture(radius=optic['Radius_m']*u.m, name=optic['Name']+" aperture"))
            
            # if building IrisAO segmented DM
            elif optic['Name'] == 'IrisAO-trans':
                #print('Enter build IrisAO map')
                if irisAOstatus == True:
                    #print('IrisAO present')
                    sys_build.add_optic(poppy.MultiHexagonAperture(name='IrisAO DM', rings=3, side=7e-4,
                                        gap=7e-6, center=True), distance=dz)
                else:
                    #print('IrisAO not present')
                    sys_build.add_optic(poppy.ScalarTransmission(planetype=PlaneType.intermediate, name=optic['Name']),distance=dz)
                    
                sys_build.add_optic(poppy.CircularAperture(radius=optic['Radius_m']*u.m, name=optic['Name']+" aperture"))
            
            elif optic['Name'] == 'IrisAO-opd':
                #print('Enter build IrisAO Zernike map')
                if axis == 'LGS' and irisAOstatus == True:
                    #print('Aberration present on Primary Mirror')
                    # pass in the OPD file for on-axis aberration
                    # call surfFITS to send out surface map
                    surf_file_loc = 'onAxisDMAberration.fits'
                    optic_surface = mf.surfFITS(file_loc = surf_file_loc, optic_type = 'opd', opdunit = 'meters', name = 'DM aberration')
                    # Add generated surface map to optical system
                    sys_build.add_optic(optic_surface,distance=dz)
                    
                else:
                    #print('Aberration not present on Primary Mirror')
                    sys_build.add_optic(poppy.ScalarTransmission(planetype=PlaneType.intermediate, name=optic['Name']),distance=dz)
                    
                sys_build.add_optic(poppy.CircularAperture(radius=optic['Radius_m']*u.m, name=optic['Name']+" aperture"))
            
            # for DM, flat mirrors
            elif optic['Type'] == 'mirror' or optic['Type'] == 'DM':
                #print('Enter mirror or DM conditon')
                sys_build.add_optic(poppy.ScalarTransmission(planetype=PlaneType.intermediate, name=optic['Name']),distance=dz)
                sys_build.add_optic(poppy.CircularAperture(radius=optic['Radius_m']*u.m, name=optic['Name']+" aperture"))

            else: # for focal plane, science plane, lyot plane
                #print('Enter focal plane conditon')
                if optic['Type'] == 'fplane':
                    # Apply focal plane correction distance
                    dz = optic['Distance_m'] * u.m + optic['Correction_m']*u.m
                
                sys_build.add_optic(poppy.ScalarTransmission(planetype=PlaneType.intermediate, name=optic['Name']),distance=dz)
                

        # if the most recent optic studied was the break plane, break out of loop.
        if optic['Name'] == break_plane: 
            #print('Finish building FresnelOpticalSystem at %s' % break_plane)
            break
        
    return sys_build
