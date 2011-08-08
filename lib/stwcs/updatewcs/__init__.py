from __future__ import division # confidence high

import os
import pyfits
import numpy as np
from stwcs import wcsutil
from stwcs.wcsutil import HSTWCS
from stwcs import __version__ as stwcsversion
import pywcs

import utils, corrections, makewcs
import npol, det2im
from stsci.tools import parseinput, fileutil
import apply_corrections

import time
import logging
logger = logging.getLogger('stwcs.updatewcs')

import atexit
atexit.register(logging.shutdown)

#Note: The order of corrections is important

__docformat__ = 'restructuredtext'

def updatewcs(input, vacorr=True, tddcorr=True, npolcorr=True, d2imcorr=True,
              checkfiles=True, wcskey=" ", wcsname=" ", clobber=False, verbose=False):
    """

    Updates HST science files with the best available calibration information.
    This allows users to retrieve from the archive self contained science files
    which do not require additional reference files.

    Basic WCS keywords are updated in the process and new keywords (following WCS
    Paper IV and the SIP convention) as well as new extensions are added to the science files.


    Example
    -------
    >>>from stwcs import updatewcs
    >>>updatewcs.updatewcs(filename)

    Dependencies
    ------------
    `stsci.tools`
    `pyfits`
    `pywcs`

    Parameters
    ----------
    input: a python list of file names or a string (wild card characters allowed)
             input files may be in fits, geis or waiver fits format
    vacorr: boolean
              If True, vecocity aberration correction will be applied
    tddcorr: boolean
             If True, time dependent distortion correction will be applied
    npolcorr: boolean
              If True, a Lookup table distortion will be applied
    d2imcorr: boolean
              If True, detector to image correction will be applied
    checkfiles: boolean
              If True, the format of the input files will be checked,
              geis and waiver fits files will be converted to MEF format.
              Default value is True for standalone mode.
    wcskey: None, one character string A-Z or an empty string of length 1
              If None - the primary WCS is not archived
              If an empty string - the next available wcskey is used for the archive
              A-Z - use this key to archive the WCS
    wcsname: a string
              The name under which the primary WCS is archived after it is updated.
              If an empty string (default), the name of the idctable is used as
              a base.
    clobber: boolean
              a flag for reusing the wcskey when archiving the primary WCS
    """
    if verbose == False:
        logger.setLevel(100)
    else:
        formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        log_filename = 'stwcs.log'
        fh = logging.FileHandler(log_filename, mode='w')
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(formatter)
        logger.addHandler(fh)
        logger.setLevel(verbose)
    args = "vacorr=%s, tddcorr=%s, npolcorr=%s, d2imcorr=%s, checkfiles=%s, \
    wcskey=%s, wcsname=%s, clobber=%s" % (str(vacorr), str(tddcorr), str(npolcorr),
                                          str(d2imcorr), str(checkfiles), str(wcskey),
                                          str(wcsname), str(clobber))
    logger.info('\n\tStarting UPDATEWCS: %s', time.asctime())

    files = parseinput.parseinput(input)[0]
    logger.info("\n\tInput files: %s, " % [i for i in files])
    logger.info("\n\tInput arguments: %s" %args)
    if checkfiles:
        files = checkFiles(files)
        if not files:
            print 'No valid input, quitting ...\n'
            return

    for f in files:
        acorr = apply_corrections.setCorrections(f, vacorr=vacorr, \
            tddcorr=tddcorr,npolcorr=npolcorr, d2imcorr=d2imcorr)
        if 'MakeWCS' in acorr and newIDCTAB(f):
            logger.warning("\n\tNew IDCTAB file detected. All current WCSs will be deleted")
            cleanWCS(f)

        makecorr(f, acorr, wkey=wcskey, wname=wcsname, clobber=False)
    return files

def makecorr(fname, allowed_corr, wkey=" ", wname=" ", clobber=False):
    """
    Purpose
    =======
    Applies corrections to the WCS of a single file

    :Parameters:
    `fname`: string
             file name
    `acorr`: list
             list of corrections to be applied
    `wkey`: None, one character string A-Z or an empty string of length 1
              If None - the primary WCS is not archived
              If an empty string - the next available wcskey is used for the archive
              A-Z - use this key to archive the WCS
    `wname`: a string
              The name under which the primary WCS is archived after it is updated.
              If an empty string (default), the name of the idctable is used as
              a base.
    `clobber`: boolean
              a flag for reusing the wcskey when archiving the primary WCS
    """
    f = pyfits.open(fname, mode='update')
    #restore the original WCS keywords
    #wcsutil.restoreWCS(f, ext=[], wcskey='O', clobber=True)
    #Determine the reference chip and create the reference HSTWCS object
    nrefchip, nrefext = getNrefchip(f)
    wcsutil.restoreWCS(f, nrefext, wcskey='O')
    rwcs = HSTWCS(fobj=f, ext=nrefext)
    rwcs.readModel(update=True,header=f[nrefext].header)

    #wcsutil.archiveWCS(f, nrefext, 'O', wcsname='OPUS', clobber=True)

    if 'DET2IMCorr' in allowed_corr:
        det2im.DET2IMCorr.updateWCS(f)

    # get a wcskey and wcsname from the first extension header
    idcname = fileutil.osfn(rwcs.idctab)
    key, name = getKeyName(f[1].header, wkey, wname, idcname)

    for i in range(len(f))[1:]:
        extn = f[i]

        if extn.header.has_key('extname'):
            extname = extn.header['extname'].lower()
            if  extname == 'sci':
                wcsutil.restoreWCS(f, ext=i, wcskey='O')
                sciextver = extn.header['extver']
                ref_wcs = rwcs.deepcopy()
                hdr = extn.header
                ext_wcs = HSTWCS(fobj=f, ext=i)
                ### check if it exists first!!!
                wcsutil.archiveWCS(f, ext=i, wcskey="O", wcsname="OPUS", reusekey=True)
                ext_wcs.readModel(update=True,header=hdr)
                for c in allowed_corr:
                    if c != 'NPOLCorr' and c != 'DET2IMCorr':
                        corr_klass = corrections.__getattribute__(c)
                        kw2update = corr_klass.updateWCS(ext_wcs, ref_wcs)
                        for kw in kw2update:
                            hdr.update(kw, kw2update[kw])
                #if wkey is None, do not archive the primary WCS
                if key is not None:
                    wcsutil.archiveWCS(f, ext=i, wcskey=key, wcsname=name, reusekey=False)
            elif extname in ['err', 'dq', 'sdq', 'samp', 'time']:
                cextver = extn.header['extver']
                if cextver == sciextver:
                    hdr = f[('SCI',sciextver)].header
                    w = pywcs.WCS(hdr, f)
                    copyWCS(w, extn.header, key, name)
            else:
                continue

    if 'NPOLCorr' in allowed_corr:
        kw2update = npol.NPOLCorr.updateWCS(f)
        for kw in kw2update:
            f[1].header.update(kw, kw2update[kw])
    # Finally record the version of the software which updated the WCS
    if f[0].header.has_key('HISTORY'):
        f[0].header.update(key='UPWCSVER', value=stwcsversion, 
            comment="Version of STWCS used to updated the WCS", before='HISTORY')
    elif f[0].header.has_key('ASN_MTYP'):
        f[0].header.update(key='UPWCSVER', value=stwcsversion, 
            comment="Version of STWCS used to updated the WCS", after='ASN_MTYP')
    else:
        # Find index of last non-blank card, and insert this new keyword after that card
        for i in range(len(f[0].header.ascard)-1,0,-1):
            if f[0].header[i].strip() != '': break
            
            f[0].header.update(key='UPWCSVER', value=stwcsversion, 
                comment="Version of STWCS used to updated the WCS",after=i)
    f.close()

def getKeyName(hdr, wkey, wname, idcname):
    if wkey is not None: # archive the primary WCS
        if wkey == " ":
            if wname == " " :
                # get the next available key and use the IDCTABLE name as WCSNAME
                idcname = os.path.split(idcname)[1]
                name = ''.join(['IDC_',idcname.split('_idc.fits')[0]])
                key = wcsutil.getKeyFromName(hdr, name)
                if not key:
                    key = wcsutil.next_wcskey(hdr)
            else:
                #try to get a key from WCSNAME
                # if not - get the next availabble key
                name = wname
                key = wcsutil.getKeyFromName(hdr, wname)
                if not key:
                    key = wcsutil.next_wcskey(hdr)
        else:
            key = wkey
            name = wname
    return key, name

def copyWCS(w, hdr, wkey, wname):
    """
    This is a convenience function to copy a WCS object
    to a header as a primary WCS. It is used only to copy the
    WCS of the 'SCI' extension to the headers of 'ERR', 'DQ', 'SDQ',
    'TIME' or 'SAMP' extensions.
    """
    hwcs = w.to_header()

    if w.wcs.has_cd():
        wcsutil.pc2cd(hwcs)
    for k in hwcs.keys():
        key = k[:7] + wkey
        hdr.update(key=key, value=hwcs[k])
    norient = np.rad2deg(np.arctan2(hwcs['CD1_2'],hwcs['CD2_2']))
    okey = 'ORIENT%s' % wkey
    hdr.update(key=okey, value=norient)

def getNrefchip(fobj):
    """

    Finds which FITS extension holds the reference chip.

    The reference chip has EXTNAME='SCI', can be in any extension and
    is instrument specific. This functions provides mappings between
    sci extensions, chips and fits extensions.
    In the case of a subarray when the reference chip is missing, the
    first 'SCI' extension is the reference chip.

    Parameters
    ----------
    fobj: pyfits HDUList object
    """
    nrefext = 1
    nrefchip = 1
    instrument = fobj[0].header['INSTRUME']

    if instrument == 'WFPC2':
        chipkw = 'DETECTOR'
        extvers = [("SCI",img.header['EXTVER']) for img in
                   fobj[1:] if img.header['EXTNAME'].lower()=='sci']
        detectors = [img.header[chipkw] for img in fobj[1:] if
                     img.header['EXTNAME'].lower()=='sci']
        fitsext = [i for i in range(len(fobj))[1:] if
                   fobj[i].header['EXTNAME'].lower()=='sci']
        det2ext=dict(map(None, detectors,extvers))
        ext2det=dict(map(None, extvers, detectors))
        ext2fitsext=dict(map(None, extvers, fitsext))

        if 3 not in detectors:
            nrefchip = ext2det.pop(extvers[0])
            nrefext = ext2fitsext.pop(extvers[0])
        else:
            nrefchip = 3
            extname = det2ext.pop(nrefchip)
            nrefext = ext2fitsext.pop(extname)

    elif (instrument == 'ACS' and fobj[0].header['DETECTOR'] == 'WFC') or \
         (instrument == 'WFC3' and fobj[0].header['DETECTOR'] == 'UVIS'):
        chipkw = 'CCDCHIP'
        extvers = [("SCI",img.header['EXTVER']) for img in
                   fobj[1:] if img.header['EXTNAME'].lower()=='sci']
        detectors = [img.header[chipkw] for img in fobj[1:] if
                     img.header['EXTNAME'].lower()=='sci']
        fitsext = [i for i in range(len(fobj))[1:] if
                   fobj[i].header['EXTNAME'].lower()=='sci']
        det2ext=dict(map(None, detectors,extvers))
        ext2det=dict(map(None, extvers, detectors))
        ext2fitsext=dict(map(None, extvers, fitsext))

        if 2 not in detectors:
            nrefchip = ext2det.pop(extvers[0])
            nrefext = ext2fitsext.pop(extvers[0])
        else:
            nrefchip = 2
            extname = det2ext.pop(nrefchip)
            nrefext = ext2fitsext.pop(extname)
    else:
        for i in range(len(fobj)):
            extname = fobj[i].header.get('EXTNAME', None)
            if extname != None and extname.lower == 'sci':
                nrefext = i
                break

    return nrefchip, nrefext

def checkFiles(input):
    """
    Checks that input files are in the correct format.
    Converts geis and waiver fits files to multiextension fits.
    """
    from stsci.tools.check_files import geis2mef, waiver2mef, checkFiles
    logger.info("\n\tChecking files %s" % input)
    removed_files = []
    newfiles = []
    if not isinstance(input, list):
        input=[input]

    for file in input:
        try:
                imgfits,imgtype = fileutil.isFits(file)
        except IOError:
            logger.warning( "\n\tFile %s could not be found, removing it from the input list.\n" %file)
            removed_files.append(file)
            continue
        # Check for existence of waiver FITS input, and quit if found.
        # Or should we print a warning and continue but not use that file
        if imgfits:
            if imgtype == 'waiver':
                newfilename = waiver2mef(file, convert_dq=True)
                if newfilename == None:
                    logger.warning("\n\tRemoving file %s from input list - could not convert waiver to mef" %file)
                    removed_files.append(file)
                else:
                    newfiles.append(newfilename)
            else:
                newfiles.append(file)

        # If a GEIS image is provided as input, create a new MEF file with
        # a name generated using 'buildFITSName()'
        # Convert the corresponding data quality file if present
        if not imgfits:
            newfilename = geis2mef(file, convert_dq=True)
            if newfilename == None:
                logger.warning("\n\tRemoving file %s from input list - could not convert geis to mef" %file)
                removed_files.append(file)
            else:
                newfiles.append(newfilename)
    if removed_files:
        logger.warning('\n\tThe following files will be removed from the list of files to be processed %s' % removed_files)
        #for f in removed_files:
        #    print f

    newfiles = checkFiles(newfiles)[0]
    logger.info("\n\tThese files passed the input check and will be processed: %s" % newfiles)
    return newfiles

def newIDCTAB(fname):
    #When this is called we know there's a kw IDCTAB in the header
    idctab = fileutil.osfn(pyfits.getval(fname, 'IDCTAB'))
    try:
        #check for the presence of IDCTAB in the first extension
        oldidctab = fileutil.osfn(pyfits.getval(fname, 'IDCTAB', ext=1))
    except KeyError:
        return False
    if idctab == oldidctab:
        return False
    else:
        return True

def cleanWCS(fname):
    # A new IDCTAB means all previously computed WCS's are invalid
    # We are deleting all of them except the original OPUS WCS.nvalidates all WCS's.
    keys = wcsutil.wcskeys(pyfits.getheader(fname, ext=1))
    f = pyfits.open(fname, mode='update')
    fext = range(len(f))
    for key in keys:
        wcsutil.deleteWCS(fname, ext=fext,wcskey=key)

def getCorrections(instrument):
    """
    Print corrections available for an instrument

    :Parameters:
    `instrument`: string, one of 'WFPC2', 'NICMOS', 'STIS', 'ACS', 'WFC3'
    """
    acorr = apply_corrections.allowed_corrections[instrument]

    print "The following corrections will be performed for instrument %s\n" % instrument
    for c in acorr: print c,': ' ,  apply_corrections.cnames[c]
