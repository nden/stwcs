import shutil, os
from astropy.io import fits
from stwcs import updatewcs
from stwcs.wcsutil import headerlet, wcsdiff
from stwcs.wcsutil import HSTWCS
import numpy as np
from numpy.testing import utils
from nose.tools import *

class TestCreateHeaderlet:
    def setUp(self):
        try:
            os.remove('j94f05bgq_flt.fits')
            os.remove('comp.fits')
            os.remove('simple.fits')
        except OSError:
            pass
        shutil.copyfile('orig/j94f05bgq_flt.fits', './j94f05bgq_flt.fits')
        shutil.copyfile('orig/simple.fits', './simple.fits')
        updatewcs.updatewcs('j94f05bgq_flt.fits')
        shutil.copyfile('j94f05bgq_flt.fits', './comp.fits')

    def testAllExt(self):
        """
        Test create_headerlet stepping through all
        extensions in the science file
        """
        hlet = headerlet.create_headerlet('comp.fits', hdrname='hdr1')
        hlet.writeto('hdr1_hlet.fits', clobber=True)
        assert(wcsdiff.is_wcs_identical('comp.fits', 'hdr1_hlet.fits',
                                        [1,4],[("SIPWCS",1),("SIPWCS",2)], verbose=True)[0])

    def testSciExtList(self):
        """
        Test create_headerlet using a list of (EXTNAME, EXTNUM)
        extensions in the science file
        """
        hlet = headerlet.create_headerlet('comp.fits',
                                          sciext=[('SCI',1), ('SCI', 2)],
                                          hdrname='hdr1')
        hlet.writeto('hdr1_hlet.fits', clobber=True)
        assert(wcsdiff.is_wcs_identical('comp.fits', 'hdr1_hlet.fits',
                                        [1, 4], [("SIPWCS",1),("SIPWCS",2)], verbose=True)[0])

    def testSciExtNumList(self):
        """
        Test create_headerlet using a list of EXTNUM
        extensions in the science file
        """
        hlet = headerlet.create_headerlet('comp.fits',
                                          sciext=[1,4],
                                          hdrname='hdr1')
        hlet.writeto('hdr1_hlet.fits', clobber=True)
        assert(wcsdiff.is_wcs_identical('comp.fits', 'hdr1_hlet.fits',
                                        [1, 4], [("SIPWCS",1),("SIPWCS",2) ], verbose=True)[0])

    def testHletFromSimpleFITS(self):
        """
        Test create_headerlet using a FITS HDUList extension
        number in the science file
        """
        hlet = headerlet.create_headerlet('simple.fits',
                                          sciext=0,
                                          hdrname='hdr1')
        hlet.writeto('hdr1_hlet.fits', clobber=True)
        assert(wcsdiff.is_wcs_identical('simple.fits', 'hdr1_hlet.fits',
                                        [0], [1], verbose=True)[0])

    @raises(KeyError)
    def test_no_HDRNAME_no_WCSNAME(self):
        """
        Test create_headerlet stepping through all
        extensions in the science file
        """
        shutil.copyfile('comp.fits', './ncomp.fits')
        fits.delval('ncomp.fits', 'HDRNAME', ext=1)
        fits.delval('ncomp.fits', 'WCSNAME', ext=1)
        hlet = headerlet.create_headerlet('ncomp.fits')

    def test1SciExt(self):
        """
        Create a headerlet from only 1 SCI ext
        """
        hlet = headerlet.create_headerlet('comp.fits',
                                          sciext=4,
                                          hdrname='hdr1')
        hlet.writeto('hdr1_hlet.fits', clobber=True)
        assert(wcsdiff.is_wcs_identical('comp.fits', 'hdr1_hlet.fits',
                                        [4], [1], verbose=True)[0])


class TestApplyHeaderlet:
    def setUp(self):
        try:
            os.remove('j94f05bgq_flt.fits')
            os.remove('comp.fits')
        except OSError:
            pass
        shutil.copyfile('orig/j94f05bgq_flt.fits', './j94f05bgq_flt.fits')
        updatewcs.updatewcs('j94f05bgq_flt.fits')
        shutil.copyfile('j94f05bgq_flt.fits', './comp.fits')

    """
    Does not raise an error currently, should it?
    @raises(TypeError)
    def testWrongDestim(self):
        hlet = headerlet.create_headerlet('comp.fits', sciext=4,
                                          hdrname='hdr1', destim='WRONG')
        hlet.apply_as_primary('comp.fits')
    """

    @raises(ValueError)
    def testWrongSIPModel(self):
        hlet=headerlet.create_headerlet('comp.fits', hdrname='test1',
                                        sipname='WRONG')
        hlet.apply_as_primary('comp.fits')

    @raises(ValueError)
    def testWrongNPOLModel(self):
        hlet=headerlet.create_headerlet('comp.fits', hdrname='test1',
                                        npolfile='WRONG')
        hlet.apply_as_primary('comp.fits')

    @raises(ValueError)
    def testWrongD2IMModel(self):
        hlet=headerlet.create_headerlet('comp.fits', hdrname='test1',
                                        d2imfile='WRONG')
        hlet.apply_as_primary('comp.fits')

    def test_apply_as_primary_method(self):
        hlet=headerlet.create_headerlet('comp.fits', hdrname='test2')
        hlet['SIPWCS',1].header['CRPIX1'] = 1
        hlet['SIPWCS',1].header['CRPIX2'] = 1
        hlet['SIPWCS',2].header['CRPIX1'] = 2
        hlet['SIPWCS',2].header['CRPIX2'] = 2
        hlet.apply_as_primary('comp.fits')
        hlet.writeto('hdr1_hlet.fits', clobber=True)
        assert(wcsdiff.is_wcs_identical('comp.fits', 'hdr1_hlet.fits',
                        [('SCI', 1), ('SCI', 2)], [("SIPWCS",1),("SIPWCS",2)], verbose=True)[0])
        # repeated hlet should not change sci file
        try:
            headerlet.apply_headerlet_as_primary('comp.fits', 'hdr1_hlet.fits')
        except:
            pass
        assert(wcsdiff.is_wcs_identical('comp.fits', 'hdr1_hlet.fits',
                                        [('SCI', 1), ('SCI', 2)],
                                        [("SIPWCS",1),("SIPWCS",2)],
                                        verbose=True)[0])

    def test_apply_as_alternate_method(self):
        hlet=headerlet.create_headerlet('comp.fits', hdrname='test1')
        hlet.apply_as_alternate('comp.fits', wcskey='K', wcsname='KK')
        hlet.writeto('hdr1_hlet.fits', clobber=True)
        assert(wcsdiff.is_wcs_identical('comp.fits', 'hdr1_hlet.fits',
                                        [('SCI',1), ('SCI', 2)],
                                        [("SIPWCS",1),("SIPWCS",2)],
                                        scikey='K', verbose=True)[0])
        headerlet.apply_headerlet_as_alternate('comp.fits',
                                               'hdr1_hlet.fits', wcskey='P')
        assert(wcsdiff.is_wcs_identical('comp.fits', 'hdr1_hlet.fits',
                                        [('SCI', 1), ('SCI', 2)],
                                        [("SIPWCS",1),("SIPWCS",2)],
                                        scikey='P', verbose=True)[0])


class TestLegacyFiles:
    def setUp(self):
        try:
            os.remove('j94f05bgq_flt.fits')
            os.remove('jlegacy.fits')
        except OSError:
            pass
        shutil.copyfile('orig/j94f05bgq_flt.fits', './j94f05bgq_flt.fits')
        shutil.copyfile('j94f05bgq_flt.fits', './jlegacy.fits')
        updatewcs.updatewcs('j94f05bgq_flt.fits')

    def test_update_legacy_file(self):
        hlet = headerlet.create_headerlet('j94f05bgq_flt.fits')
        hlet.apply_as_primary('jlegacy.fits', archive=False, attach=False)
        assert(wcsdiff.is_wcs_identical('j94f05bgq_flt.fits', './jlegacy.fits',
                                        [('SCI', 1), ('SCI', 2)], [("SCI",1),("SCI",2)], verbose=True)[0])
