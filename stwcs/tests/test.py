import shutil, os  

from astropy import wcs
from astropy.io import fits
from stwcs import updatewcs
from stwcs.updatewcs import apply_corrections
from stwcs.distortion import utils as dutils
from stwcs.wcsutil import HSTWCS
import numpy as np
from numpy.testing import utils
import pytest


class TestStwcs:
    def setUp(self):
        try:
            os.remove('j94f05bgq_flt.fits')
            os.remove('j94f05bgq_flt_r.fits')
        except OSError:
            pass
        shutil.copyfile('orig/j94f05bgq_flt.fits', './j94f05bgq_flt.fits')
        updatewcs.updatewcs('j94f05bgq_flt.fits')
        shutil.copyfile('j94f05bgq_flt.fits', './j94f05bgq_flt_r.fits')
        self.w1 = HSTWCS('j94f05bgq_flt.fits',ext=1)
        self.w4 = HSTWCS('j94f05bgq_flt.fits',ext=4)
        self.m1 = HSTWCS('mj94f05bgq_flt.fits',ext=1)
        self.m4 = HSTWCS('mj94f05bgq_flt.fits',ext=4)
        self.w = wcs.WCS()

    def test_default(self):
        crval = np.array([0.,0.])
        crpix = np.array([0.,0.])
        cdelt = np.array([1.,1.])
        pc = np.array([[1.,0],[0.,1.]])
        ctype = np.array(['',''])
        utils.assert_almost_equal(self.w.wcs.crval, crval)
        utils.assert_almost_equal(self.w.wcs.crpix, crpix)
        utils.assert_almost_equal(self.w.wcs.cdelt, cdelt)
        utils.assert_almost_equal(self.w.wcs.pc, pc)
        assert((self.w.wcs.ctype == np.array(['',''])).all())

    def test_simple_sci1(self):
        """
        A simple sanity check that CRPIX corresponds to CRVAL within wcs
        """
        px1 = np.array([self.w1.wcs.crpix])
        rd1 = np.array([self.w1.wcs.crval])
        assert(((self.w1.all_pix2world(px1,1)- rd1)< 5E-7).all())

    def test_simple_sci2(self):
        """
        A simple sanity check that CRPIX corresponds to CRVAL within wcs
        """
        px4 = np.array([self.w4.wcs.crpix])
        rd4 = np.array([self.w4.wcs.crval])
        assert(((self.w4.all_pix2world(px4,1) - rd4)< 2E-6).all())

    def test_pipeline_sci1(self):
        """
        Internal consistency check of the wcs pipeline
        """
        px = np.array([[100,125]])
        sky1 = self.w1.all_pix2world(px,1)
        dpx1 = self.w1.det2im(px,1)
        #fpx1 = dpx1 + (self.w1.sip_pix2foc(dpx1,1)-dpx1) + (self.w1.p4_pix2foc(dpx1,1)-dpx1)
        fpx1 = dpx1 + (self.w1.sip_pix2foc(dpx1,1)-dpx1+self.w1.wcs.crpix) + (self.w1.p4_pix2foc(dpx1,1)-dpx1)
        pipelinepx1 = self.w1.wcs_pix2world(fpx1,1)
        utils.assert_almost_equal(pipelinepx1, sky1)

    def test_pipeline_sci2(self):
        """
        Internal consistency check of the wcs pipeline
        """
        px = np.array([[100,125]])
        sky4 = self.w4.all_pix2world(px,1)
        dpx4 = self.w4.det2im(px,1)
        fpx4 = dpx4 + (self.w4.sip_pix2foc(dpx4,1)-dpx4 + self.w4.wcs.crpix) + (self.w4.p4_pix2foc(dpx4,1)-dpx4)
        pipelinepx4 = self.w4.wcs_pix2world(fpx4,1)
        utils.assert_almost_equal(pipelinepx4, sky4)

    def test_outwcs(self):
        """
        Test the WCS of the output image
        """
        outwcs=dutils.output_wcs([self.w1,self.w4])

        #print('outwcs.wcs.crval = {0}'.format(outwcs.wcs.crval))
        utils.assert_allclose(
            outwcs.wcs.crval, np.array([5.65109952, -72.0674181]), rtol=1e-7)
        
        utils.assert_almost_equal(outwcs.wcs.crpix, np.array([2107.0,2118.5]))
        utils.assert_almost_equal(
            outwcs.wcs.cd,
            np.array([[1.2787045268089949e-05, 5.4215042082174661e-06],
                      [5.4215042082174661e-06, -1.2787045268089949e-05]]))
        assert(outwcs._naxis1 == 4214)
        assert(outwcs._naxis2 == 4237)

    def test_repeate(self):
        # make sure repeated runs of updatewcs do not change the WCS.
        updatewcs.updatewcs('j94f05bgq_flt.fits')
        self.w1 = HSTWCS('j94f05bgq_flt.fits',ext=('SCI',1))
        self.w4 = HSTWCS('j94f05bgq_flt.fits',ext=('SCI',2))
        self.w1r = HSTWCS('j94f05bgq_flt_r.fits',ext=('SCI',1))
        self.w4r = HSTWCS('j94f05bgq_flt_r.fits',ext=('SCI',2))
        utils.assert_almost_equal(self.w1.wcs.crval, self.w1r.wcs.crval)
        utils.assert_almost_equal(self.w1.wcs.crpix, self.w1r.wcs.crpix)
        utils.assert_almost_equal(self.w1.wcs.cdelt, self.w1r.wcs.cdelt)
        utils.assert_almost_equal(self.w1.wcs.cd, self.w1r.wcs.cd)
        assert((np.array(self.w1.wcs.ctype) == np.array(self.w1r.wcs.ctype)).all())
        utils.assert_almost_equal(self.w1.sip.a, self.w1r.sip.a)
        utils.assert_almost_equal(self.w1.sip.b, self.w1r.sip.b)
        utils.assert_almost_equal(self.w4.wcs.crval, self.w4r.wcs.crval)
        utils.assert_almost_equal(self.w4.wcs.crpix, self.w4r.wcs.crpix)
        utils.assert_almost_equal(self.w4.wcs.cdelt, self.w4r.wcs.cdelt)
        utils.assert_almost_equal(self.w4.wcs.cd, self.w4r.wcs.cd)
        assert((np.array(self.w4.wcs.ctype) == np.array(self.w4r.wcs.ctype)).all())
        utils.assert_almost_equal(self.w4.sip.a, self.w4r.sip.a)
        utils.assert_almost_equal(self.w4.sip.b, self.w4r.sip.b)


def test_remove_npol_distortion():
    try:
        os.remove('j94f05bgq_flt.fits')
    except OSError:
        pass
    shutil.copyfile('orig/j94f05bgq_flt.fits', './j94f05bgq_flt.fits')
    updatewcs.updatewcs('j94f05bgq_flt.fits')
    fits.setval('j94f05bgq_flt.fits', keyword="NPOLFILE", value="N/A")
    updatewcs.updatewcs('j94f05bgq_flt.fits')
    w1 = HSTWCS('j94f05bgq_flt.fits', ext=("SCI", 1))
    w4 = HSTWCS('j94f05bgq_flt.fits', ext=("SCI", 2))
    assert w1.cpdis1 is None
    assert w4.cpdis2 is None


def test_remove_d2im_distortion():
    try:
        os.remove('j94f05bgq_flt.fits')
    except OSError:
        pass
    shutil.copyfile('orig/j94f05bgq_flt.fits', './j94f05bgq_flt.fits')
    updatewcs.updatewcs('j94f05bgq_flt.fits')
    fits.setval('j94f05bgq_flt.fits', keyword="D2IMFILE", value="N/A")
    updatewcs.updatewcs('j94f05bgq_flt.fits')
    w1 = HSTWCS('j94f05bgq_flt.fits', ext=("SCI", 1))
    w4 = HSTWCS('j94f05bgq_flt.fits', ext=("SCI", 2))
    assert w1.det2im1 is None
    assert w4.det2im2 is None


def test_missing_idctab():
    """ Tests that an IOError is raised if an idctab file is not found on disk."""
    try:
        os.remove('j94f05bgq_flt.fits')
    except OSError:
        pass
    shutil.copyfile('orig/j94f05bgq_flt.fits', './j94f05bgq_flt.fits')
    fits.setval('j94f05bgq_flt.fits', keyword="IDCTAB", value="my_missing_idctab.fits")
    with pytest.raises(IOError):
        updatewcs.updatewcs('j94f05bgq_flt.fits')


def test_missing_npolfile():
    """ Tests that an IOError is raised if an NPOLFILE file is not found on disk."""
    try:
        os.remove('j94f05bgq_flt.fits')
    except OSError:
        pass
    shutil.copyfile('orig/j94f05bgq_flt.fits', './j94f05bgq_flt.fits')
    fits.setval('j94f05bgq_flt.fits', keyword="NPOLFILE", value="missing_npl.fits")
    with pytest.raises(IOError):
        updatewcs.updatewcs('j94f05bgq_flt.fits')


def test_missing_d2imfile():
    """ Tests that an IOError is raised if a D2IMFILE file is not found on disk."""
    try:
        os.remove('j94f05bgq_flt.fits')
    except OSError:
        pass
    shutil.copyfile('orig/j94f05bgq_flt.fits', './j94f05bgq_flt.fits')
    fits.setval('j94f05bgq_flt.fits', keyword="D2IMFILE", value="missing_d2i.fits")
    with pytest.raises(IOError):
        updatewcs.updatewcs('j94f05bgq_flt.fits')


def test_found_idctab():
    """ Tests the return value of apply_corrections.foundIDCTAB()."""
    try:
        os.remove('j94f05bgq_flt.fits')
    except OSError:
        pass
    shutil.copyfile('orig/j94f05bgq_flt.fits', './j94f05bgq_flt.fits')
    fits.setval('j94f05bgq_flt.fits', keyword="IDCTAB", value="N/A")
    corrections = apply_corrections.setCorrections('j94f05bgq_flt.fits')
    assert('MakeWCS' not in corrections)
    assert('TDDCor' not in corrections)
    assert('CompSIP' not in corrections)


def test_add_radesys():
    """ test that RADESYS was successfully added to headers."""
    shutil.copyfile('orig/j94f05bgq_flt.fits', './j94f05bgq_flt.fits')
    shutil.copyfile('orig/ibof01ahq_flt.fits', './ibof01ahq_flt.fits')
    updatewcs.updatewcs('j94f05bgq_flt.fits')
    updatewcs.updatewcs('ibof01ahq_flt.fits')
    for ext in [('SCI', 1), ('SCI', 2)]:
        hdr = fits.getheader('j94f05bgq_flt.fits', ext)
        assert hdr['RADESYS'] == 'FK5'

    hdr = fits.getheader('ibof01ahq_flt.fits', ext=('SCI', 1))
    assert hdr['RADESYS'] == 'ICRS'
