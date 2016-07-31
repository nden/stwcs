import shutil, os
from astropy.io import fits as pyfits
from stwcs.wcsutil import altwcs
from stwcs import updatewcs
from stwcs.wcsutil import HSTWCS
import numpy as np
from numpy.testing import utils

def compare_wcs(w1, w2, exclude_keywords=None):
    """
    Compare two WCSs.

    Parameters
    ----------
    w1, w2 : `astropy.wcs.WCS` objects
    exclude_keywords : list
        List of keywords to excude from comparison.
    """
    exclude_ctype = False
    keywords = ['crval', 'crpix', 'cd']
    if exclude_keywords is not None:
        exclude_keywords = [kw.lower() for kw in exclude_keywords]
        if 'ctype' in exclude_keywords:
            exclude_ctype = True
            exclude_keywords.remove('ctype')
        for kw in exclude_keywords:
            keywords.remove(kw)
    for kw in keywords:
        kw1 = getattr(w1.wcs, kw)
        kw2 = getattr(w2.wcs, kw)
        utils.assert_allclose(kw1, kw2, 1e-10)
    #utils.assert_allclose(w1.wcs.crpix, w2.wcs.crpix, 1e-10)
    #utils.assert_allclose(w1.wcs.cd, w2.wcs.cd, 1e-10)
    if not exclude_ctype:
        utils.assert_array_equal(np.array(w1.wcs.ctype), np.array(w2.wcs.ctype))

class TestAltWCS:
    def setUp(self):
        try:
            os.remove('j94f05bgq_flt.fits')
            os.remove('simple.fits')
        except OSError:
            pass
        shutil.copyfile('orig/j94f05bgq_flt.fits', './j94f05bgq_flt.fits')
        shutil.copyfile('orig/simple.fits', './simple.fits')
        updatewcs.updatewcs('j94f05bgq_flt.fits')
        self.filename = 'j94f05bgq_flt.fits'
        self.simplefits = 'simple.fits'
        self.ww = HSTWCS(self.filename, ext=1)

    def test_archive(self):
        altwcs.archiveWCS(self.filename, ext=1, wcskey='Z', wcsname='ZTEST', reusekey=False)
        w1 = HSTWCS(self.filename, ext=1)
        w1z = HSTWCS(self.filename, ext=1, wcskey='Z')
        compare_wcs(w1, w1z)

    def test_archive_clobber(self):
        altwcs.archiveWCS(self.filename, ext=1, wcskey='Z', wcsname='ZTEST', reusekey=True)
        w1 = HSTWCS(self.filename, ext=1)
        w1z = HSTWCS(self.filename, ext=1, wcskey='Z')
        compare_wcs(w1, w1z)

    def test_restoreWCS(self):
        # test restore on a file
        altwcs.restoreWCS(self.filename, ext=1, wcskey='O')
        w1o = HSTWCS(self.filename, ext=1, wcskey='O')
        w1 = HSTWCS(self.filename, ext=1)
        compare_wcs(w1, w1o, exclude_keywords=['ctype'])

    def test_restoreWCSMem(self):
        # test restore on an HDUList object
        altwcs.archiveWCS(self.filename, ext=[('SCI',1), ('SCI',2)], wcskey='T')
        pyfits.setval(self.filename, ext=('SCI',1), keyword='CRVAL1', value=1)
        pyfits.setval(self.filename, ext=('SCI',2), keyword='CRVAL1', value=1)
        f = pyfits.open(self.filename, mode='update')
        altwcs.restoreWCS(f, ext=1, wcskey='T')
        f.close()
        w1o = HSTWCS(self.filename, ext=1, wcskey='T')
        w1 = HSTWCS(self.filename, ext=1)
        compare_wcs(w1, w1o)

    def test_restoreSimple(self):
        # test restore on simple fits format
        altwcs.archiveWCS(self.simplefits, ext=0, wcskey='R')
        pyfits.setval(self.simplefits, ext=0, keyword='CRVAL1R', value=1)
        altwcs.restoreWCS(self.simplefits, ext=0, wcskey='R')
        wo = HSTWCS(self.simplefits, ext=0, wcskey='R')
        ws = HSTWCS(self.simplefits, ext=0)
        compare_wcs(ws, wo)

    def test_restoreWCSFromTo(self):
        # test restore from ... to ...
        altwcs.archiveWCS(self.filename, ext=[('SCI',1), ('SCI',2)], wcskey='T')
        pyfits.setval(self.filename, ext=('SCI',1), keyword='CRVAL1', value=1)
        pyfits.setval(self.filename, ext=('SCI',2), keyword='CRVAL1', value=1)
        f = pyfits.open(self.filename, mode='update')
        altwcs.restore_from_to(f, fromext='SCI', toext=['SCI', 'ERR', 'DQ'],
                          wcskey='T')
        f.close()
        w1o = HSTWCS(self.filename, ext=('SCI',1), wcskey='T')
        w1 = HSTWCS(self.filename, ext=('SCI', 1))
        compare_wcs(w1, w1o)
        w2 = HSTWCS(self.filename, ext=('ERR',1))
        compare_wcs(w2, w1o, exclude_keywords=['ctype'])
        w3 = HSTWCS(self.filename, ext=('DQ',1))
        compare_wcs(w3, w1o, exclude_keywords=['ctype'])
        w4o = HSTWCS(self.filename, ext=4, wcskey='T')
        w4 = HSTWCS(self.filename, ext=('SCI',2))
        compare_wcs(w4, w4o)
        w5 = HSTWCS(self.filename, ext=('ERR', 2))
        compare_wcs(w5, w4o, exclude_keywords=['ctype'])
        w6 = HSTWCS(self.filename, ext=('DQ',2))
        compare_wcs(w3, w1o, exclude_keywords=['ctype'])

    def test_deleteWCS(self):
        altwcs.archiveWCS(self.filename, ext=1, wcskey='Z')
        altwcs.deleteWCS(self.filename, ext=1, wcskey='Z')
        utils.assert_raises(KeyError, HSTWCS, self.filename, ext=1, wcskey='Z')

    def test_pars_file_mode1(self):
        assert( not altwcs._parpasscheck(self.filename, ext=1, wcskey='Z'))

    def test_pars_file_mode2(self):
        f = pyfits.open(self.filename)
        assert( not altwcs._parpasscheck(f, ext=1, wcskey='Z'))
        f.close()

    def test_pars_ext(self):
        f = pyfits.open(self.filename, mode='update')
        assert(altwcs._parpasscheck(f, ext=1, wcskey='Z'))
        assert(altwcs._parpasscheck(f, ext=[('sci',1),('sci',2)], wcskey='Z'))
        assert(altwcs._parpasscheck(f, ext=('sci', 1), wcskey='Z'))
        f.close()

    def test_pars_wcskey_not1char(self):
        f = pyfits.open(self.filename, mode='update')
        assert(not altwcs._parpasscheck(f, ext=1, wcskey='ZZ'))
        f.close()

    def test_pars_wcskey(self):
        f = pyfits.open(self.filename, mode='update')
        assert(altwcs._parpasscheck(f, ext=1, wcskey=' '))
        #assert(not altwcs._parpasscheck(f, ext=1, wcskey=' ', reusekey=False))
        #assert(altwcs._parpasscheck(f, ext=1, wcskey='O'))
        #assert(not altwcs._parpasscheck(f, ext=1, wcskey='O', reusekey=False))
        f.close()
