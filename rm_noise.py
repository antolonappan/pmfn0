import numpy as np
from ducc0.misc import wigner3j_int
import camb
from so_models_v3 import SO_Noise_Calculator_Public_v3_1_2 as models

ARCMIN = np.pi / 10800

def camb_spectra(lmax):

    params = camb.CAMBparams()
    params.set_cosmology(
        H0=67.01904, ombh2=0.02216571, omch2=0.1202944,
        tau=0.06018107, mnu=0.06, nnu=3.046, num_massive_neutrinos=1,
        YHe=0.2453006, TCMB=2.7255,
    )
    params.InitPower.set_params(As=2.119631e-9, ns=0.9636852, pivot_scalar=0.05)
    params.set_for_lmax(lmax + 500, lens_potential_accuracy=1)
    results = camb.get_results(params)
    
    fields = ("tt", "ee", "bb", "te")
    options = dict(lmax=lmax, CMB_unit="muK", raw_cl=True)
    prim = dict(zip(fields, results.get_unlensed_scalar_cls(**options).T))
    lensed = dict(zip(fields, results.get_lensed_scalar_cls(**options).T))
    return prim, lensed

def gaussian_beams(fwhm_arcmin, lmax):
    theta = np.atleast_1d(fwhm_arcmin) * ARCMIN
    ell = np.arange(lmax + 1)
    return np.exp(-ell[:, None]*(ell[:, None]+1)*theta**2/(16*np.log(2)))


def white_instrument(bands, lmax):
    bands = np.asarray(bands, dtype=float)
    if bands.ndim != 2 or bands.shape[1] != 3 or np.any(bands <= 0):
        raise ValueError("bands must be positive (nband, 3)")
    nu, fwhm, depth = bands.T
    noise = np.broadcast_to(np.diag((depth*ARCMIN)**2),
                            (lmax+1, len(nu), len(nu))).copy()
    return nu, gaussian_beams(fwhm, lmax), noise.copy(), noise


def so_lat_instrument(lmax, fsky=0.6, sensitivity='baseline'):
    lat = models.SOLatV3point1(sensitivity_mode=sensitivity,
        N_tubes=[1.8, 12, 6], survey_years=5., survey_efficiency=0.2)
    ls, nt, npol = lat.get_noise_curves(fsky, lmax+1, 1,
                                       full_covar=True, deconv_beam=False)
    np.testing.assert_array_equal(ls, np.arange(2, lmax+1))
    nu = np.asarray(lat.get_bands())
    padded = []
    for a in (nt, npol):
        v = np.zeros((lmax+1, len(nu), len(nu)))
        v[2:] = np.moveaxis(a, -1, 0)
        padded.append(v)
    return nu, gaussian_beams(lat.get_beams(), lmax), *padded


def leg_information(prim, lensed, instrument, residual=1., use_tb=True):
    """Eqs. 8-12"""
    if not 0 <= residual <= 1:
        raise ValueError('residual must be in [0,1]')
    nu, beam, nt, npol = map(np.asarray, instrument)
    n = len(nu); size = len(prim['ee'])
    if beam.shape != (size,n) or nt.shape != (size,n,n) or npol.shape != nt.shape:
        raise ValueError('instrument and spectra shapes are inconsistent')
    if np.any(nu <= 0):
        raise ValueError('frequencies must be positive')
    for cov in (nt, npol):
        if not np.all(np.isfinite(cov)) or not np.allclose(cov, cov.swapaxes(1,2)):
            raise ValueError('noise matrices must be finite and symmetric')
  
    beam = beam[2:]; nt = nt[2:]; npol = npol[2:]
    sky = {k: prim[k][2:] + residual*(lensed[k][2:]-prim[k][2:])
           for k in ('tt','ee','bb','te')}
    pair = beam[:,:,None]*beam[:,None,:]
    def block(k): return sky[k][:,None,None]*pair
    ee = block('ee') + npol
    u = prim['ee'][2:,None]*beam
    cx = ee
    if use_tb:
        te = block('te')
        cx = np.concatenate([np.concatenate([block('tt')+nt,te],axis=2),
                             np.concatenate([te,ee],axis=2)],axis=1)
        u = np.concatenate([prim['te'][2:,None]*beam,u],axis=1)
    cb = block('bb')+npol
    v = beam*(0.299792458/nu)**2
    def contract(cov, vec):
        result = np.zeros(size)
        result[2:] = np.sum(vec*np.linalg.solve(cov,vec[...,None])[...,0],axis=1)
        return result
    return contract(cx,u), contract(cb,v)

class Geometry:

    def __init__(self, lmax, Lmax):
        if lmax < 2 or Lmax < 0:
            raise ValueError("Require lmax >= 2 and Lmax >= 0")
        self.lmax, self.Lmax = lmax, Lmax
        self.blocks = []
        for d in range(-min(Lmax, lmax - 2), min(Lmax, lmax - 2) + 1):
            xs = self.x_for(d)
            rows = np.arange(abs(d), Lmax + 1, 2)
            block = np.zeros((len(rows), len(xs)))
            for j, x in enumerate(xs):
                b = int(x + d)
                lo, w = wigner3j_int(int(x), b, 2, -2)
                ok = (rows >= lo) & (rows < lo + len(w))
                block[ok, j] = (2 * x + 1) * (2 * b + 1) / np.pi * w[rows[ok] - lo] ** 2
            self.blocks.append((d, block))

    def x_for(self, d):
        return np.arange(max(2, 2 - d), min(self.lmax, self.lmax - d) + 1)


    def noise(self, x_info, b_info):
        """Eq. 7"""
        if len(x_info) != self.lmax+1 or len(b_info) != self.lmax+1:
            raise ValueError('leg arrays must match geometry lmax')
        fisher = np.zeros(self.Lmax+1)
        for delta, coupling in self.blocks:
            x = self.x_for(delta)
            fisher[abs(delta)::2] += coupling @ (x_info[x]*b_info[x+delta])
        return np.divide(1., fisher, out=np.full_like(fisher,np.inf), where=fisher>0)


def reconstruction_noise(geometry, prim, lensed, instrument, residual=1., use_tb=True):
    """Return N_L^RM"""
    return geometry.noise(*leg_information(prim,lensed,instrument,residual,use_tb))


def noise_dl(noise):
    ls = np.arange(len(noise))
    result = np.zeros_like(noise)
    result[1:] = ls[1:]*(ls[1:]+1)*noise[1:]/(2*np.pi)
    return result


def pmf_dl(b_ng):
    return (50*np.asarray(b_ng))**2
