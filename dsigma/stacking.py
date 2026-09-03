"""Module for stacking lensing results after pre-computation."""

import numpy as np

from astropy import units as u
from astropy.cosmology import FlatLambdaCDM
from astropy.table import Table
from astropy.units import UnitConversionError
from scipy.optimize import minimize
from .physics import mpc_per_degree, lens_magnification_shear_bias

__all__ = ['number_of_pairs', 'raw_tangential_shear',
           'raw_excess_surface_density', 'photo_z_dilution_factor',
           'boost_factor', 'boost_factor_from_pz', 'scalar_shear_response_factor',
           'matrix_shear_response_factor', 'shear_responsivity_factor',
           'mean_lens_redshift', 'mean_source_redshift',
           'mean_critical_surface_density', 'lens_magnification_bias',
           'tangential_shear', 'excess_surface_density']

def gaussian(x, mu, sigma):
    """Evaluate a Gaussian function.

    Parameters
    ----------
    x : numpy.ndarray
        The input values at which to evaluate the Gaussian.
    mu : float
        The mean of the Gaussian.
    sigma : float
        The standard deviation of the Gaussian.

    Returns
    -------
    g : numpy.ndarray
        The value of the Gaussian function at each input value.

    """
    return np.exp(-0.5 * ((x - mu) / sigma)**2) / (sigma * np.sqrt(2 * np.pi))

def number_of_pairs(table_l):
    """Compute the number of lens-source pairs per bin.

    Parameters
    ----------
    table_l : astropy.table.Table
        Precompute results for the lenses.

    Returns
    -------
    n_pairs : numpy.ndarray
        The number of lens-source pairs in each radial bin.

    """
    return np.sum(table_l['sum 1'].data, axis=0)


def raw_tangential_shear(table_l):
    """Compute the average tangential shear for a catalog.

    Parameters
    ----------
    table_l : astropy.table.Table
        Precompute results for the lenses.

    Returns
    -------
    delta_sigma : numpy.ndarray
        The raw, uncorrected tangential shear in each radial bin.

    """
    return (np.sum(table_l['sum w_ls e_t'].data *
                   table_l['w_sys'].data[:, None], axis=0) /
            np.sum(table_l['sum w_ls'].data * table_l['w_sys'].data[:, None],
                   axis=0))


def raw_excess_surface_density(table_l):
    """Compute the raw, uncorrected excess surface density for a catalog.

    Parameters
    ----------
    table_l : astropy.table.Table
        Precompute results for the lenses.

    Returns
    -------
    delta_sigma : numpy.ndarray
        The raw, uncorrected excess surface density in each radial bin.

    """
    return (np.sum(table_l['sum w_ls e_t sigma_crit'].data *
                   table_l['w_sys'].data[:, None], axis=0) /
            np.sum(table_l['sum w_ls'].data *
                   table_l['w_sys'].data[:, None], axis=0))


def photo_z_dilution_factor(table_l):
    r"""Compute the photometric redshift bias averaged over the entire catalog.

    Parameters
    ----------
    table_l : astropy.table.Table
        Precompute results for the lenses.

    Returns
    -------
    f_bias : float
        Photometric redshift bias :math:`f_{\mathrm{bias}}`.

    """
    return (np.sum(table_l['sum w_ls e_t sigma_crit f_bias'].data *
                   table_l['w_sys'].data[:, None], axis=0) /
            np.sum(table_l['sum w_ls e_t sigma_crit'].data *
                   table_l['w_sys'].data[:, None], axis=0))


def boost_factor(table_l, table_r):
    """Compute the boost factor.

    Boost factor is computed by comparing the number of lens-source pairs
    in real lenses and random lenses.

    Parameters
    ----------
    table_l : astropy.table.Table
        Precompute results for the lenses.
    table_r : astropy.table.Table, optional
        Precompute results for random lenses.

    Returns
    -------
    b : numpy.ndarray
        Boost factor in each radial bin.

    """
    return (
        np.sum(table_l['sum w_ls'].data *
               table_l['w_sys'].data[:, None], axis=0) /
        np.sum(table_l['w_sys'].data) /
        np.sum(table_r['sum w_ls'].data *
               table_r['w_sys'].data[:, None], axis=0) *
        np.sum(table_r['w_sys'].data))

def normalize_pair_pdf(table):
    z_mids = table.meta['boostFactor_pdf_zbins']
    z_mids = (z_mids[1:] + z_mids[:-1]) / 2
    sum_w_ls = np.sum(table['sum w_ls'] * table['w_sys'].data[:, None], axis=0)
    pdf_hist = np.sum(table['PDF w_ls z_s'] * table['w_sys'].data[:, None, None], axis=0)
    pdf = np.full(pdf_hist.shape, np.nan, dtype=np.float64)
    valid = sum_w_ls > 0
    if not np.any(valid):
        return pdf, valid

    pdf[valid] = pdf_hist[valid] / sum_w_ls[valid, None]
    norm = np.trapz(pdf[valid], z_mids, axis=1)
    finite = np.isfinite(norm) & (norm > 0)
    valid_indices = np.flatnonzero(valid)
    valid = np.zeros_like(valid, dtype=bool)
    if not np.any(finite):
        pdf[:] = np.nan
        return pdf, valid

    pdf_valid = pdf[valid_indices[finite]].copy()
    pdf[:] = np.nan
    pdf_valid /= norm[finite, None]
    pdf[valid_indices[finite]] = pdf_valid
    valid[valid_indices[finite]] = True
    return pdf, valid

def boost_factor_from_pz(table_l, return_pdf=False, optimize_method='trust-constr'):
    assert 'PDF w_ls z_s' in table_l.colnames, 'table_l must contain PDF w_ls z_s column'
    z_bins = table_l.meta['boostFactor_pdf_zbins']
    z_mids = 0.5 * (z_bins[1:] + z_bins[:-1])
    if table_l.meta['save_individual_pdfz']:
        pdf_z, valid = normalize_pair_pdf(table_l)
    else:
        raise Exception("Individual P(z) is not saved. Can't perform boost factor estimates from p(z)!")
        pdf_z = table_l.meta['PDF z_s']
        valid = table_l.meta['PDF valididx']
        
    # if background is not None:
    #     if background.meta['save_individual_pdfz']:
    #         pdf_z_r, valid_r = normalize_pair_pdf(background)
    #         valid &= valid_r
    #         background = pdf_z_r
    #     else:
    #         pdf_z_r = background.meta['PDF z_s']
    #         valid &= background.meta['PDF valididx']
    #         background = pdf_z_r
    # else:
    background = np.full(pdf_z.shape, np.nan, dtype=np.float64)
    valid_idx = np.flatnonzero(valid)
    if valid_idx.size:
        background[valid] = pdf_z[valid_idx[-1]]

    if not np.any(valid):
        raise ValueError('Cannot compute boost factor from p(z): no radial bins have finite pair-weight PDFs.')

    pdf_z_fit = pdf_z[valid]
    background_fit = background[valid]
    nrbins = len(pdf_z)
    n_fit_bins = len(pdf_z_fit)

    def model_pdf(theta, decompose=False):
        mean, std = theta[:2]
        f_s = theta[2:]
        pdf_fg = f_s[:, None] * gaussian(z_mids, mean, std).reshape(1, -1)
        pdf_bg = (1 - f_s[:, None]) * background_fit
        if decompose:
            return pdf_fg, pdf_bg
        return pdf_fg + pdf_bg

    def chisq(theta):
        model = model_pdf(theta, False)
        return np.sum((pdf_z_fit - model)**2)

    bounds = [(0, 100), (1e-4, .2)] + [(0, 1)] * n_fit_bins
    init = [0.5, 0.05] + [0.5] * n_fit_bins
    optres = minimize(chisq, init, bounds=bounds, method=optimize_method, tol=1e-6,
                      options={'maxiter': 100000})
    if not optres.success:
        raise RuntimeError('Optimization failed: ' + optres.message)

    boost = np.full(nrbins, np.nan, dtype=np.float64)
    boost[valid] = 1 / (1. - optres.x[2:])
    
    if return_pdf:
        pdf_fg, pdf_bg = model_pdf(optres.x, decompose=True)
        model_fg = np.full(pdf_z.shape, np.nan, dtype=np.float64)
        model_bg = np.full(pdf_z.shape, np.nan, dtype=np.float64)
        model_fg[valid] = pdf_fg
        model_bg[valid] = pdf_bg
        return {'data': (pdf_z, background), 'model': (model_fg, model_bg),
                'z_mids': z_mids, 'opt_result': optres, 'boost': boost}
        
    return boost # boost factor is 1/(1-f_s)

def scalar_shear_response_factor(table_l, selection_bias=False):
    r"""Compute the mean shear response.

    The shear response factor :math:`m` is defined such that
    :math:`\gamma_{\mathrm obs} = (1 + m) \gamma_{\mathrm intrinsic}`.

    Parameters
    ----------
    table_l : astropy.table.Table
        Precompute results for the lenses.
    selection_bias : bool
        If True, calculate the selection bias :math:`m_\mathrm{sel}`, instead.
        Default is False.

    Returns
    -------
    m : numpy.ndarray
        Multiplicative shear bias in each radial bin.

    """
    if selection_bias:
        m = 'm_sel'
    else:
        m = 'm'

    return (
        np.sum(table_l[f'sum w_ls {m}'].data *
               table_l['w_sys'].data[:, None], axis=0) /
        np.sum(table_l['sum w_ls'].data *
               table_l['w_sys'].data[:, None], axis=0))

def additive_bias(table_l, sigma_crit=True):
    r"""Compute the tangential additive bias.
    The additive bias :math:`c_t` is defined such that
    `\gamma_{\mathrm obs} = (1 + m ) \gamma_{\mathrm intrinsic} + c_t`.

    Parameters
    ----------
    table_l : astropy.table.Table
        Precompute results for the lenses.

    Returns
    -------
    ct : numpy.ndarray
        Tangential additive bias in each radial bin.
    """
    if sigma_crit:
        return (
                np.sum(table_l['sum w_ls c_t sigma_crit'].data *
                    table_l['w_sys'].data[:, None], axis=0) /
                np.sum(table_l['sum w_ls'].data *
                    table_l['w_sys'].data[:, None], axis=0))
    else:
        return (
                np.sum(table_l['sum w_ls c_t'].data *
                    table_l['w_sys'].data[:, None], axis=0) /
                np.sum(table_l['sum w_ls'].data *
                    table_l['w_sys'].data[:, None], axis=0))

def matrix_shear_response_factor(table_l):
    r"""Compute the mean tangential response.

    The tangential shear response factor:math:`R_t` is defined such that
    :math:`\gamma_{\mathrm obs} = R_t \gamma_{\mathrm intrinsic}`.

    Parameters
    ----------
    table_l : astropy.table.Table
        Precompute results for the lenses.

    Returns
    -------
    r_t : numpy.ndarray
        Tangential shear response factor in each radial bin.

    """
    return (
        np.sum(table_l['sum w_ls R_T'] * table_l['w_sys'][:, None],
               axis=0) /
        np.sum(table_l['sum w_ls'] * table_l['w_sys'][:, None], axis=0))


def shear_responsivity_factor(table_l):
    """Compute the shear responsitivity factor.

    Parameters
    ----------
    table_l : astropy.table.Table
        Precompute results for the lenses.

    Returns
    -------
    r : numpy.ndarray
        Shear responsitivity factor in each radial bin.

    """
    return (
        np.sum(table_l['sum w_ls (1 - e_rms^2)'] *
               table_l['w_sys'][:, None], axis=0) /
        np.sum(table_l['sum w_ls'] * table_l['w_sys'][:, None], axis=0))


def mean_lens_redshift(table_l):
    """Compute the weighted-average lens redshift.

    Parameters
    ----------
    table_l : astropy.table.Table
        Precompute results for the lenses.

    Returns
    -------
    z_l : numpy.ndarray
        Mean lens redshift in each bin.

    """
    return (
        np.sum(table_l['sum w_ls z_l'] * table_l['w_sys'][:, None], axis=0) /
        np.sum(table_l['sum w_ls'] * table_l['w_sys'][:, None], axis=0))


def mean_source_redshift(table_l):
    """Compute the weighted-average source redshift.

    Parameters
    ----------
    table_l : astropy.table.Table
        Precompute results for the lenses.

    Returns
    -------
    z_s : numpy.ndarray
        Mean source redshift in each bin.

    """
    return (
        np.sum(table_l['sum w_ls z_s'] * table_l['w_sys'][:, None], axis=0) /
        np.sum(table_l['sum w_ls'] * table_l['w_sys'][:, None], axis=0))


def mean_critical_surface_density(table_l, photo_z_dilution_correction=False):
    """Compute the weighted-average (effective) critical surface density.

    Parameters
    ----------
    table_l : astropy.table.Table
        Precompute results for the lenses.
    photo_z_dilution_correction : bool, optional
        If True, correct for photo-z biases. This can only be done if a
        calibration catalog has been provided in the Precomputation phase.
        Default is False.

    Returns
    -------
    sigma_crit : numpy.ndarray
        Mean (effective) critical surface density.

    """
    if photo_z_dilution_correction:
        key = 'sum w_ls sigma_crit f_bias'
    else:
        key = 'sum w_ls sigma_crit'
    return (
        np.sum(table_l[key] * table_l['w_sys'][:, None], axis=0) /
        np.sum(table_l['sum w_ls'] * table_l['w_sys'][:, None], axis=0))


def lens_magnification_bias(table_l, alpha_l, camb_results,
                            photo_z_dilution_correction=False, shear=False):
    """Estimate the additive lens magnification bias.

    Parameters
    ----------
    table_l : astropy.table.Table
        Precompute results for the lenses.
    alpha_l : float
        The response of the lenses to magnification.
    camb_results : camb.results.CAMBdata
        CAMB results object that contains information on cosmology and the
        matter power spectrum.
    photo_z_dilution_correction : bool, optional
        If True, correct the mean critical surface density for photo-z biases.
        Not used if `shear` is True. This should be consistent with what is
        used for calculating the total excess surface density. Default is
        False.
    shear : bool, optional
        If True, return bias of the mean tangential shear. Otherwise, return
        an estimate for the bias of the excess surface density. Default is
        False.

    Returns
    -------
    ds_lm : numpy.ndarray
        The lens magnification bias in each radial bin.

    """
    cosmology = FlatLambdaCDM(H0=table_l.meta['H0'], Om0=table_l.meta['Om0'])

    z_l = mean_lens_redshift(table_l)
    z_s = mean_source_redshift(table_l)
    bins = table_l.meta['bins']
    d = 2.0 / 3.0 * np.diff(bins**3) / np.diff(bins**2)

    try:
        theta = d.to(u.rad).value
    except UnitConversionError:
        theta = np.deg2rad(d.to(u.Mpc).value / mpc_per_degree(
            z_l, cosmology=cosmology, comoving=table_l.meta['comoving']))

    gt = np.array([lens_magnification_shear_bias(
        theta[i], alpha_l, z_l[i], z_s[i], camb_results) for i in
        range(len(theta))])

    if shear:
        return gt
    else:
        return gt * mean_critical_surface_density(
            table_l, photo_z_dilution_correction=photo_z_dilution_correction)


def tangential_shear(table_l, table_r=None, boost_correction=False,
                     additive_bias_correction=False,
                     scalar_shear_response_correction=False,
                     matrix_shear_response_correction=False,
                     shear_responsivity_correction=False,
                     selection_bias_correction=False,
                     random_subtraction=False, return_table=False):
    """Compute the mean tangential shear with corrections, if applicable.

    Parameters
    ----------
    table_l : astropy.table.Table
        Precompute results for the lenses.
    table_r : astropy.table.Table, optional
        Precompute results for random lenses. Default is None.
    additive_bias_correction : bool, optional
        If True, correct for the additive bias. Default is False.
    boost_correction : bool, optional
        If True, calculate and apply a boost factor correction. This can only
        be done if a random catalog is provided. Default is False.
    scalar_shear_response_correction : bool or string, optional
        Whether to correct for the multiplicative shear bias (scalar form).
        Default is False.
    matrix_shear_response_correction : bool or string, optional
        Whether to correct for the multiplicative shear bias (tensor form).
        Default is False.
    shear_responsivity_correction : bool, optional
        If True, correct for the shear responsivity. Default is False.
    selection_bias_correction : bool, optional
        If True, correct for the multiplicative selection bias in, e.g., HSC.
        Default is False.
    random_subtraction : bool, optional
        If True, subtract the signal around randoms. This can only be done if
        a random catalog is provided. Default is False.
    return_table : bool, optional
        If True, return a table with many intermediate steps of the
        computation. Otherwise, a simple array with just the final tangential
        shearis returned. Default is False.
    
    Returns
    -------
    e_t : numpy.ndarray or astropy.table.Table
        The tangential shear in each radial bin specified in the precomputation
        phase. If `return_table` is True, will return a table with detailed
        information for each radial bin. The final result is in the column
        `et`.

    Raises
    ------
    ValueError
        If boost or random subtraction correction are requested but no random
        catalog is provided.

    """
    result = Table()

    result['rp_min'] = table_l.meta['bins'][:-1]
    result['rp_max'] = table_l.meta['bins'][1:]
    result['n_pairs'] = number_of_pairs(table_l)
    result['rp'] = np.sqrt(result['rp_min'] * result['rp_max'])
    result['et_raw'] = raw_tangential_shear(table_l)
    result['et'] = raw_tangential_shear(table_l)
    result['z_l'] = mean_lens_redshift(table_l)
    result['z_s'] = mean_source_redshift(table_l)

    if shear_responsivity_correction:
        result['2R'] = 2 * shear_responsivity_factor(table_l)
        result['et'] /= result['2R']

    if matrix_shear_response_correction:
        result['R_t'] = matrix_shear_response_factor(table_l)
        result['et'] /= result['R_t']

    if additive_bias_correction:
        result['ct'] = additive_bias(table_l, sigma_crit=False)
        result['et'] -= result['ct']

    if scalar_shear_response_correction:
        result['1+m'] = 1 + scalar_shear_response_factor(table_l)
        result['et'] /= result['1+m']

    if selection_bias_correction:
        result['1+m_sel'] = 1 + scalar_shear_response_factor(
            table_l, selection_bias=True)
        result['et'] /= result['1+m_sel']

    if random_subtraction:
        if table_r is None:
            raise ValueError('Cannot subtract random results without ' +
                                'results from a random catalog.')
        result['et_r'] = tangential_shear(
            table_r, boost_correction=False,
            additive_bias_correction=additive_bias_correction,
            scalar_shear_response_correction=scalar_shear_response_correction,
            matrix_shear_response_correction=matrix_shear_response_correction,
            shear_responsivity_correction=shear_responsivity_correction,
            selection_bias_correction=selection_bias_correction,
            random_subtraction=False, return_table=False)
        result['et'] -= result['et_r']

    if boost_correction:
        if table_r is None:
            raise ValueError('Cannot compute boost factor correction without' +
                                ' results from a random catalog.')
        result['b'] = boost_factor(table_l, table_r)
        result['et'] *= result['b']

    if not return_table:
        return result['et'].data

    return result


def excess_surface_density(table_l, table_r=None,
                           additive_bias_correction=False,
                           photo_z_dilution_correction=False,
                           boost_correction=False,
                           pz_boost_correction=False,
                           scalar_shear_response_correction=False,
                           matrix_shear_response_correction=False,
                           shear_responsivity_correction=False,
                           selection_bias_correction=False,
                           random_subtraction=False,
                           return_table=False):
    """Compute the mean excess surface density with corrections, if applicable.

    Parameters
    ----------
    table_l : astropy.table.Table
        Precompute results for the lenses.
    table_r : astropy.table.Table, optional
        Precompute results for random lenses. Default is None.
    photo_z_dilution_correction : bool, optional
        If True, correct for photo-z biases. This can only be done if a
        calibration catalog has been provided in the precomputation phase.
        Default is False.
    additive_bias_correction : bool, optional
        If True, correct for the additive bias. Default is False.
    boost_correction : bool, optional
        If true, calculate and apply a boost factor correction. This can only
        be done if a random catalog is provided. Default is False.
    pz_boost_correction : bool, optional
        If True, calculate and apply a boost factor correction using the
        p(z) decomposition method, only if boost_correction is False.
    scalar_shear_response_correction : bool or string, optional
        Whether to correct for the multiplicative shear bias (scalar form).
        Default is False.
    matrix_shear_response_correction : bool or string, optional
        Whether to correct for the multiplicative shear bias (tensor form).
        Default is False.
    shear_responsivity_correction : bool, optional
        If True, correct for the shear responsivity. Default is False.
    selection_bias_correction : bool, optional
        If True, correct for the multiplicative selection bias in, e.g., HSC.
        Default is False.
    random_subtraction : bool, optional
        If True, subtract the signal around randoms. This can only be done if
        a random catalog is provided. Default is False.
    return_table : bool, optional
        If True, return a table with many intermediate steps of the
        computation. Otherwise, a simple array with just the final excess
        surface density is returned. Default is False.

    Returns
    -------
    delta_sigma : numpy.ndarray or astropy.table.Table
        The excess surface density in each radial bin specified in the
        precomputation phase. If `return_table` is True, will return a table
        with detailed information for each radial bin. The final result is in
        the column `ds`.

    Raises
    ------
    ValueError
        If boost or random subtraction correction are requested but no random
        catalog is provided.

    """
    result = Table()

    result['rp_min'] = table_l.meta['bins'][:-1]
    result['rp_max'] = table_l.meta['bins'][1:]
    result['rp'] = np.sqrt(result['rp_min'] * result['rp_max'])
    result['ds_raw'] = raw_excess_surface_density(table_l)
    # result['ds'] = raw_excess_surface_density(table_l)
    result['ds'] = result['ds_raw']

    if shear_responsivity_correction:
        result['2R'] = 2 * shear_responsivity_factor(table_l)
        result['ds'] /= result['2R']
    
    if matrix_shear_response_correction:
        result['R_t'] = matrix_shear_response_factor(table_l)
        result['ds'] /= result['R_t']
    
    if additive_bias_correction:
        result['ct_sigma_crit'] = additive_bias(table_l, sigma_crit=True)
        result['ds'] -= result['ct_sigma_crit']

    if scalar_shear_response_correction:
        result['1+m'] = 1 + scalar_shear_response_factor(table_l)
        result['ds'] /= result['1+m']

    if selection_bias_correction:
        result['1+m_sel'] = 1 + scalar_shear_response_factor(
            table_l, selection_bias=True)
        result['ds'] /= result['1+m_sel']

    if photo_z_dilution_correction:
        result['f_bias'] = photo_z_dilution_factor(table_l)
        result['ds'] *= result['f_bias']

    if random_subtraction:
        if table_r is None:
            raise ValueError('Cannot subtract random results without ' +
                                'results from a random catalog.')
        result['ds_r'] = excess_surface_density(
            table_r, photo_z_dilution_correction=photo_z_dilution_correction,
            boost_correction=False,
            additive_bias_correction=additive_bias_correction,
            scalar_shear_response_correction=scalar_shear_response_correction,
            matrix_shear_response_correction=matrix_shear_response_correction,
            shear_responsivity_correction=shear_responsivity_correction,
            selection_bias_correction=selection_bias_correction,
            random_subtraction=False, return_table=False)
        result['ds'] -= result['ds_r']

    if boost_correction:
        if table_r is None:
            raise ValueError('Cannot compute boost factor correction without' +
                                ' results from a random catalog.')
        result['b'] = boost_factor(table_l, table_r)
        result['ds'] *= result['b']

    if pz_boost_correction:
        # if table_r is None:
        #     raise ValueError('Cannot compute p(z) boost factor correction without' +
        #                         ' results from a random catalog.')
        result['b_pz'] = boost_factor_from_pz(table_l)
        if not boost_correction:
            result['ds'] *= result['b_pz']
    
    if not return_table:
        return result['ds'].data
    
    result['n_pairs'] = number_of_pairs(table_l)
    result['z_l'] = mean_lens_redshift(table_l)
    result['z_s'] = mean_source_redshift(table_l)

    return result
