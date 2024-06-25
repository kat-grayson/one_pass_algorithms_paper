import xarray as xr
import numpy as np
# importing the TDigest package
from crick import TDigest

"""Functions to calculate the mean, variance, standard deviation and t-digests 
using one-pass algorithms
"""

def two_pass_mean(data_chunk : xr.DataArray):
    """Computes normal mean using numpy two pass

    Arguments
    ----------
    data_chunk : xr.DataArray. incoming data chunk with a time
            dimension greater than 1

    Returns
    ---------
    temp: xr.DataArray two-pass mean over the data chunk
    """
    ax_num = data_chunk.get_axis_num("time")
    temp = np.mean(data_chunk, axis=ax_num, dtype=np.float64, keepdims=True)

    return temp

def two_pass_var(data_chunk : xr.DataArray):
    """Computes normal variance using numpy two pass,
    setting ddof = 1 for sample variance

    Arguments
    ----------
    data_chunk : xr.DataArray. incoming data chunk with a time
            dimension greater than 1

    Returns
    ---------
    temp: xr.DataArray two-pass variance over the data chunk
    """
    ax_num = data_chunk.get_axis_num("time")
    temp = np.var(
        data_chunk, axis=ax_num, dtype=np.float64, keepdims=True, ddof=1
    )

    return temp

def update_mean(data_chunk : xr.DataArray, w : int, n : int,
        mean_cumulative: xr.DataArray):
    """Computes one pass mean with w corresponding to the number
    of timesteps being added. Also updates n.

    Arguments
    ---------
    data_chunk : incoming xr.DataArray data chunk
    w : length of time dimension of incoming data chunk
    n : number of steps that has been added to the statistic
    mean_cumulative: cumulative mean

    Returns
    ---------
    n : updated with w
    mean_cumulative: updated cumulative mean
    """
    n += w

    if w > 1:
        # compute two pass mean first
        data_chunk = two_pass_mean(data_chunk)

    mean_cumulative = (mean_cumulative+ w * \
        (data_chunk - mean_cumulative) / (
        n
    ))

    return mean_cumulative, n

def update_var(
        data_chunk : xr.DataArray, w : int, n : int,
        mean_cumulative: xr.DataArray, var_cumulative: xr.DataArray
    ):
    """Computes one pass variance with w corresponding to
    the number of timesteps being added. It does not update
    the n as this is done in update_mean which update_var
    calls.

    Arguments
    ---------
    data_chunk : incoming xr.DataArray data chunk
    w : length of time dimension of incoming data chunk
    n : current number of samples that have contributed to the variance
    mean_cumulative: cumulative mean of the streamed data
    var_cumulative : cumulative variance of the streamed data


    Returns
    ---------
    var_cumulative: updated cumulative variance. If n < c
            and the statistic is not complete, this is equal to M2
            (see docs). If n == c and enough samples have been
            addded var_cumulativeis divded by (n-1) to get actual variance.
    """

    # storing 'old' mean temporarily
    old_mean = mean_cumulative

    if w == 1 :

        mean_cumulative, n = update_mean(data_chunk, w, n, mean_cumulative)

        var_cumulative = var_cumulative+ w * (
            data_chunk - old_mean) * (data_chunk - mean_cumulative)

    else:
        # two-pass mean
        temp_mean = two_pass_mean(data_chunk)
        # two pass variance
        temp_var = (two_pass_var(data_chunk)) * (w - 1)
        # see paper Mastelini. S

        var_cumulative= (
            var_cumulative
            + temp_var
            + np.square(old_mean - temp_mean)
            * ((n * w) / (n + w))
        )

        mean_cumulative, n = update_mean(data_chunk, w, n, mean_cumulative)

    return n, mean_cumulative, var_cumulative


def init_mean(data_chunk : xr.DataArray):
    
    """Function to initalise an empty numpy array of the same shape
    as the spatial grid, to store the cumulative mean

    Arguments 
    ----------
    data_chunk : incoming xr.DataArray data chunk

    Returns
    ---------
    mean_cumulative : an empty numpy array of the same shape as the
            spatial grid of the data chunk (compressed in the time 
            dimension) for storing the cumulative mean
    """

    data_chunk_tail = data_chunk.tail(time=1)
    shape_data_chunk_tail = np.shape(data_chunk_tail)

    # initialise cumulative mean and cumulative standard deviation
    mean_cumulative = np.zeros(
            shape_data_chunk_tail, dtype=np.float64
        )

    return mean_cumulative

def init_var(data_chunk : xr.DataArray):
    
    """Function to initalise two empty numpy arrays of the same shape
    as the spatial grid, to store the cumulative mean and variance

    Arguments 
    ----------
    data_chunk : incoming xr.DataArray data chunk(or in some cases
            numpy array)

    Returns
    ---------
    mean_cumulative : an empty numpy array of the same shape as the
            spatial grid of the data chunk (compressed in the time 
            dimension) for storing the cumulative mean
    var_cumulative : an empty numpy array of the same shape as the
            spatial grid of the data chunk (compressed in the time 
            dimension) for storing the cumulative variance
    """

    if type(data_chunk) == xr.DataArray:
        data_chunk_tail = data_chunk.tail(time=1)
        shape_data_chunk_tail = np.shape(data_chunk_tail)
    else:
        shape_data_chunk_tail = np.shape(data_chunk)[1:]

    # initialise cumulative mean and cumulative standard deviation
    mean_cumulative = np.zeros(
            shape_data_chunk_tail, dtype=np.float64
        )
    var_cumulative = np.zeros(
            shape_data_chunk_tail, dtype=np.float64
        )

    return mean_cumulative, var_cumulative

def calc_std(var_cumulative : xr.DataArray, n : int):
        """Function to convert the one pass varience into standard
        deviation by dividing by number of samples

        Arguments 
        ----------
        var_cumulative : cumulative variance to be converted.
        n : current number of samples that have contributed to the variance.

        Returns
        ---------
        std_cumulative : standard deviation of the streamed data set.
        """

        # using sample variance NOT population variance
        if (n - 1) != 0:
            var_cumulative = (
                var_cumulative / (n - 1)
            )
            std_cumulative = np.sqrt(var_cumulative)

        return std_cumulative

def init_tdigests(data_chunk : xr.DataArray, compression = 60):
        """Function to initalise a flat array full of empty 
        tDigest objects.

        Arguments 
        ----------
        data_chunk : incoming xr.DataArray data chunk
        compression : compression value for the digests, default 60

        Returns
        ---------
        digest_list : a flat array of of the size of data_source_tail
                full of empty t digest objects with compression = 60
        size_data_source_tail : size of global grid without time dimension
        """

        data_source_tail = data_chunk.tail(time=1)
        size_data_source_tail = np.size(data_source_tail)

        # list of dictionaries for each grid cell, preserves order
        digest_list = [{} for _ in range(
                size_data_source_tail
            )]
        # converts the list into a numpy array which helps with re-sizing time
        digest_list = np.reshape(
                digest_list, size_data_source_tail
            )

        for j in range(size_data_source_tail):
            digest_list[j] = TDigest(compression=compression)

        return digest_list, size_data_source_tail

def update_tdigest(
        data_chunk : xr.DataArray, size_data_source_tail : int,
        digest_list : list, n : int,  w : int = 1
    ):
    """Sequential loop that updates the digest for each grid point.
    If the statistic is not bias correction, it will also update the
    n with the w. For bias correction, this is done in the
    daily means calculation.

    Arguments
    ---------
    data_chunk : incoming xr.DataArray data chunk
    size_data_source_tail : size of global grid without time dimension
    digest_list : a flat array of of the size of data_source_tail
        full of empty t digest objects with compression = 60
    n : current number of samples that have contributed to the statistic
    w : length of time dimension of incoming data chunk

    Returns
    ---------
    digest_list: each digest for each grid cell is updated with
            the new data
    n : updated with w
    """
    if w == 1:

        data_source_values = np.reshape(
            data_chunk.values, size_data_source_tail
        )

        iterable = range(size_data_source_tail)
        # this is looping through every grid cell
        for j in iterable:
            digest_list[j].update(
                data_source_values[j]
            )

    else:
        data_source_values = data_chunk.values.reshape((w, -1))
        iterable = range(size_data_source_tail)
        for j in iterable:
            # using crick
            digest_list[j].update(
                data_source_values[:, j]
            )

    n += w

    return n, digest_list