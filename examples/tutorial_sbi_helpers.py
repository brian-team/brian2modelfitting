from brian2 import *


def spike_times(t, x):
    """Return time points in seconds in which a spike occurs.

    Parameters
    ----------
    t : numpy.ndarray
        Time steps in ms.
    x : numpy.ndarray
        Voltage trace in mV.

    Returns
    -------
    numpy.ndarray
        Spike times in seconds.
    """
    x = x.copy()

    ind = where(x < -0.04)
    x[ind] = -0.04
    ind = where(diff(x) < 0)
    x[ind] = -0.04

    ind = where(diff(x) < 0)
    spike_times = array(t)[ind]

    if spike_times.shape[0] > 0:
        spike_times = spike_times[append(1, diff(spike_times)) > 0.5]
    return spike_times / 1000


def plot_traces(t, inp_traces, out_traces, spike_times=None, inf_traces=None):
    """Visualize input current and output voltage traces.

    Parameters
    ----------
    t : numpy.ndarray
        Time steps in ms.
    inp_traces : numpy.ndarray
        Synaptic stimulus data in Amperes.
    inp_traces : numpy.ndarray
        Recorded voltage traces in mV.
    spike_times : numpy.ndarray
        Spike times in s.
    inf_traces : numpy.ndarray, optional
        Sampled voltage traces from the approximated posterior in mV.

    Returns
    -------
    tuple
        Figure and axes.
    """
    if (inp_traces.ndim, out_traces.ndim) != (2, 2):
        inp_traces = inp_traces.reshape(1, -1)
        out_traces = out_traces.reshape(1, -1)
    if inf_traces is not None:
        if inf_traces.ndim != 2:
            inf_traces = inf_traces.reshape(1, -1)
    if spike_times is not None:
        spike_times = spike_times.ravel()
        spike_i = []
        for spike_time in spike_times:
            spike_i.append(where(isclose(spike_time * 1000, t))[0].item())
        spike_v = (out_traces.min(), out_traces.max())

    fig, ax = subplots(2, 1, sharex=True,
                       gridspec_kw={'height_ratios': [3, 1]}, figsize=(6, 3))

    # voltage traces
    ax[0].plot(t, out_traces.T, 'C0', lw=3, label='recordings', zorder=1)
    if inf_traces is not None:
        ax[0].plot(t, inf_traces.T, 'C1--', lw=2, label='sampled traces',
                   zorder=2)
    if spike_times is not None:
        ax[0].vlines(t[spike_i], *spike_v, lw=2, color='C3', label='spikes',
                     zorder=3)
    ax[0].set(ylabel='$V$ [mV]')
    ax[0].legend(loc='upper right')

    # input stimulus current
    ax[1].plot(t, inp_traces.T * 1e9, lw=2, c='k', label='stimulus')
    ax[1].set(xlabel='$t$ [ms]', ylabel='$I$, nA')
    ax[1].legend(loc='upper right')

    tight_layout()
    return fig, ax


def plot_cond_coeff_mat(cond_coeff_mat):
    """Visualize conditional coerrelation matrix.

    Parameters
    ----------
    cond_coeff_mat : numpy.ndarray
        Average conditional correlation matrix.

    Returnss
    -------
    tuple
        Figure and axes.
    """
    fig, ax = subplots(1, 1, figsize=(4, 4))
    im = imshow(cond_coeff_mat, clim=[-1, 1])
    _ = fig.colorbar(im)
    return fig, ax
