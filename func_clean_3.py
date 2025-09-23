import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter as sgolay
from obspy.signal.detrend import spline
from obspy import read
from scipy.signal import welch
from matplotlib.patches import Polygon
import matplotlib
from datetime import datetime, timedelta
from obspy.signal.trigger import classic_sta_lta


def preprocess(tr, detrend=True, bandpass=True, order=3, dspline=500, freqmin=0.5, freqmax=10):
    for i in range(len(tr)):
        tr[i].data = tr[i].data.astype('float64')
        if detrend:
            tr[i].detrend("spline", order=order, dspline=dspline)
    if bandpass:
        tr.filter("bandpass", freqmin=freqmin, freqmax=freqmax)
    return tr

def calculate_vh_data(trdat, windows, shift, cft_max=1.5, cft_min=0.5):
    if windows < shift:
        print('Ensure windows > shift!')
        return
    
    tr = trdat.copy()
    cft = classic_sta_lta(tr[0].data, int(windows * tr[0].stats['sampling_rate']), int(windows * tr[0].stats['sampling_rate'] * 3))
    cftN = classic_sta_lta(tr[1].data, int(windows * tr[1].stats['sampling_rate']), int(windows * tr[1].stats['sampling_rate'] * 3))
    cftE = classic_sta_lta(tr[2].data, int(windows * tr[2].stats['sampling_rate']), int(windows * tr[2].stats['sampling_rate'] * 3))
    
    iseg = 0
    iseg_count = 0

    t_start = max(tr[0].stats['starttime'], tr[1].stats['starttime'], tr[2].stats['starttime'])
    t_end = min(tr[0].stats['endtime'], tr[1].stats['endtime'], tr[2].stats['endtime']) 

    t1 = t_start + iseg * timedelta(seconds=shift)
    t2 = t1 + timedelta(seconds=windows)

    VperH_list = []
    N_data = 0
    valid_windows = []

    while t1 < t_end - timedelta(seconds=windows):
        cft_i_start = int((t1 - tr[0].stats['starttime']) * tr[0].stats['sampling_rate'])
        cft_i_end = int((t2 - tr[0].stats['starttime']) * tr[0].stats['sampling_rate']) 

        if (max(cft[cft_i_start:cft_i_end]) > cft_max or min(cft[cft_i_start:cft_i_end]) < cft_min or
            max(cftN[cft_i_start:cft_i_end]) > cft_max or min(cftN[cft_i_start:cft_i_end]) < cft_min or
            max(cftE[cft_i_start:cft_i_end]) > cft_max or min(cftE[cft_i_start:cft_i_end]) < cft_min):
            iseg += 1
            t1 = tr[0].stats['starttime'] + iseg * timedelta(seconds=shift)
            t2 = t1 + timedelta(seconds=windows)
            continue
        valid_windows.append((t1, t2))

        trc = tr.copy()
        trc.trim(t1, t2)

        N_data = len(trc[0].data)
        fftZ = (1 / N_data) * abs(np.fft.fft(trc[0].data))
        fftN = (1 / N_data) * abs(np.fft.fft(trc[1].data))
        fftE = (1 / N_data) * abs(np.fft.fft(trc[2].data))

        VperH = np.sqrt((fftZ ** 2) / (fftN ** 2 + fftE ** 2))
        VperH = sgolay(VperH, 100, 2)

        VperH_list.append(VperH[:N_data // 2])

        iseg += 1
        iseg_count += 1
        t1 = tr[0].stats['starttime'] + iseg * timedelta(seconds=shift)
        t2 = t1 + timedelta(seconds=windows)

    if iseg_count == 0 or N_data == 0:
        return None
    
    samplingrate = tr[0].stats['sampling_rate']
    VperH_array = np.array(VperH_list)

    freq = np.fft.fftfreq(N_data, 1 / samplingrate)
    freq = freq[:N_data // 2]

    return {
        "freq": freq,
        "VperH_array": VperH_array,
        "median": np.median(VperH_array, axis=0),
        "mean": np.mean(VperH_array, axis=0),
        "q1": np.quantile(VperH_array, 0.25, axis=0),
        "q3": np.quantile(VperH_array, 0.75, axis=0),
        "window_count": iseg_count,
        "window_len": windows,
        "shift_len": shift,
        "valid_windows": valid_windows
    }

def calculateVHwithSTALTA(trdat, windows, shift, savename, fmin=1, fmax=4, fmin_plot=0.5, fmax_plot=10, cft_max=1.5, cft_min=0.5):
    
    results = calculate_vh_data(trdat, windows, shift, cft_max=cft_max, cft_min=cft_min)

    if not results:
        print("No valid windows found.")
        return np.nan, np.nan
    
    fig, ax = plt.subplots()

    for vh_curve in results["VperH_array"]:
        ax.plot(results["freq"], vh_curve, alpha=0.1, color='gray') 

    ax.plot(results["freq"], results["median"], color='black', linewidth=2, label='Median')
    ax.plot(results["q1"], color='black', linewidth=1, linestyle='dashed')
    ax.plot(results["q3"], color='black', linewidth=1, linestyle='dashed')
    ax.plot(results["freq"], results["mean"], color='red', linewidth=2, label='Mean')

    ax.set_ylim(0, 2)
    ax.set_xlim(fmin_plot, fmax_plot)
    ax.set_xlabel('Frequency (Hz)')
    ax.set_ylabel('V/H')
    ax.set_title(f'V/H Ratio - {results["window_count"]} Windows')
    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.5)

    fig.savefig(savename + '_VH.png')
    plt.close(fig)

    finterp = np.arange(fmin, fmax, 0.05)
    vinterp = np.interp(finterp, results["freq"], results["median"])
    return max(vinterp), finterp[vinterp.argmax()]

def calculatePSD_STALTA(trdat, windows, shift, savename, fmin=1, fmax=4, fmin_plot=0.5, fmax_plot=10, cft_max=1.5, cft_min=0.5):
    if windows < shift:
        print('Ensure windows > shift!')
        return
    tr=trdat.copy()
    cft = classic_sta_lta(tr[0].data, windows * tr[0].stats['sampling_rate'], windows * tr[0].stats['sampling_rate'] * 3)
    cftN = classic_sta_lta(tr[1].data, windows * tr[1].stats['sampling_rate'], windows * tr[1].stats['sampling_rate'] * 3)
    cftE = classic_sta_lta(tr[2].data, windows * tr[2].stats['sampling_rate'], windows * tr[2].stats['sampling_rate'] * 3)

    fig_seis = plt.figure()
    ax_seis = fig_seis.add_subplot(2, 1, 1)
    ax_seis.plot(tr[0].times("matplotlib"), tr[0].data, 'k')
    ax_seis.xaxis_date()
    ax_seis = fig_seis.add_subplot(2, 1, 2)
    ax_seis.plot(tr[0].times("matplotlib"), cft, 'k')
    ax_seis.xaxis_date()

    figZ, axZ = plt.subplots()
    figN, axN = plt.subplots()
    figE, axE = plt.subplots()

    iseg = 0
    t_start = max(tr[0].stats['starttime'], tr[1].stats['starttime'], tr[2].stats['starttime'])
    t_end = min(tr[0].stats['endtime'], tr[1].stats['endtime'], tr[2].stats['endtime'])
    t1 = t_start + iseg * timedelta(seconds=shift)
    t2 = t1 + timedelta(seconds=windows)

    fftZ_sum = np.zeros(int(((windows * tr[0].stats['sampling_rate']) + 1) // 2))
    fftN_sum = np.zeros_like(fftZ_sum)
    fftE_sum = np.zeros_like(fftZ_sum)
    iseg_count = 0
    color_iter = plt.cm.rainbow(np.linspace(0, 1, int((t_end - t_start) / shift)))

    while t1 < t_end - windows:
        cft_i_start = int((t1 - tr[0].stats['starttime']) * tr[0].stats['sampling_rate'])
        cft_i_end = int((t2 - tr[0].stats['starttime']) * tr[0].stats['sampling_rate'])

        if (max(cft[cft_i_start:cft_i_end]) > cft_max or min(cft[cft_i_start:cft_i_end]) < cft_min or
            max(cftN[cft_i_start:cft_i_end]) > cft_max or min(cftN[cft_i_start:cft_i_end]) < cft_min or
            max(cftE[cft_i_start:cft_i_end]) > cft_max or min(cftE[cft_i_start:cft_i_end]) < cft_min):
            iseg += 1
            t1 = tr[0].stats['starttime'] + iseg * timedelta(seconds=shift)
            t2 = t1 + timedelta(seconds=windows)
            continue

        ax_seis.axvspan(t1, t2, color=color_iter[iseg], alpha=0.3)
        trc = tr.copy()
        trc.trim(t1, t2)

        N = len(trc[0].data)
        fftZ = (1 / N) * abs(np.fft.fft(trc[0].data)) ** 2
        fftN = (1 / N) * abs(np.fft.fft(trc[1].data)) ** 2
        fftE = (1 / N) * abs(np.fft.fft(trc[2].data)) ** 2
        fftZ, fftN, fftE = fftZ[:N // 2], fftN[:N // 2], fftE[:N // 2]

        samplingrate = tr[0].stats['sampling_rate']
        freq = np.fft.fftfreq(N, 1 / samplingrate)[:N // 2]

        axZ.plot(freq, 10 * np.log10(fftZ), alpha=0.1, color=color_iter[iseg])
        axN.plot(freq, 10 * np.log10(fftN), alpha=0.1, color=color_iter[iseg])
        axE.plot(freq, 10 * np.log10(fftE), alpha=0.1, color=color_iter[iseg])

        fftZ_sum += fftZ
        fftN_sum += fftN
        fftE_sum += fftE
        iseg += 1
        iseg_count += 1
        t1 = tr[0].stats['starttime'] + iseg * timedelta(seconds=shift)
        t2 = t1 + timedelta(seconds=windows)

    fig_seis.savefig(savename + '_seis.png')
    plt.close(fig_seis)

    if iseg_count == 0:
        for ax in [axZ, axN, axE]:
            ax.set_xlim(fmin_plot, fmax_plot)
            ax.set_ylim(0, 150)
        figZ.savefig(savename + '_Z.png')
        figN.savefig(savename + '_N.png')
        figE.savefig(savename + '_E.png')
        plt.close(figZ)
        plt.close(figN)
        plt.close(figE)
        return np.nan, np.nan, np.nan

    fftZ_array = sgolay(10 * np.log10(fftZ_sum / iseg_count), 100, 2)
    fftN_array = sgolay(10 * np.log10(fftN_sum / iseg_count), 100, 2)
    fftE_array = sgolay(10 * np.log10(fftE_sum / iseg_count), 100, 2)

    freq = np.fft.fftfreq(N, 1 / samplingrate)[:N // 2]

    axZ.plot(freq, fftZ_array, color='black', linewidth=2)
    axZ.set_xlim(fmin_plot, fmax_plot)
    axZ.set_ylim(0, 150)

    axN.plot(freq, fftN_array, color='black', linewidth=2)
    axN.set_xlim(fmin_plot, fmax_plot)
    axN.set_ylim(0, 150)

    axE.plot(freq, fftE_array, color='black', linewidth=2)
    axE.set_xlim(fmin_plot, fmax_plot)
    axE.set_ylim(0, 150)

    finterp = np.arange(fmin, fmax, 0.05)
    Zinterp = np.interp(finterp, freq, fftZ_array)

    # verts = [(1, 20), *zip(finterp[(finterp >= 1) & (finterp <= 4)], Zinterp[(finterp >= 1) & (finterp <= 4)]), (4, 20)]
    # poly = Polygon(verts, facecolor='0.5', edgecolor='0.5')
    # psd_iz = np.trapz(Zinterp[(finterp >= 1) & (finterp <= 4)], x=finterp[(finterp >= 1) & (finterp <= 4)])
    # axZ.add_patch(poly)

    figZ.savefig(savename + '_Z.png')
    figN.savefig(savename + '_N.png')
    figE.savefig(savename + '_E.png')
    plt.close(figZ)
    plt.close(figN)
    plt.close(figE)

    from PIL import Image

    imgZ = Image.open(savename + '_Z.png')
    imgN = Image.open(savename + '_N.png')
    imgE = Image.open(savename + '_E.png')

    # Pastikan semua ukuran sama
    width, height = imgZ.size
    combined = Image.new('RGB', (width, height * 3))
    combined.paste(imgZ, (0, 0))
    combined.paste(imgN, (0, height))
    combined.paste(imgE, (0, height * 2))

    combined.save(savename + '_ZNE.png')

    # return max(Zinterp), finterp[Zinterp.argmax()], psd_iz
    return max(Zinterp), finterp[Zinterp.argmax()]

def calculatePSD_STALTA_scaling(trdat, windows, shift, savename, fmin=1, fmax=4, fmin_plot=0.5, fmax_plot=10, cft_max=1.5, cft_min=0.5):
    if windows < shift:
        print('Ensure windows > shift!')
        return
    tr=trdat.copy()
    cft = classic_sta_lta(tr[0].data, windows * tr[0].stats['sampling_rate'], windows * tr[0].stats['sampling_rate'] * 3)
    cftN = classic_sta_lta(tr[1].data, windows * tr[1].stats['sampling_rate'], windows * tr[1].stats['sampling_rate'] * 3)
    cftE = classic_sta_lta(tr[2].data, windows * tr[2].stats['sampling_rate'], windows * tr[2].stats['sampling_rate'] * 3)

    fig_seis = plt.figure()
    ax_seis = fig_seis.add_subplot(2, 1, 1)
    ax_seis.plot(tr[0].times("matplotlib"), tr[0].data, 'k')
    ax_seis.xaxis_date()
    ax_seis = fig_seis.add_subplot(2, 1, 2)
    ax_seis.plot(tr[0].times("matplotlib"), cft, 'k')
    ax_seis.xaxis_date()

    figZ, axZ = plt.subplots()
    figN, axN = plt.subplots()
    figE, axE = plt.subplots()

    iseg = 0
    t_start = max(tr[0].stats['starttime'], tr[1].stats['starttime'], tr[2].stats['starttime'])
    t_end = min(tr[0].stats['endtime'], tr[1].stats['endtime'], tr[2].stats['endtime'])
    t1 = t_start + iseg * timedelta(seconds=shift)
    t2 = t1 + timedelta(seconds=windows)

    fftZ_sum = np.zeros(int(((windows * tr[0].stats['sampling_rate']) + 1) // 2))
    fftN_sum = np.zeros_like(fftZ_sum)
    fftE_sum = np.zeros_like(fftZ_sum)
    iseg_count = 0
    color_iter = plt.cm.rainbow(np.linspace(0, 1, int((t_end - t_start) / shift)))

    while t1 < t_end - windows:
        cft_i_start = int((t1 - tr[0].stats['starttime']) * tr[0].stats['sampling_rate'])
        cft_i_end = int((t2 - tr[0].stats['starttime']) * tr[0].stats['sampling_rate'])

        if (max(cft[cft_i_start:cft_i_end]) > cft_max or min(cft[cft_i_start:cft_i_end]) < cft_min or
            max(cftN[cft_i_start:cft_i_end]) > cft_max or min(cftN[cft_i_start:cft_i_end]) < cft_min or
            max(cftE[cft_i_start:cft_i_end]) > cft_max or min(cftE[cft_i_start:cft_i_end]) < cft_min):
            iseg += 1
            t1 = tr[0].stats['starttime'] + iseg * timedelta(seconds=shift)
            t2 = t1 + timedelta(seconds=windows)
            continue

        ax_seis.axvspan(t1, t2, color=color_iter[iseg], alpha=0.3)
        trc = tr.copy()
        trc.trim(t1, t2)

        N = len(trc[0].data)
        fftZ = ((1 / N) * abs(np.fft.fft(trc[0].data)) ** 2)*(10**17)
        fftN = ((1 / N) * abs(np.fft.fft(trc[1].data)) ** 2)*(10**17)
        fftE = ((1 / N) * abs(np.fft.fft(trc[2].data)) ** 2)*(10**17)
        fftZ, fftN, fftE = fftZ[:N // 2], fftN[:N // 2], fftE[:N // 2]

        samplingrate = tr[0].stats['sampling_rate']
        freq = np.fft.fftfreq(N, 1 / samplingrate)[:N // 2]

        axZ.plot(freq, 10 * np.log10(fftZ), alpha=0.1, color=color_iter[iseg])
        axN.plot(freq, 10 * np.log10(fftN), alpha=0.1, color=color_iter[iseg])
        axE.plot(freq, 10 * np.log10(fftE), alpha=0.1, color=color_iter[iseg])

        fftZ_sum += fftZ
        fftN_sum += fftN
        fftE_sum += fftE
        iseg += 1
        iseg_count += 1
        t1 = tr[0].stats['starttime'] + iseg * timedelta(seconds=shift)
        t2 = t1 + timedelta(seconds=windows)

    fig_seis.savefig(savename + '_seis.png')
    plt.close(fig_seis)

    if iseg_count == 0:
        for ax in [axZ, axN, axE]:
            ax.set_xlim(fmin_plot, fmax_plot)
            ax.set_ylim(0, 150)
        figZ.savefig(savename + '_Z.png')
        figN.savefig(savename + '_N.png')
        figE.savefig(savename + '_E.png')
        plt.close(figZ)
        plt.close(figN)
        plt.close(figE)
        return np.nan, np.nan, np.nan

    fftZ_array = sgolay(10 * np.log10(fftZ_sum / iseg_count), 100, 2)
    fftN_array = sgolay(10 * np.log10(fftN_sum / iseg_count), 100, 2)
    fftE_array = sgolay(10 * np.log10(fftE_sum / iseg_count), 100, 2)

    freq = np.fft.fftfreq(N, 1 / samplingrate)[:N // 2]

    axZ.plot(freq, fftZ_array, color='black', linewidth=2)
    axZ.set_xlim(fmin_plot, fmax_plot)
    axZ.set_ylim(0, 150)

    axN.plot(freq, fftN_array, color='black', linewidth=2)
    axN.set_xlim(fmin_plot, fmax_plot)
    axN.set_ylim(0, 150)

    axE.plot(freq, fftE_array, color='black', linewidth=2)
    axE.set_xlim(fmin_plot, fmax_plot)
    axE.set_ylim(0, 150)

    finterp = np.arange(fmin, fmax, 0.05)
    Zinterp = np.interp(finterp, freq, fftZ_array)

    # verts = [(1, 20), *zip(finterp[(finterp >= 1) & (finterp <= 4)], Zinterp[(finterp >= 1) & (finterp <= 4)]), (4, 20)]
    # poly = Polygon(verts, facecolor='0.5', edgecolor='0.5')
    # psd_iz = np.trapz(Zinterp[(finterp >= 1) & (finterp <= 4)], x=finterp[(finterp >= 1) & (finterp <= 4)])
    # axZ.add_patch(poly)

    figZ.savefig(savename + '_Z.png')
    figN.savefig(savename + '_N.png')
    figE.savefig(savename + '_E.png')
    plt.close(figZ)
    plt.close(figN)
    plt.close(figE)

    from PIL import Image

    imgZ = Image.open(savename + '_Z.png')
    imgN = Image.open(savename + '_N.png')
    imgE = Image.open(savename + '_E.png')

    # Pastikan semua ukuran sama
    width, height = imgZ.size
    combined = Image.new('RGB', (width, height * 3))
    combined.paste(imgZ, (0, 0))
    combined.paste(imgN, (0, height))
    combined.paste(imgE, (0, height * 2))

    combined.save(savename + '_ZNE.png')

    # return max(Zinterp), finterp[Zinterp.argmax()], psd_iz
    return max(Zinterp), finterp[Zinterp.argmax()]

def calculate_polar(trdat, windows, shift, savename, fmin=1, fmax=4, fmin_plot=0.5, fmax_plot=10, cft_max=1.5, cft_min=0.5):
    if windows < shift:
        print('Ensure windows > shift!')
        return

    tr = trdat.copy()
    t_start = max(tr[0].stats['starttime'], tr[1].stats['starttime'], tr[2].stats['starttime'])
    t_end = min(tr[0].stats['endtime'], tr[1].stats['endtime'], tr[2].stats['endtime'])
    tr.trim(t_start, t_end)

    cftZ = classic_sta_lta(tr[0].data, windows * tr[0].stats['sampling_rate'], windows * tr[0].stats['sampling_rate'] * 3)
    cftN = classic_sta_lta(tr[1].data, windows * tr[1].stats['sampling_rate'], windows * tr[1].stats['sampling_rate'] * 3)
    cftE = classic_sta_lta(tr[2].data, windows * tr[2].stats['sampling_rate'], windows * tr[2].stats['sampling_rate'] * 3)

    idx_sel = np.logical_and.reduce([
        np.logical_and(cftZ > 0.5, cftZ < 1.5),
        np.logical_and(cftN > 0.5, cftN < 1.5),
        np.logical_and(cftE > 0.5, cftE < 1.5)
    ])

    dataZ, dataN, dataE = tr[0].data[idx_sel], tr[1].data[idx_sel], tr[2].data[idx_sel]
    N = len(dataZ)
    Czz = np.sum(dataZ * dataZ) / N
    Cnn = np.sum(dataN * dataN) / N
    Cee = np.sum(dataE * dataE) / N
    Czn = np.sum(dataZ * dataN) / N
    Cze = np.sum(dataZ * dataE) / N
    Cne = np.sum(dataN * dataE) / N

    cov_max = np.array([[Cee, Cne, Cze], [Cne, Cnn, Czn], [Cze, Czn, Czz]])
    eigenvalues, eigenvectors = np.linalg.eig(cov_max)
    idx = eigenvalues.argsort()[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]

    L_all = 1 - ((eigenvalues[1] + eigenvalues[2]) / (2 * eigenvalues[0]))
    eigenvectors_max = eigenvectors[:, 0]
    dip_all = abs(np.rad2deg(np.arctan(eigenvectors_max[2] / np.linalg.norm(eigenvectors_max[:2]))))
    azimuth_all = np.rad2deg(np.arctan2(eigenvectors_max[1], eigenvectors_max[0]))
    eig_all = eigenvalues[0]

    iseg = 0
    t1 = t_start + iseg * timedelta(seconds=shift)
    t2 = t1 + timedelta(seconds=windows)

    L_list, dip_list, azimuth_list, eig_list, t_list = [], [], [], [], []

    while t1 < t_end - windows:
        cft_i_start = int((t1 - tr[0].stats['starttime']) * tr[0].stats['sampling_rate'])
        cft_i_end = int((t2 - tr[0].stats['starttime']) * tr[0].stats['sampling_rate'])

        if (max(cftZ[cft_i_start:cft_i_end]) > cft_max or min(cftZ[cft_i_start:cft_i_end]) < cft_min or
            max(cftN[cft_i_start:cft_i_end]) > cft_max or min(cftN[cft_i_start:cft_i_end]) < cft_min or
            max(cftE[cft_i_start:cft_i_end]) > cft_max or min(cftE[cft_i_start:cft_i_end]) < cft_min):
            iseg += 1
            t1 = tr[0].stats['starttime'] + iseg * timedelta(seconds=shift)
            t2 = t1 + timedelta(seconds=windows)
            continue

        trc = tr.copy()
        trc.trim(t1, t2)


        N = len(trc[0].data)
        Czz = np.sum(trc[0].data ** 2) / N
        Cnn = np.sum(trc[1].data ** 2) / N
        Cee = np.sum(trc[2].data ** 2) / N
        Czn = np.sum(trc[0].data * trc[1].data) / N
        Cze = np.sum(trc[0].data * trc[2].data) / N
        Cne = np.sum(trc[1].data * trc[2].data) / N

        cov_mat = np.array([[Cee, Cne, Cze], [Cne, Cnn, Czn], [Cze, Czn, Czz]])
        eigenvalues, eigenvectors = np.linalg.eig(cov_mat)
        idx = eigenvalues.argsort()[::-1]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]

        L_i = 1 - ((eigenvalues[1] + eigenvalues[2]) / (2 * eigenvalues[0]))
        eigenvectors_max = eigenvectors[:, 0]
        dip_i = abs(np.rad2deg(np.arctan(eigenvectors_max[2] / np.linalg.norm(eigenvectors_max[:2]))))
        azimuth_i = np.rad2deg(np.arctan2(eigenvectors_max[1], eigenvectors_max[0]))
        eig_i = eigenvalues[0]

        L_list.append(L_i)
        dip_list.append(dip_i)
        azimuth_list.append(azimuth_i)
        eig_list.append(eig_i)
        t_list.append(iseg * shift + windows / 2)

        iseg += 1
        t1 = tr[0].stats['starttime'] + iseg * timedelta(seconds=shift)
        t2 = t1 + timedelta(seconds=windows)

    fig = plt.figure(figsize=(7, 5))
    ax_L = fig.add_subplot(2, 2, 1)
    ax_L.scatter(t_list, L_list, color='black')
    ax_L.axhline(L_all, color='black')
    ax_L.axhline(np.mean(L_list), color='red', linestyle='dashed')
    ax_L.set_ylabel('Rectilinearity')
    ax_L.set_ylim(0, 1)

    ax_dip = fig.add_subplot(2, 2, 2)
    ax_dip.scatter(t_list, dip_list, color='black')
    ax_dip.axhline(dip_all, color='black')
    ax_dip.axhline(np.mean(dip_list), color='red', linestyle='dashed')
    ax_dip.set_ylabel('Dip (°)')
    ax_dip.set_ylim(0, 90)

    ax_azimuth = fig.add_subplot(2, 2, 3)
    ax_azimuth.scatter(t_list, azimuth_list, color='black')
    ax_azimuth.axhline(azimuth_all, color='black')
    ax_azimuth.axhline(np.mean(azimuth_list), color='red', linestyle='dashed')
    ax_azimuth.set_ylabel('Azimuth (°)')
    ax_azimuth.set_ylim(-50, 50)

    ax_eig = fig.add_subplot(2, 2, 4)
    ax_eig.scatter(t_list, eig_list, color='black')
    ax_eig.axhline(eig_all, color='black')
    ax_eig.axhline(np.mean(eig_list), color='red', linestyle='dashed')
    ax_eig.set_ylabel('Largest Eigenvalue')

    fig.savefig(savename + ".png")
    plt.close(fig)

    return L_all, dip_all, azimuth_all, eig_all

def calquality_STALTA(trdat, windows, shift, savename, fmin=1, fmax=4, fmin_plot=0.5, fmax_plot=10, cft_max=1.5, cft_min=0.5):
    if windows < shift:
        print('Ensure windows > shift!')
        return

    tr = trdat.copy()

    cft = classic_sta_lta(tr[0].data, windows * tr[0].stats['sampling_rate'], windows * tr[0].stats['sampling_rate'] * 3)
    cftN = classic_sta_lta(tr[1].data, windows * tr[1].stats['sampling_rate'], windows * tr[1].stats['sampling_rate'] * 3)
    cftE = classic_sta_lta(tr[2].data, windows * tr[2].stats['sampling_rate'], windows * tr[2].stats['sampling_rate'] * 3)

    fig_seis = plt.figure()
    ax_seis = fig_seis.add_subplot(2, 1, 1)
    ax_seis.plot(tr[0].times("matplotlib"), tr[0].data, 'k')
    ax_seis.xaxis_date()
    ax_seis = fig_seis.add_subplot(2, 1, 2)
    ax_seis.plot(tr[0].times("matplotlib"), cft, 'k')
    ax_seis.xaxis_date()

    iseg = 0
    t_start = max(tr[0].stats['starttime'], tr[1].stats['starttime'], tr[2].stats['starttime'])
    t_end = min(tr[0].stats['endtime'], tr[1].stats['endtime'], tr[2].stats['endtime'])
    t1 = t_start + iseg * timedelta(seconds=shift)
    t2 = t1 + timedelta(seconds=windows)

    iseg_count = 0
    color_iter = plt.cm.rainbow(np.linspace(0, 1, int((t_end - t_start) / shift)))

    while t1 < t_end - windows:
        cft_i_start = int((t1 - tr[0].stats['starttime']) * tr[0].stats['sampling_rate'])
        cft_i_end = int((t2 - tr[0].stats['starttime']) * tr[0].stats['sampling_rate'])

        if (max(cft[cft_i_start:cft_i_end]) > cft_max or min(cft[cft_i_start:cft_i_end]) < cft_min or
            max(cftN[cft_i_start:cft_i_end]) > cft_max or min(cftN[cft_i_start:cft_i_end]) < cft_min or
            max(cftE[cft_i_start:cft_i_end]) > cft_max or min(cftE[cft_i_start:cft_i_end]) < cft_min):
            iseg += 1
            t1 = tr[0].stats['starttime'] + iseg * timedelta(seconds=shift)
            t2 = t1 + timedelta(seconds=windows)
            continue

        ax_seis.axvspan(t1, t2, color=color_iter[iseg], alpha=0.3)

        iseg += 1
        iseg_count += 1
        t1 = tr[0].stats['starttime'] + iseg * timedelta(seconds=shift)
        t2 = t1 + timedelta(seconds=windows)

    fig_seis.savefig(savename + '_cftwindow.png')
    plt.close(fig_seis)

    percent = (iseg_count / iseg) * 100
    group = 'good'
    if percent < 60:
        group = 'moderate'
    if percent < 30:
        group = 'poor'

    return percent, group

def plot_spec(trdat, savename, noverlap=128):
    tr=trdat.copy()
    fig_spec = plt.figure()
    axZ = fig_spec.add_subplot(3, 1, 1)
    axN = fig_spec.add_subplot(3, 1, 2, sharex=axZ)
    axE = fig_spec.add_subplot(3, 1, 3, sharex=axZ)

    N = len(tr[0].data)
    fftZ = (1 / N) * (abs(np.fft.fft(tr[0].data)) ** 2)
    fftN = (1 / N) * (abs(np.fft.fft(tr[1].data)) ** 2)
    fftE = (1 / N) * (abs(np.fft.fft(tr[2].data)) ** 2)
    fftZ = fftZ[:N // 2]
    fftN = fftN[:N // 2]
    fftE = fftE[:N // 2]

    samplingrate = tr[0].stats['sampling_rate']
    freq = np.fft.fftfreq(N, 1 / samplingrate)
    freq = freq[:N // 2]

    axZ.plot(freq, 10 * np.log10(fftZ))
    axN.plot(freq, 10 * np.log10(fftN))
    axE.plot(freq, 10 * np.log10(fftE))

    for ax in [axZ, axN, axE]:
        ax.set_xscale('log')

    axZ.set_title("Z")
    axN.set_title("N")
    axE.set_title("E")
    fig_spec.suptitle("frequency spectrum")
    axE.set_xlabel('frequency')
    axZ.set_xlim(0.1, 100)
    fig_spec.savefig(savename + '_spectrum.png')

    fig_specgram = plt.figure()
    axsZ = fig_specgram.add_subplot(3, 1, 1)
    axsN = fig_specgram.add_subplot(3, 1, 2, sharex=axsZ)
    axsE = fig_specgram.add_subplot(3, 1, 3, sharex=axsZ)

    axsZ.specgram(tr[0].data, int(tr[0].stats['sampling_rate']), noverlap=noverlap)
    axsN.specgram(tr[1].data, int(tr[1].stats['sampling_rate']), noverlap=noverlap)
    axsE.specgram(tr[2].data, int(tr[2].stats['sampling_rate']), noverlap=noverlap)

    axsZ.set_title("Z")
    axsN.set_title("N")
    axsE.set_title("E")
    fig_specgram.savefig(savename + '_specgram.png')

    fig_seis = plt.figure()
    ax_Z = fig_seis.add_subplot(3, 1, 1)
    ax_Z.plot(tr[0].times("matplotlib"), tr[0].data, 'k')
    ax_Z.xaxis_date()
    
    ax_N = fig_seis.add_subplot(3, 1, 2, sharex=ax_Z)
    ax_N.plot(tr[1].times("matplotlib"), tr[1].data, 'k')
    ax_N.xaxis_date()

    ax_E = fig_seis.add_subplot(3, 1, 3, sharex=ax_Z)
    ax_E.plot(tr[2].times("matplotlib"), tr[2].data, 'k')
    ax_E.xaxis_date()

    ax_Z.set_title("Z")
    ax_N.set_title("N")
    ax_E.set_title("E")

    fig_seis.savefig(savename + '_waveform.png')

def plot_window(trdat, windows, shift, savename, fmin=1, fmax=4, fmin_plot=0.5, fmax_plot=10, cft_max=1.5, cft_min=0.5):
    if windows < shift:
        print('Ensure windows > shift!')
        return

    tr=trdat.copy()
    cft = classic_sta_lta(tr[0].data, windows * tr[0].stats['sampling_rate'], windows * tr[0].stats['sampling_rate'] * 3)
    cftN = classic_sta_lta(tr[1].data, windows * tr[1].stats['sampling_rate'], windows * tr[1].stats['sampling_rate'] * 3)
    cftE = classic_sta_lta(tr[2].data, windows * tr[2].stats['sampling_rate'], windows * tr[2].stats['sampling_rate'] * 3)

    fig_seis = plt.figure()
    ax_seisZ = fig_seis.add_subplot(3, 1, 1)
    ax_seisZ.plot(tr[0].times("matplotlib"), tr[0].data, 'k')
    ax_seisZ.xaxis_date()
    ax_seisZ.set_ylim(-0.0002,0.0002)
    ax_seisN = fig_seis.add_subplot(3, 1, 2)
    ax_seisN.plot(tr[1].times("matplotlib"), tr[1].data, 'k')
    ax_seisN.xaxis_date()
    ax_seisN.set_ylim(-0.0002,0.0002)
    ax_seisE = fig_seis.add_subplot(3, 1, 3)
    ax_seisE.plot(tr[2].times("matplotlib"), tr[2].data, 'k')
    ax_seisE.xaxis_date()
    ax_seisE.set_ylim(-0.0002,0.0002)
    #ax_seis = fig_seis.add_subplot(2, 1, 2)
    #ax_seis.plot(tr[0].times("matplotlib"), cft, 'k')
    #ax_seis.xaxis_date()

    iseg = 0
    t_start = max(tr[0].stats['starttime'], tr[1].stats['starttime'], tr[2].stats['starttime'])
    t_end = min(tr[0].stats['endtime'], tr[1].stats['endtime'], tr[2].stats['endtime'])
    t1 = t_start + iseg * timedelta(seconds=shift)
    t2 = t1 + timedelta(seconds=windows)

    iseg_count = 0
    color_iter = plt.cm.rainbow(np.linspace(0, 1, int((t_end - t_start) / shift)))

    while t1 < t_end - windows:
        cft_i_start = int((t1 - tr[0].stats['starttime']) * tr[0].stats['sampling_rate'])
        cft_i_end = int((t2 - tr[0].stats['starttime']) * tr[0].stats['sampling_rate'])

        if (max(cft[cft_i_start:cft_i_end]) > cft_max or min(cft[cft_i_start:cft_i_end]) < cft_min or
            max(cftN[cft_i_start:cft_i_end]) > cft_max or min(cftN[cft_i_start:cft_i_end]) < cft_min or
            max(cftE[cft_i_start:cft_i_end]) > cft_max or min(cftE[cft_i_start:cft_i_end]) < cft_min):
            iseg += 1
            t1 = tr[0].stats['starttime'] + iseg * timedelta(seconds=shift)
            t2 = t1 + timedelta(seconds=windows)
            continue

        ax_seisZ.axvspan(t1, t2, color=color_iter[iseg], alpha=0.3)
        ax_seisN.axvspan(t1, t2, color=color_iter[iseg], alpha=0.3)
        ax_seisE.axvspan(t1, t2, color=color_iter[iseg], alpha=0.3)

        iseg += 1
        iseg_count += 1
        t1 = tr[0].stats['starttime'] + iseg * timedelta(seconds=shift)
        t2 = t1 + timedelta(seconds=windows)

    fig_seis.savefig(savename + '_selwindow.png')
    plt.close(fig_seis)

    percent = (iseg_count / iseg) * 100
    group = 'good'
    if percent < 60:
        group = 'moderate'
    if percent < 30:
        group = 'poor'

    return percent, group

def calculateVH_STALTA_dip(trdat, windows, shift, savename, fmin=1, fmax=4, fmin_plot=0.5, fmax_plot=10, cft_max=1.5, cft_min=0.5, dip_tres=45):
    if windows < shift:
        print('Ensure windows > shift!')
        return
    tr=trdat.copy()
    cftZ = classic_sta_lta(tr[0].data, windows * tr[0].stats['sampling_rate'], windows * tr[0].stats['sampling_rate'] * 3)
    cftN = classic_sta_lta(tr[1].data, windows * tr[1].stats['sampling_rate'], windows * tr[1].stats['sampling_rate'] * 3)
    cftE = classic_sta_lta(tr[2].data, windows * tr[2].stats['sampling_rate'], windows * tr[2].stats['sampling_rate'] * 3)

    iskip_sum = np.logical_and.reduce([np.logical_and(cftZ > 0.9999, cftZ < 1.0001), np.logical_and(tr[0].data > -0.00000001, tr[0].data < 0.00000001)])

    fig_seis = plt.figure()
    ax_seis = fig_seis.add_subplot(2, 1, 1)
    ax_seis.plot(tr[0].times("matplotlib"), tr[0].data, 'k')
    ax_seis.xaxis_date()
    ax_seis = fig_seis.add_subplot(2, 1, 2)
    ax_seis.plot(tr[0].times("matplotlib"), cftZ, 'k')
    ax_seis.xaxis_date()

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    iseg = 0
    iseg_count = 0

    t_start = max(tr[0].stats['starttime'], tr[1].stats['starttime'], tr[2].stats['starttime'])
    t_end = min(tr[0].stats['endtime'], tr[1].stats['endtime'], tr[2].stats['endtime'])

    t1 = t_start + iseg * timedelta(seconds=shift)
    t2 = t1 + timedelta(seconds=windows)

    fftZ_list = []
    fftN_list = []
    fftE_list = []
    VperH_list = []

    fig_window = plt.figure()
    ax_winZ = fig_window.add_subplot(3, 1, 1)
    ax_winN = fig_window.add_subplot(3, 1, 2)
    ax_winE = fig_window.add_subplot(3, 1, 3)
    color_iter = plt.cm.rainbow(np.linspace(0, 1, int((t_end - t_start) / shift)))

    while t1 < t_end - windows:
        cft_i_start = int((t1 - tr[0].stats['starttime']) * tr[0].stats['sampling_rate'])
        cft_i_end = int((t2 - tr[0].stats['starttime']) * tr[0].stats['sampling_rate'])

        if (max(cftZ[cft_i_start:cft_i_end]) > cft_max or min(cftZ[cft_i_start:cft_i_end]) < cft_min or
            max(cftN[cft_i_start:cft_i_end]) > cft_max or min(cftN[cft_i_start:cft_i_end]) < cft_min or
            max(cftE[cft_i_start:cft_i_end]) > cft_max or min(cftE[cft_i_start:cft_i_end]) < cft_min or
            iskip_sum[cft_i_start:cft_i_end].any()==True):
            iseg += 1
            t1 = tr[0].stats['starttime'] + iseg * timedelta(seconds=shift)
            t2 = t1 + timedelta(seconds=windows)
            continue

        ax_seis.axvspan(t1, t2, color=color_iter[iseg], alpha=0.3)

        trc = tr.copy()
        trc.trim(t1, t2)

        #calculate dip
        N = len(trc[0].data)
        Czz = np.sum(trc[0].data ** 2) / N
        Cnn = np.sum(trc[1].data ** 2) / N
        Cee = np.sum(trc[2].data ** 2) / N
        Czn = np.sum(trc[0].data * trc[1].data) / N
        Cze = np.sum(trc[0].data * trc[2].data) / N
        Cne = np.sum(trc[1].data * trc[2].data) / N

        cov_mat = np.array([[Cee, Cne, Cze], [Cne, Cnn, Czn], [Cze, Czn, Czz]])
        eigenvalues, eigenvectors = np.linalg.eig(cov_mat)
        idx = eigenvalues.argsort()[::-1]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]

        L_i = 1 - ((eigenvalues[1] + eigenvalues[2]) / (2 * eigenvalues[0]))
        eigenvectors_max = eigenvectors[:, 0]
        dip_i = abs(np.rad2deg(np.arctan(eigenvectors_max[2] / np.linalg.norm(eigenvectors_max[:2]))))
        azimuth_i = np.rad2deg(np.arctan2(eigenvectors_max[1], eigenvectors_max[0]))
        eig_i = eigenvalues[0]

        if dip_i<dip_tres:
            iseg += 1
            t1 = tr[0].stats['starttime'] + iseg * timedelta(seconds=shift)
            t2 = t1 + timedelta(seconds=windows)
            continue

        ax_winZ.plot(trc[0].data, alpha=0.3)
        ax_winN.plot(trc[1].data, alpha=0.3)
        ax_winE.plot(trc[2].data, alpha=0.3)

        N = len(trc[0].data)
        fftZ = (1 / N) * abs(np.fft.fft(trc[0].data))
        fftN = (1 / N) * abs(np.fft.fft(trc[1].data))
        fftE = (1 / N) * abs(np.fft.fft(trc[2].data))

        VperH = np.sqrt((fftZ ** 2) / (fftN ** 2 + fftE ** 2))
        VperH = sgolay(VperH, 100, 2)

        samplingrate = tr[0].stats['sampling_rate']
        freq = np.fft.fftfreq(N, 1 / samplingrate)
        VperH = VperH[:N // 2]
        freq = freq[:N // 2]

        ax.plot(freq, VperH, alpha=0.1, color=color_iter[iseg])
        fftZ_list.append(fftZ)
        fftN_list.append(fftN)
        fftE_list.append(fftE)
        VperH_list.append(VperH)

        iseg += 1
        iseg_count += 1
        t1 = tr[0].stats['starttime'] + iseg * timedelta(seconds=shift)
        t2 = t1 + timedelta(seconds=windows)
        trc.clear()

    fig_seis.savefig(savename + '_seis.png')
    plt.close(fig_seis)
    fig_window.savefig(savename + '_windowseis.png')
    plt.close(fig_window)

    if iseg_count == 0:
        ax.set_xlim(fmin_plot, fmax_plot)
        fig.savefig(savename + '.png')
        plt.close(fig)
        return np.nan, np.nan

    VperH_array = np.array(VperH_list)
    Vplot = np.median(VperH_array, axis=0)
    Vmean = np.mean(VperH_array, axis=0)
    freq = np.fft.fftfreq(N, 1 / samplingrate)
    freq = freq[:N // 2]

    q1plot = np.quantile(VperH_array, 0.25, axis=0)
    q3plot = np.quantile(VperH_array, 0.75, axis=0)

    ax.plot(freq, Vplot, color='black', linewidth=2)
    ax.plot(freq, q1plot, color='black', linewidth=1, linestyle='dashed')
    ax.plot(freq, q3plot, color='black', linewidth=1, linestyle='dashed')
    ax.plot(freq, Vmean, color='red', linewidth=2)
    ax.set_ylim(0, 2)
    ax.set_xlim(fmin_plot, fmax_plot)


    finterp = np.arange(fmin, fmax, 0.05)
    vinterp = np.interp(finterp, freq, Vplot)
    finterp_forplot=np.arange(fmin_plot,fmax_plot,0.05)
    vinterp_forplot=np.interp(finterp_forplot,freq,Vplot)
    ax.vlines(x=finterp[vinterp.argmax()],ymin=0,ymax=max(vinterp),colors='red',linestyles='dashdot',label='maximum v/h')
    ax.legend(loc='upper right')
    ax.text(x=finterp[vinterp.argmax()],y=max(vinterp),s="f: "+str(round(finterp[vinterp.argmax()],5))+"\nv/h: "+str(round(max(vinterp),5)),bbox=dict(facecolor='blue', alpha=0.5),wrap=True)
    fig.savefig(savename + '_vhsr.png')
    plt.close(fig)
    tr.clear()
    return max(vinterp), finterp[vinterp.argmax()], finterp, vinterp, finterp_forplot, vinterp_forplot

def calPSD_STALTA_dip(trdat, windows, shift, savename, fmin=1, fmax=4, fmin_plot=0.5, fmax_plot=10, cft_max=1.5, cft_min=0.5, dip_tres=45):
    if windows < shift:
        print('Ensure windows > shift!')
        return
    tr=trdat.copy()
    cftZ = classic_sta_lta(tr[0].data, windows * tr[0].stats['sampling_rate'], windows * tr[0].stats['sampling_rate'] * 3)
    cftN = classic_sta_lta(tr[1].data, windows * tr[1].stats['sampling_rate'], windows * tr[1].stats['sampling_rate'] * 3)
    cftE = classic_sta_lta(tr[2].data, windows * tr[2].stats['sampling_rate'], windows * tr[2].stats['sampling_rate'] * 3)

    iskip_sum = np.logical_and.reduce([np.logical_and(cftZ > 0.9999, cftZ < 1.0001), np.logical_and(tr[0].data > -0.00000001, tr[0].data < 0.00000001)])

    fig_seis = plt.figure()
    ax_seis = fig_seis.add_subplot(2, 1, 1)
    ax_seis.plot(tr[0].times("matplotlib"), tr[0].data, 'k')
    ax_seis.xaxis_date()
    ax_seis = fig_seis.add_subplot(2, 1, 2)
    ax_seis.plot(tr[0].times("matplotlib"), cftZ, 'k')
    ax_seis.xaxis_date()

    figZ, axZ = plt.subplots()
    figN, axN = plt.subplots()
    figE, axE = plt.subplots()

    iseg = 0
    t_start = max(tr[0].stats['starttime'], tr[1].stats['starttime'], tr[2].stats['starttime'])
    t_end = min(tr[0].stats['endtime'], tr[1].stats['endtime'], tr[2].stats['endtime'])
    t1 = t_start + iseg * timedelta(seconds=shift)
    t2 = t1 + timedelta(seconds=windows)

    fftZ_sum = np.zeros(int(((windows * tr[0].stats['sampling_rate']) + 1) // 2))
    fftN_sum = np.zeros_like(fftZ_sum)
    fftE_sum = np.zeros_like(fftZ_sum)
    iseg_count = 0
    color_iter = plt.cm.rainbow(np.linspace(0, 1, int((t_end - t_start) / shift)))

    while t1 < t_end - windows:
        cft_i_start = int((t1 - tr[0].stats['starttime']) * tr[0].stats['sampling_rate'])
        cft_i_end = int((t2 - tr[0].stats['starttime']) * tr[0].stats['sampling_rate'])

        if (max(cftZ[cft_i_start:cft_i_end]) > cft_max or min(cftZ[cft_i_start:cft_i_end]) < cft_min or
            max(cftN[cft_i_start:cft_i_end]) > cft_max or min(cftN[cft_i_start:cft_i_end]) < cft_min or
            max(cftE[cft_i_start:cft_i_end]) > cft_max or min(cftE[cft_i_start:cft_i_end]) < cft_min or
            iskip_sum[cft_i_start:cft_i_end].any()==True):
            iseg += 1
            t1 = tr[0].stats['starttime'] + iseg * timedelta(seconds=shift)
            t2 = t1 + timedelta(seconds=windows)
            continue

        ax_seis.axvspan(t1, t2, color=color_iter[iseg], alpha=0.3)
        trc = tr.copy()
        trc.trim(t1, t2)

        #calculate dip
        N = len(trc[0].data)
        Czz = np.sum(trc[0].data ** 2) / N
        Cnn = np.sum(trc[1].data ** 2) / N
        Cee = np.sum(trc[2].data ** 2) / N
        Czn = np.sum(trc[0].data * trc[1].data) / N
        Cze = np.sum(trc[0].data * trc[2].data) / N
        Cne = np.sum(trc[1].data * trc[2].data) / N

        cov_mat = np.array([[Cee, Cne, Cze], [Cne, Cnn, Czn], [Cze, Czn, Czz]])
        eigenvalues, eigenvectors = np.linalg.eig(cov_mat)
        idx = eigenvalues.argsort()[::-1]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]

        L_i = 1 - ((eigenvalues[1] + eigenvalues[2]) / (2 * eigenvalues[0]))
        eigenvectors_max = eigenvectors[:, 0]
        dip_i = abs(np.rad2deg(np.arctan(eigenvectors_max[2] / np.linalg.norm(eigenvectors_max[:2]))))
        azimuth_i = np.rad2deg(np.arctan2(eigenvectors_max[1], eigenvectors_max[0]))
        eig_i = eigenvalues[0]

        if dip_i<dip_tres:
            iseg += 1
            t1 = tr[0].stats['starttime'] + iseg * timedelta(seconds=shift)
            t2 = t1 + timedelta(seconds=windows)
            continue

        N = len(trc[0].data)
        fftZ = ((1 / N) * abs(np.fft.fft(trc[0].data)) ** 2)*(10**17)
        fftN = ((1 / N) * abs(np.fft.fft(trc[1].data)) ** 2)*(10**17)
        fftE = ((1 / N) * abs(np.fft.fft(trc[2].data)) ** 2)*(10**17)
        fftZ, fftN, fftE = fftZ[:N // 2], fftN[:N // 2], fftE[:N // 2]

        samplingrate = tr[0].stats['sampling_rate']
        freq = np.fft.fftfreq(N, 1 / samplingrate)[:N // 2]

        axZ.plot(freq, 10 * np.log10(fftZ), alpha=0.1, color=color_iter[iseg])
        axN.plot(freq, 10 * np.log10(fftN), alpha=0.1, color=color_iter[iseg])
        axE.plot(freq, 10 * np.log10(fftE), alpha=0.1, color=color_iter[iseg])

        fftZ_sum += fftZ
        fftN_sum += fftN
        fftE_sum += fftE
        iseg += 1
        iseg_count += 1
        t1 = tr[0].stats['starttime'] + iseg * timedelta(seconds=shift)
        t2 = t1 + timedelta(seconds=windows)
        trc.clear()

    fig_seis.savefig(savename + '_seis.png')
    plt.close(fig_seis)

    if iseg_count == 0:
        for ax in [axZ, axN, axE]:
            ax.set_xlim(fmin_plot, fmax_plot)
            ax.set_ylim(0, 150)
        figZ.savefig(savename + '_Z.png')
        figN.savefig(savename + '_N.png')
        figE.savefig(savename + '_E.png')
        plt.close(figZ)
        plt.close(figN)
        plt.close(figE)
        return np.nan, np.nan, np.nan

    fftZ_array = sgolay(10 * np.log10(fftZ_sum / iseg_count), 100, 2)
    fftN_array = sgolay(10 * np.log10(fftN_sum / iseg_count), 100, 2)
    fftE_array = sgolay(10 * np.log10(fftE_sum / iseg_count), 100, 2)

    freq = np.fft.fftfreq(N, 1 / samplingrate)[:N // 2]

    axZ.plot(freq, fftZ_array, color='black', linewidth=2)
    axZ.set_xlim(fmin_plot, fmax_plot)
    axZ.set_ylim(0, 150)

    axN.plot(freq, fftN_array, color='black', linewidth=2)
    axN.set_xlim(fmin_plot, fmax_plot)
    axN.set_ylim(0, 150)

    axE.plot(freq, fftE_array, color='black', linewidth=2)
    axE.set_xlim(fmin_plot, fmax_plot)
    axE.set_ylim(0, 150)

    finterp = np.arange(fmin, fmax, 0.05)
    Zinterp = np.interp(finterp, freq, fftZ_array)
    finterp_forplot=np.arange(fmin_plot,fmax_plot,0.05)
    Zinterp_forplot=np.interp(finterp_forplot,freq,fftZ_array)

    figZ.savefig(savename + '_Z.png')
    figN.savefig(savename + '_N.png')
    figE.savefig(savename + '_E.png')

    #integral
    verts = [(fmin, 20), *zip(finterp[(finterp >= fmin) & (finterp <= fmax)], Zinterp[(finterp >= fmin) & (finterp <= fmax)]), (fmax, 20)]
    poly = Polygon(verts, facecolor='0.5', edgecolor='0.5')
    psd_iz = np.trapz(Zinterp[(finterp >= fmin) & (finterp <= fmax)], x=finterp[(finterp >= fmin) & (finterp <= fmax)])
    axZ.add_patch(poly)

    axZ.vlines(x=finterp[Zinterp.argmax()],ymin=0,ymax=max(Zinterp),colors='red',linestyles='dashdot',label='maximum PSD-Z')
    axZ.legend(loc='upper right')
    axZ.text(x=finterp[Zinterp.argmax()], y=max(Zinterp),
            s="f: " + str(round(finterp[Zinterp.argmax()],5)) + "\nZmax: " + str(round(max(Zinterp),5))+ "\nIZ: "+ str(round(psd_iz,5)),
            bbox=dict(facecolor='blue', alpha=0.5), wrap=True)

    figZ.savefig(savename + '_Zintegral.png')
    plt.close(figZ)
    plt.close(figN)
    plt.close(figE)

    from PIL import Image

    imgZ = Image.open(savename + '_Z.png')
    imgN = Image.open(savename + '_N.png')
    imgE = Image.open(savename + '_E.png')

    # Pastikan semua ukuran sama
    width, height = imgZ.size
    combined = Image.new('RGB', (width, height * 3))
    combined.paste(imgZ, (0, 0))
    combined.paste(imgN, (0, height))
    combined.paste(imgE, (0, height * 2))

    combined.save(savename + '_ZNE.png')
    tr.clear()
    return max(Zinterp), finterp[Zinterp.argmax()], psd_iz, finterp, Zinterp, finterp_forplot, Zinterp_forplot

def cal_polar_dip(trdat, windows, shift, savename, fmin=1, fmax=4, fmin_plot=0.5, fmax_plot=10, cft_max=1.5, cft_min=0.5, dip_tres=45):
    if windows < shift:
        print('Ensure windows > shift!')
        return

    tr = trdat.copy()
    t_start = max(tr[0].stats['starttime'], tr[1].stats['starttime'], tr[2].stats['starttime'])
    t_end = min(tr[0].stats['endtime'], tr[1].stats['endtime'], tr[2].stats['endtime'])
    tr.trim(t_start, t_end)

    cftZ = classic_sta_lta(tr[0].data, windows * tr[0].stats['sampling_rate'], windows * tr[0].stats['sampling_rate'] * 3)
    cftN = classic_sta_lta(tr[1].data, windows * tr[1].stats['sampling_rate'], windows * tr[1].stats['sampling_rate'] * 3)
    cftE = classic_sta_lta(tr[2].data, windows * tr[2].stats['sampling_rate'], windows * tr[2].stats['sampling_rate'] * 3)

    iskip_sum = np.logical_and.reduce([np.logical_and(cftZ > 0.9999, cftZ < 1.0001), np.logical_and(tr[0].data > -0.00000001, tr[0].data < 0.00000001)])

    idx_sel = np.logical_and.reduce([
        np.logical_and(cftZ > cft_min, cftZ < cft_max),
        np.logical_and(cftN > cft_min, cftN < cft_max),
        np.logical_and(cftE > cft_min, cftE < cft_max),
        ~iskip_sum
    ])

    dataZ, dataN, dataE = tr[0].data[idx_sel], tr[1].data[idx_sel], tr[2].data[idx_sel]
    N = len(dataZ)
    Czz = np.sum(dataZ * dataZ) / N
    Cnn = np.sum(dataN * dataN) / N
    Cee = np.sum(dataE * dataE) / N
    Czn = np.sum(dataZ * dataN) / N
    Cze = np.sum(dataZ * dataE) / N
    Cne = np.sum(dataN * dataE) / N

    cov_max = np.array([[Cee, Cne, Cze], [Cne, Cnn, Czn], [Cze, Czn, Czz]])
    eigenvalues, eigenvectors = np.linalg.eig(cov_max)
    idx = eigenvalues.argsort()[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]

    L_all = 1 - ((eigenvalues[1] + eigenvalues[2]) / (2 * eigenvalues[0]))
    eigenvectors_max = eigenvectors[:, 0]
    dip_all = abs(np.rad2deg(np.arctan(eigenvectors_max[2] / np.linalg.norm(eigenvectors_max[:2]))))
    azimuth_all = np.rad2deg(np.arctan2(eigenvectors_max[1], eigenvectors_max[0]))
    eig_all = eigenvalues[0]

    iseg = 0
    t1 = t_start + iseg * timedelta(seconds=shift)
    t2 = t1 + timedelta(seconds=windows)

    L_list, dip_list, azimuth_list, eig_list, t_list = [], [], [], [], []

    while t1 < t_end - windows:
        cft_i_start = int((t1 - tr[0].stats['starttime']) * tr[0].stats['sampling_rate'])
        cft_i_end = int((t2 - tr[0].stats['starttime']) * tr[0].stats['sampling_rate'])

        if (max(cftZ[cft_i_start:cft_i_end]) > cft_max or min(cftZ[cft_i_start:cft_i_end]) < cft_min or
            max(cftN[cft_i_start:cft_i_end]) > cft_max or min(cftN[cft_i_start:cft_i_end]) < cft_min or
            max(cftE[cft_i_start:cft_i_end]) > cft_max or min(cftE[cft_i_start:cft_i_end]) < cft_min or
            iskip_sum[cft_i_start:cft_i_end].any()==True):
            iseg += 1
            t1 = tr[0].stats['starttime'] + iseg * timedelta(seconds=shift)
            t2 = t1 + timedelta(seconds=windows)
            continue

        trc = tr.copy()
        trc.trim(t1, t2)


        N = len(trc[0].data)
        Czz = np.sum(trc[0].data ** 2) / N
        Cnn = np.sum(trc[1].data ** 2) / N
        Cee = np.sum(trc[2].data ** 2) / N
        Czn = np.sum(trc[0].data * trc[1].data) / N
        Cze = np.sum(trc[0].data * trc[2].data) / N
        Cne = np.sum(trc[1].data * trc[2].data) / N

        cov_mat = np.array([[Cee, Cne, Cze], [Cne, Cnn, Czn], [Cze, Czn, Czz]])
        eigenvalues, eigenvectors = np.linalg.eig(cov_mat)
        idx = eigenvalues.argsort()[::-1]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]

        L_i = 1 - ((eigenvalues[1] + eigenvalues[2]) / (2 * eigenvalues[0]))
        eigenvectors_max = eigenvectors[:, 0]
        dip_i = abs(np.rad2deg(np.arctan(eigenvectors_max[2] / np.linalg.norm(eigenvectors_max[:2]))))
        azimuth_i = np.rad2deg(np.arctan2(eigenvectors_max[1], eigenvectors_max[0]))
        eig_i = eigenvalues[0]
        if dip_i<dip_tres:
            iseg += 1
            t1 = tr[0].stats['starttime'] + iseg * timedelta(seconds=shift)
            t2 = t1 + timedelta(seconds=windows)
            continue

        L_list.append(L_i)
        dip_list.append(dip_i)
        azimuth_list.append(azimuth_i)
        eig_list.append(eig_i)
        t_list.append(iseg * shift + windows / 2)

        iseg += 1
        t1 = tr[0].stats['starttime'] + iseg * timedelta(seconds=shift)
        t2 = t1 + timedelta(seconds=windows)
        trc.clear()

    fig = plt.figure(figsize=(7, 5))
    ax_L = fig.add_subplot(2, 2, 1)
    ax_L.scatter(t_list, L_list, color='black')
    ax_L.axhline(L_all, color='black')
    ax_L.axhline(np.mean(L_list), color='red', linestyle='dashed')
    ax_L.set_ylabel('Rectilinearity')
    ax_L.set_ylim(0, 1)

    ax_dip = fig.add_subplot(2, 2, 2)
    ax_dip.scatter(t_list, dip_list, color='black')
    ax_dip.axhline(dip_all, color='black')
    ax_dip.axhline(np.mean(dip_list), color='red', linestyle='dashed')
    ax_dip.set_ylabel('Dip (°)')
    ax_dip.set_ylim(0, 90)

    ax_azimuth = fig.add_subplot(2, 2, 3)
    ax_azimuth.scatter(t_list, azimuth_list, color='black')
    ax_azimuth.axhline(azimuth_all, color='black')
    ax_azimuth.axhline(np.mean(azimuth_list), color='red', linestyle='dashed')
    ax_azimuth.set_ylabel('Azimuth (°)')
    ax_azimuth.set_ylim(-50, 50)

    ax_eig = fig.add_subplot(2, 2, 4)
    ax_eig.scatter(t_list, eig_list, color='black')
    ax_eig.axhline(eig_all, color='black')
    ax_eig.axhline(np.mean(eig_list), color='red', linestyle='dashed')
    ax_eig.set_ylabel('Largest Eigenvalue')

    fig.savefig(savename + ".png")
    plt.close(fig)
    tr.clear()
    return (L_all, dip_all, azimuth_all, eig_all), (np.mean(L_list),np.mean(dip_list), np.mean(azimuth_list), np.mean(eig_list))

def plot_window_dip(trdat, windows, shift, savename, fmin=1, fmax=4, fmin_plot=0.5, fmax_plot=10, cft_max=1.5, cft_min=0.5,dip_tres=45):
    if windows < shift:
        print('Ensure windows > shift!')
        return

    tr=trdat.copy()
    cftZ = classic_sta_lta(tr[0].data, windows * tr[0].stats['sampling_rate'], windows * tr[0].stats['sampling_rate'] * 3)
    cftN = classic_sta_lta(tr[1].data, windows * tr[1].stats['sampling_rate'], windows * tr[1].stats['sampling_rate'] * 3)
    cftE = classic_sta_lta(tr[2].data, windows * tr[2].stats['sampling_rate'], windows * tr[2].stats['sampling_rate'] * 3)

    iskip_sum = np.logical_and.reduce([np.logical_and(cftZ > 0.9999, cftZ < 1.0001), np.logical_and(tr[0].data > -0.00000001, tr[0].data < 0.00000001)])

    fig_seis = plt.figure()
    ax_seisZ = fig_seis.add_subplot(3, 1, 1)
    ax_seisZ.plot(tr[0].times("matplotlib"), tr[0].data, 'k')
    ax_seisZ.xaxis_date()
    ax_seisZ.set_ylim(-0.0002,0.0002)
    ax_seisN = fig_seis.add_subplot(3, 1, 2)
    ax_seisN.plot(tr[1].times("matplotlib"), tr[1].data, 'k')
    ax_seisN.xaxis_date()
    ax_seisN.set_ylim(-0.0002,0.0002)
    ax_seisE = fig_seis.add_subplot(3, 1, 3)
    ax_seisE.plot(tr[2].times("matplotlib"), tr[2].data, 'k')
    ax_seisE.xaxis_date()
    ax_seisE.set_ylim(-0.0002,0.0002)
    #ax_seis = fig_seis.add_subplot(2, 1, 2)
    #ax_seis.plot(tr[0].times("matplotlib"), cft, 'k')
    #ax_seis.xaxis_date()

    iseg = 0
    t_start = max(tr[0].stats['starttime'], tr[1].stats['starttime'], tr[2].stats['starttime'])
    t_end = min(tr[0].stats['endtime'], tr[1].stats['endtime'], tr[2].stats['endtime'])
    t1 = t_start + iseg * timedelta(seconds=shift)
    t2 = t1 + timedelta(seconds=windows)

    iseg_count = 0
    color_iter = plt.cm.rainbow(np.linspace(0, 1, int((t_end - t_start) / shift)))

    while t1 < t_end - windows:
        cft_i_start = int((t1 - tr[0].stats['starttime']) * tr[0].stats['sampling_rate'])
        cft_i_end = int((t2 - tr[0].stats['starttime']) * tr[0].stats['sampling_rate'])

        if (max(cftZ[cft_i_start:cft_i_end]) > cft_max or min(cftZ[cft_i_start:cft_i_end]) < cft_min or
            max(cftN[cft_i_start:cft_i_end]) > cft_max or min(cftN[cft_i_start:cft_i_end]) < cft_min or
            max(cftE[cft_i_start:cft_i_end]) > cft_max or min(cftE[cft_i_start:cft_i_end]) < cft_min or
            iskip_sum[cft_i_start:cft_i_end].any()==True):
            iseg += 1
            t1 = tr[0].stats['starttime'] + iseg * timedelta(seconds=shift)
            t2 = t1 + timedelta(seconds=windows)
            continue

        trc = tr.copy()
        trc.trim(t1, t2)
        #caldip
        N = len(trc[0].data)
        Czz = np.sum(trc[0].data ** 2) / N
        Cnn = np.sum(trc[1].data ** 2) / N
        Cee = np.sum(trc[2].data ** 2) / N
        Czn = np.sum(trc[0].data * trc[1].data) / N
        Cze = np.sum(trc[0].data * trc[2].data) / N
        Cne = np.sum(trc[1].data * trc[2].data) / N

        cov_mat = np.array([[Cee, Cne, Cze], [Cne, Cnn, Czn], [Cze, Czn, Czz]])
        eigenvalues, eigenvectors = np.linalg.eig(cov_mat)
        idx = eigenvalues.argsort()[::-1]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]

        L_i = 1 - ((eigenvalues[1] + eigenvalues[2]) / (2 * eigenvalues[0]))
        eigenvectors_max = eigenvectors[:, 0]
        dip_i = abs(np.rad2deg(np.arctan(eigenvectors_max[2] / np.linalg.norm(eigenvectors_max[:2]))))
        azimuth_i = np.rad2deg(np.arctan2(eigenvectors_max[1], eigenvectors_max[0]))
        eig_i = eigenvalues[0]
        if dip_i<dip_tres:
            iseg += 1
            t1 = tr[0].stats['starttime'] + iseg * timedelta(seconds=shift)
            t2 = t1 + timedelta(seconds=windows)
            continue

        ax_seisZ.axvspan(t1, t2, color=color_iter[iseg], alpha=0.3)
        ax_seisN.axvspan(t1, t2, color=color_iter[iseg], alpha=0.3)
        ax_seisE.axvspan(t1, t2, color=color_iter[iseg], alpha=0.3)

        iseg += 1
        iseg_count += 1
        t1 = tr[0].stats['starttime'] + iseg * timedelta(seconds=shift)
        t2 = t1 + timedelta(seconds=windows)
        trc.clear()

    fig_seis.savefig(savename + '_selwindow.png')
    plt.close(fig_seis)

    percent = (iseg_count / iseg) * 100
    percent_2h = (iseg_count / 239) * 100
    group = 'good'
    if percent < 60:
        group = 'moderate'
    if percent < 30:
        group = 'poor'
    group_2h = 'good'
    if percent_2h < 60:
        group_2h = 'moderate'
    if percent_2h < 30:
        group_2h = 'poor'
    tr.clear()
    return percent, group, percent_2h, group_2h

def calculate_psd_for_gui(trdat, windows, shift, fmin=1, fmax=4, cft_max=1.5, cft_min=0.5, dip_tres=45):
    if windows < shift:
        print('Make sure windows > shift!')
        return None
        
    tr = trdat.copy()
    cftZ = classic_sta_lta(tr[0].data, int(windows * tr[0].stats['sampling_rate']), int(windows * tr[0].stats['sampling_rate'] * 3))
    cftN = classic_sta_lta(tr[1].data, int(windows * tr[1].stats['sampling_rate']), int(windows * tr[1].stats['sampling_rate'] * 3))
    cftE = classic_sta_lta(tr[2].data, int(windows * tr[2].stats['sampling_rate']), int(windows * tr[2].stats['sampling_rate'] * 3))

    iskip_sum = np.logical_and.reduce([np.logical_and(cftZ > 0.9999, cftZ < 1.0001), np.logical_and(tr[0].data > -0.00000001, tr[0].data < 0.00000001)])

    iseg = 0
    t_start = max(tr[0].stats['starttime'], tr[1].stats['starttime'], tr[2].stats['starttime'])
    t_end = min(tr[0].stats['endtime'], tr[1].stats['endtime'], tr[2].stats['endtime'])
    t1 = t_start + iseg * timedelta(seconds=shift)
    t2 = t1 + timedelta(seconds=windows)

    psd_Z_list, psd_N_list, psd_E_list = [], [], []
    valid_windows_info = []
    iseg_count = 0
    N_data = 0

    while t1 < t_end - timedelta(seconds=windows):
        cft_i_start = int((t1 - tr[0].stats['starttime']) * tr[0].stats['sampling_rate'])
        cft_i_end = int((t2 - tr[0].stats['starttime']) * tr[0].stats['sampling_rate'])

        if (max(cftZ[cft_i_start:cft_i_end]) > cft_max or min(cftZ[cft_i_start:cft_i_end]) < cft_min or
            max(cftN[cft_i_start:cft_i_end]) > cft_max or min(cftN[cft_i_start:cft_i_end]) < cft_min or
            max(cftE[cft_i_start:cft_i_end]) > cft_max or min(cftE[cft_i_start:cft_i_end]) < cft_min or
            iskip_sum[cft_i_start:cft_i_end].any()==True):
            iseg += 1
            t1 = tr[0].stats['starttime'] + iseg * timedelta(seconds=shift)
            t2 = t1 + timedelta(seconds=windows)
            continue

        trc = tr.copy()
        trc.trim(t1, t2)

        # Calculate dip
        N = len(trc[0].data)
        Czz = np.sum(trc[0].data ** 2) / N
        Cnn = np.sum(trc[1].data ** 2) / N
        Cee = np.sum(trc[2].data ** 2) / N
        Czn = np.sum(trc[0].data * trc[1].data) / N
        Cze = np.sum(trc[0].data * trc[2].data) / N
        Cne = np.sum(trc[1].data * trc[2].data) / N

        cov_mat = np.array([[Cee, Cne, Cze], [Cne, Cnn, Czn], [Cze, Czn, Czz]])
        eigenvalues, eigenvectors = np.linalg.eig(cov_mat)
        idx = eigenvalues.argsort()[::-1]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]
        eigenvectors_max = eigenvectors[:, 0]
        dip_i = abs(np.rad2deg(np.arctan(eigenvectors_max[2] / np.linalg.norm(eigenvectors_max[:2]))))

        if dip_i < dip_tres:
            iseg += 1
            t1 = tr[0].stats['starttime'] + iseg * timedelta(seconds=shift)
            t2 = t1 + timedelta(seconds=windows)
            continue
        
        valid_windows_info.append((t1, t2))
        N_data = len(trc[0].data)
        
        psd_Z = ((1 / N_data) * abs(np.fft.fft(trc[0].data)) ** 2)
        psd_N = ((1 / N_data) * abs(np.fft.fft(trc[1].data)) ** 2)
        psd_E = ((1 / N_data) * abs(np.fft.fft(trc[2].data)) ** 2)

        psd_Z_list.append(psd_Z[:N_data // 2])
        psd_N_list.append(psd_N[:N_data // 2])
        psd_E_list.append(psd_E[:N_data // 2])
        
        iseg_count += 1
        iseg += 1
        t1 = tr[0].stats['starttime'] + iseg * timedelta(seconds=shift)
        t2 = t1 + timedelta(seconds=windows)

    if iseg_count == 0:
        return None

    samplingrate = tr[0].stats['sampling_rate']
    freq = np.fft.fftfreq(N_data, 1 / samplingrate)[:N_data // 2]
    
    psd_Z_array = np.array(psd_Z_list)
    psd_N_array = np.array(psd_N_list)
    psd_E_array = np.array(psd_E_list)

    return {
        "freq": freq,
        "psd_Z_array": psd_Z_array,
        "psd_N_array": psd_N_array,
        "psd_E_array": psd_E_array,
        "mean_psd_Z": np.mean(psd_Z_array, axis=0),
        "mean_psd_N": np.mean(psd_N_array, axis=0),
        "mean_psd_E": np.mean(psd_E_array, axis=0),
        "window_count": iseg_count,
        "window_len": windows,
        "shift_len": shift,
        "valid_windows": valid_windows_info
    }

def calculate_polar_for_gui(trdat, windows, shift, cft_max=1.5, cft_min=0.5, dip_tres=45):
    if windows < shift:
        print('Make sure windows > shift!')
        return None

    tr = trdat.copy()
    
    if len(tr) < 3:
        print("Data does not have three components.")
        return None

    t_start = max(tr[0].stats['starttime'], tr[1].stats['starttime'], tr[2].stats['starttime'])
    t_end = min(tr[0].stats['endtime'], tr[1].stats['endtime'], tr[2].stats['endtime'])
    tr.trim(t_start, t_end)

    if tr[0].stats.npts == 0:
        print("Not enough data.")
        return None

    sampling_rate = tr[0].stats['sampling_rate']
    nsta = int(windows * sampling_rate)
    nlta = int(windows * sampling_rate * 3)

    cftZ = classic_sta_lta(tr[0].data, nsta, nlta)
    cftN = classic_sta_lta(tr[1].data, nsta, nlta)
    cftE = classic_sta_lta(tr[2].data, nsta, nlta)

    iskip_sum = np.logical_and.reduce([np.logical_and(cftZ > 0.9999, cftZ < 1.0001), np.logical_and(tr[0].data > -1e-8, tr[0].data < 1e-8)])
    
    idx_sel = np.logical_and.reduce([
        np.logical_and(cftZ > cft_min, cftZ < cft_max), np.logical_and(cftN > cft_min, cftN < cft_max),
        np.logical_and(cftE > cft_min, cftE < cft_max), ~iskip_sum
    ])

    dataZ, dataN, dataE = tr[0].data[idx_sel], tr[1].data[idx_sel], tr[2].data[idx_sel]
    N = len(dataZ)
    if N == 0: return None
    
    Czz = np.sum(dataZ * dataZ) / N; Cnn = np.sum(dataN * dataN) / N; Cee = np.sum(dataE * dataE) / N
    Czn = np.sum(dataZ * dataN) / N; Cze = np.sum(dataZ * dataE) / N; Cne = np.sum(dataN * dataE) / N

    cov_max = np.array([[Cee, Cne, Cze], [Cne, Cnn, Czn], [Cze, Czn, Czz]])
    eigenvalues, eigenvectors = np.linalg.eig(cov_max)
    idx = eigenvalues.argsort()[::-1]
    eigenvalues = eigenvalues[idx]; eigenvectors = eigenvectors[:, idx]

    L_all = 1 - ((eigenvalues[1] + eigenvalues[2]) / (2 * eigenvalues[0]))
    eigenvectors_max = eigenvectors[:, 0]
    dip_all = abs(np.rad2deg(np.arctan(eigenvectors_max[2] / np.linalg.norm(eigenvectors_max[:2]))))
    azimuth_all = np.rad2deg(np.arctan2(eigenvectors_max[1], eigenvectors_max[0]))
    eig_all = eigenvalues[0]

    iseg = 0
    L_list, dip_list, azimuth_list, eig_list, t_list, valid_windows = [], [], [], [], [], []

    while True:
        t1 = t_start + timedelta(seconds=(iseg * shift))
        t2 = t1 + timedelta(seconds=windows)

        if t2 > t_end:
            break
        
        time_diff_t1 = t1 - t_start 
        time_diff_t2 = t2 - t_start
        cft_i_start = int(time_diff_t1 * sampling_rate)
        cft_i_end = int(time_diff_t2 * sampling_rate)

        if (cft_i_end > len(cftZ) or 
            max(cftZ[cft_i_start:cft_i_end]) > cft_max or min(cftZ[cft_i_start:cft_i_end]) < cft_min or
            max(cftN[cft_i_start:cft_i_end]) > cft_max or min(cftN[cft_i_start:cft_i_end]) < cft_min or
            max(cftE[cft_i_start:cft_i_end]) > cft_max or min(cftE[cft_i_start:cft_i_end]) < cft_min or
            iskip_sum[cft_i_start:cft_i_end].any()):
            iseg += 1
            continue

        trc = tr.copy().trim(t1, t2)
        N = len(trc[0].data)
        if N == 0:
            iseg += 1
            continue

        Czz = np.sum(trc[0].data**2)/N; Cnn = np.sum(trc[1].data**2)/N; Cee = np.sum(trc[2].data**2)/N
        Czn = np.sum(trc[0].data*trc[1].data)/N; Cze = np.sum(trc[0].data*trc[2].data)/N; Cne = np.sum(trc[1].data*trc[2].data)/N

        cov_mat = np.array([[Cee, Cne, Cze], [Cne, Cnn, Czn], [Cze, Czn, Czz]])
        eigenvalues, eigenvectors = np.linalg.eig(cov_mat)
        idx = eigenvalues.argsort()[::-1]
        eigenvalues = eigenvalues[idx]; eigenvectors = eigenvectors[:, idx]

        eigenvectors_max = eigenvectors[:, 0]
        dip_i = abs(np.rad2deg(np.arctan(eigenvectors_max[2] / np.linalg.norm(eigenvectors_max[:2]))))
        
        if dip_i < dip_tres:
            iseg += 1
            continue
        
        valid_windows.append((t1, t2))
        L_i = 1 - ((eigenvalues[1] + eigenvalues[2]) / (2 * eigenvalues[0]))
        azimuth_i = np.rad2deg(np.arctan2(eigenvectors_max[1], eigenvectors_max[0]))
        eig_i = eigenvalues[0]

        L_list.append(L_i); dip_list.append(dip_i); azimuth_list.append(azimuth_i)
        eig_list.append(eig_i); t_list.append(t1 + timedelta(seconds=windows/2))

        iseg += 1
    
    if not t_list:
        return None

    return {
        "t_list": t_list, "L_list": L_list, "dip_list": dip_list, "azimuth_list": azimuth_list,
        "eig_list": eig_list, "L_all": L_all, "dip_all": dip_all, "azimuth_all": azimuth_all,
        "eig_all": eig_all, "mean_L": np.mean(L_list), "mean_dip": np.mean(dip_list),
        "mean_azimuth": np.mean(azimuth_list), "mean_eig": np.mean(eig_list),
        "valid_windows": valid_windows, "window_count": len(valid_windows)
    }