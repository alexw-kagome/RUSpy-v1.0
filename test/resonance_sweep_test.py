import numpy as np

def build_sweep_segments(start_freq, stop_freq, resonances, tracking_window,
                         dense_spacing, sparse_spacing):
    """
    Return an array of [start, end, size] rows that cover [start_freq, stop_freq].
    Dense spacing is used within each resonance window [r - tw/2, r + tw/2],
    sparse spacing elsewhere. Overlapping dense windows are merged.
    """
    start = float(min(start_freq, stop_freq))
    stop  = float(max(start_freq, stop_freq))
    tw = float(tracking_window)

    # Normalize resonances
    res = np.atleast_1d(resonances).astype(float)
    res = res[np.isfinite(res)]

    # Build initial dense intervals
    dense_intervals = []
    if res.size > 0 and tw > 0:
        for r in np.sort(res):
            a = max(start, r - tw/2.0)
            b = min(stop,  r + tw/2.0)
            if a < b:
                dense_intervals.append([a, b])

        # Merge overlapping intervals
        dense_intervals.sort()
        merged = []
        for a, b in dense_intervals:
            if not merged or a > merged[-1][1]:
                merged.append([a, b])
            else:
                merged[-1][1] = max(merged[-1][1], b)
        dense_intervals = merged
    else:
        dense_intervals = []

    segments = []
    cursor = start
    for a, b in dense_intervals:
        if cursor < a:
            npts = int(np.floor((a - cursor) / sparse_spacing)) + 1
            segments.append([cursor, a, npts])
        npts = int(np.floor((b - a) / dense_spacing)) + 1
        segments.append([a, b, npts])
        cursor = b

    if cursor < stop:
        npts = int(np.floor((stop - cursor) / sparse_spacing)) + 1
        segments.append([cursor, stop, npts])

    return np.array(segments, dtype=float)


if __name__ == "__main__":
    start_freq = 1e6
    stop_freq = 2e6
    resonances = [1.2e6, 1.5e6, 1.8e6]
    tracking_window = 1e3
    dense_spacing = 1
    sparse_spacing = 10

    segments = build_sweep_segments(start_freq, stop_freq, resonances,
                                    tracking_window, dense_spacing, sparse_spacing)
    print(segments)