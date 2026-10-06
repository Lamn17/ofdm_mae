"""Unit-energy Gray QAM, MMSE-aware demapping, terminated rate-1/2 FEC."""
import runtime as _runtime
import numpy as np
from scipy.special import erfc
from channels.generation import complex_noise


def constellation(modulation):
    if modulation not in (4, 16):
        raise ValueError('Only QPSK and 16-QAM are supported')
    bps = int(np.log2(modulation))
    labels = ((np.arange(modulation)[:, None] >> np.arange(bps - 1, -1, -1)) & 1).astype(np.int8)
    if modulation == 4:
        points = ((1 - 2 * labels[:, 0]) + 1j * (1 - 2 * labels[:, 1])) / np.sqrt(2)
    else:
        # 00 -> +3, 01 -> +1, 11 -> -1, 10 -> -3 on each axis.
        points = ((1 - 2 * labels[:, 0]) * (3 - 2 * labels[:, 1])
                  + 1j * (1 - 2 * labels[:, 2]) * (3 - 2 * labels[:, 3])) / np.sqrt(10)
    return points, labels


def modulate(bits, modulation):
    points, labels = constellation(modulation)
    bits = np.asarray(bits).reshape(-1, labels.shape[1])
    return points[bits @ (2 ** np.arange(labels.shape[1] - 1, -1, -1))]


def mmse_equalize(y, hhat, sigma2_over_Es, sigma_e2=0.0):
    return hhat.conj() * y / (np.abs(hhat) ** 2 + sigma2_over_Es + sigma_e2)


def zf_equalize(y, hhat, eps=1e-12):
    return y / np.where(np.abs(hhat) > eps, hhat, eps + 0j)


def post_equalization_snr(hhat, noise_var, sigma_e2=0.0, Es=1.0):
    return np.abs(hhat) ** 2 / np.maximum(noise_var + np.asarray(sigma_e2) * Es, 1e-15)


def demap(y, hhat, noise_var, modulation, sigma_e2=0.0):
    points, labels = constellation(modulation)
    # Eq.22. Compare to gain-scaled constellation: ignoring this gain breaks 16-QAM.
    effective_noise = noise_var + np.asarray(sigma_e2)
    w = hhat.conj() / (np.abs(hhat)**2 + effective_noise)
    equalized = w * y
    gain = w * hhat
    variance = np.maximum(np.abs(w)**2 * effective_noise, 1e-15)
    distances = np.abs(equalized[:, None] - gain[:, None] * points)**2 / variance[:, None]
    llr = np.stack([distances[:, labels[:, b] == 1].min(axis=1)
                    - distances[:, labels[:, b] == 0].min(axis=1)
                    for b in range(labels.shape[1])], axis=1)
    return llr.reshape(-1), equalized


# K=3, (7,5)_oct convolutional code: two memory bits, four states.
NEXT = np.empty((4, 2), dtype=np.int64)
OUT = np.empty((4, 2, 2), dtype=np.int8)
for _state in range(4):
    for _bit in range(2):
        _reg = (_bit << 2) | _state
        NEXT[_state, _bit] = (_bit << 1) | (_state >> 1)
        OUT[_state, _bit] = [(_reg & p).bit_count() % 2 for p in (0o7, 0o5)]
PREV = np.array([[s for s in range(4) if d in NEXT[s]] for d in range(4)])
INPUT = np.arange(4) >> 1


def encode(bits):
    state, coded = 0, []
    for bit in np.r_[bits, [0, 0]]:
        coded.extend(OUT[state, int(bit)])
        state = NEXT[state, int(bit)]
    return np.asarray(coded, dtype=np.int8)


def decode(llr):
    llr = np.asarray(llr).reshape(-1, 2)
    metrics = np.array([0., np.inf, np.inf, np.inf])
    trace = np.empty((len(llr), 4), dtype=np.int8)
    for t, value in enumerate(llr):
        costs = metrics[PREV] + ((2 * OUT[PREV, INPUT[:, None]] - 1) * value).sum(-1) / 2
        choices = costs.argmin(axis=1)
        trace[t] = PREV[np.arange(4), choices]
        metrics = costs[np.arange(4), choices]
        metrics -= metrics.min()
    state = 0  # Encoder terminated in state zero.
    bits = np.empty(len(llr), dtype=np.int8)
    for t in range(len(llr) - 1, -1, -1):
        bits[t] = INPUT[state]
        state = trace[t, state]
    return bits[:-2]


def measure_link(h, hhat, observed, snr_db, modulation, seed, interleave=True):
    rng = np.random.default_rng(seed)
    data = ~observed
    hd, est = h[data], hhat[data]
    bps = int(np.log2(modulation))
    if len(hd) < 4:
        raise ValueError('Not enough data resource elements for a terminated codeword')
    noise_var = 10. ** (-snr_db / 10)
    uncoded = rng.integers(0, 2, (*h.shape, bps), dtype=np.int8)[data].reshape(-1)
    # Full-grid noise permits matched resource-element noise across different masks.
    noise = complex_noise(rng, h.shape, noise_var)[data]
    llr, _ = demap(hd * modulate(uncoded, modulation) + noise, est, noise_var, modulation)
    payload = rng.integers(0, 2, h.size * bps // 2, dtype=np.int8)[:len(hd) * bps // 2 - 2]
    coded = encode(payload)
    permutation = rng.permutation(len(coded)) if interleave else np.arange(len(coded))
    llr_tx, _ = demap(hd * modulate(coded[permutation], modulation) + noise, est, noise_var, modulation)
    llr_c = np.empty_like(llr_tx)
    llr_c[permutation] = llr_tx
    decoded = decode(llr_c)
    return dict(bit_errors=int(((llr < 0) != uncoded).sum()), bits=len(uncoded),
                block_errors=int(np.any(decoded != payload)), blocks=1,
                payload_bits=len(payload), coded_bit_errors=int((decoded != payload).sum()))


def measure_link_batch(h, hhat, observed, snr_db, modulation, seeds, interleave=True):
    """Vectorized equivalent of measure_link for equal-shape independent blocks.

    `observed` is one shared mask or one mask per block; blocks must carry the same
    number of data REs so their codewords have equal length.
    """
    h, hhat = np.asarray(h), np.asarray(hhat)
    seeds = list(seeds)
    if h.ndim != 3 or hhat.shape != h.shape or len(seeds) != len(h):
        raise ValueError('Batch channel/estimate/seed dimensions do not match')
    data = ~np.broadcast_to(np.asarray(observed, bool), h.shape)
    counts = data.reshape(len(h), -1).sum(1)
    if len(h) and (counts != counts[0]).any():
        raise ValueError('Batched blocks need equal data RE counts; use measure_link_grouped')
    hd, est = h[data].reshape(len(h), -1), hhat[data].reshape(len(h), -1)
    bps = int(np.log2(modulation))
    if hd.shape[1] < 4:
        raise ValueError('Not enough data resource elements for a terminated codeword')
    noise_var = 10. ** (-snr_db / 10)
    uncoded, noise, payload, permutations = [], [], [], []
    payload_length = hd.shape[1] * bps // 2 - 2
    for block, seed in zip(data, seeds):
        rng = np.random.default_rng(seed)
        uncoded.append(rng.integers(0, 2, (*block.shape, bps), dtype=np.int8)[block].reshape(-1))
        noise.append(complex_noise(rng, block.shape, noise_var)[block])
        payload.append(rng.integers(0, 2, block.size * bps // 2, dtype=np.int8)[:payload_length])
        permutations.append(rng.permutation(hd.shape[1] * bps) if interleave
                            else np.arange(hd.shape[1] * bps))
    uncoded, noise, payload = map(np.asarray, (uncoded, noise, payload))
    llr, _ = demap((hd * modulate(uncoded, modulation).reshape(hd.shape) + noise).reshape(-1),
                   est.reshape(-1), noise_var, modulation)
    llr = llr.reshape(len(h), -1)
    coded = np.asarray([encode(bits) for bits in payload])
    transmitted = np.asarray([bits[order] for bits,order in zip(coded,permutations)])
    llr_tx, _ = demap((hd * modulate(transmitted, modulation).reshape(hd.shape) + noise).reshape(-1),
                      est.reshape(-1), noise_var, modulation)
    llr_tx = llr_tx.reshape(len(h), -1)
    llr_c = np.empty_like(llr_tx)
    for i, order in enumerate(permutations):
        llr_c[i, order] = llr_tx[i]
    decoded = decode_batch(llr_c)
    results = []
    for i in range(len(h)):
        results.append(dict(bit_errors=int(((llr[i] < 0) != uncoded[i]).sum()), bits=uncoded.shape[1],
                            block_errors=int(np.any(decoded[i] != payload[i])), blocks=1,
                            payload_bits=payload.shape[1],
                            coded_bit_errors=int((decoded[i] != payload[i]).sum())))
    return results


def measure_link_grouped(h, hhat, observed, snr_db, modulation, seeds, interleave=True):
    """measure_link per block, batched over blocks that share a data-RE count."""
    observed = np.asarray(observed, bool)
    counts = observed.reshape(len(observed), -1).sum(1)
    results = [None] * len(observed)
    for count in np.unique(counts):
        ids = np.flatnonzero(counts == count)
        for i, result in zip(ids, measure_link_batch(h[ids], hhat[ids], observed[ids], snr_db, modulation,
                                                     [seeds[i] for i in ids], interleave)):
            results[i] = result
    return results


def awgn_ber(snr_linear, modulation):
    snr_linear = np.asarray(snr_linear)
    if modulation == 4:
        return .5 * erfc(np.sqrt(snr_linear / 2))
    if modulation == 16:
        # Exact Gray 16-QAM bit error probability as a function of Es/N0.
        x = np.sqrt(snr_linear / 10)
        return .375 * erfc(x) + .25 * erfc(3 * x) - .125 * erfc(5 * x)
    raise ValueError(modulation)


def gate0(seed=123, symbols=100000):
    rng = np.random.default_rng(seed)
    results = []
    for modulation in (4, 16):
        for snr_db in (0, 5, 10):
            bps = int(np.log2(modulation))
            bits = rng.integers(0, 2, symbols * bps, dtype=np.int8)
            x = modulate(bits, modulation)
            for channel in ('awgn', 'rayleigh'):
                h = np.ones(symbols, complex) if channel == 'awgn' else complex_noise(rng, (symbols,))
                nv = 10**(-snr_db / 10)
                y = h * x + complex_noise(rng, (symbols,), nv)
                llr, _ = demap(y, h, nv, modulation)
                measured = np.mean((llr < 0) != bits)
                expected = float(np.mean(awgn_ber(np.abs(h)**2 / nv, modulation)))
                tolerance = 6 * np.sqrt(max(expected * (1 - expected), 1 / len(bits)) / len(bits)) + 2 / len(bits)
                results.append(dict(modulation=modulation, snr_db=snr_db, channel=channel,
                                    ber=float(measured), theory=expected, tolerance=float(tolerance),
                                    passed=bool(abs(measured - expected) <= tolerance)))
    return results


def decode_batch(llrs):
    """Vectorized over independent codewords, with the same trellis as decode()."""
    llrs=np.asarray(llrs).reshape(len(llrs),-1,2)
    batch,steps=llrs.shape[:2]
    metrics=np.full((batch,4),np.inf);metrics[:,0]=0.
    trace=np.empty((steps,batch,4),np.int8)
    branch=2*OUT[PREV,INPUT[:,None]]-1
    for t in range(steps):
        costs=metrics[:,PREV]+(branch[None]*llrs[:,t,None,None,:]).sum(-1)/2
        choices=costs.argmin(-1)
        trace[t]=np.take_along_axis(np.broadcast_to(PREV,(batch,4,2)),choices[...,None],-1)[...,0]
        metrics=np.take_along_axis(costs,choices[...,None],-1)[...,0]
        metrics-=metrics.min(-1,keepdims=True)
    state=np.zeros(batch,np.int64);bits=np.empty((batch,steps),np.int8)
    for t in range(steps-1,-1,-1):
        bits[:,t]=INPUT[state]
        state=trace[t,np.arange(batch),state]
    return bits[:,:-2]


def calibration_bler(data_symbols, modulation, snr_db, blocks, seed):
    rng=np.random.default_rng(seed)
    bps=int(np.log2(modulation))
    payload=rng.integers(0,2,(blocks,data_symbols*bps//2-2),dtype=np.int8)
    coded=np.array([encode(b) for b in payload])
    x=modulate(coded.ravel(),modulation)
    nv=10**(-snr_db/10)
    llr,_=demap(x+complex_noise(rng,x.shape,nv),np.ones_like(x),nv,modulation)
    decoded=decode_batch(llr.reshape(blocks,-1))
    return int(np.any(decoded!=payload,axis=1).sum())


def gate0_backend(cfg, seed=123):
    from channels.generation import trajectory
    rng=np.random.default_rng(seed)
    h=trajectory(cfg,'C',60,cfg.val_ids[0],seed).reshape(-1)
    h=np.tile(h,max(1,100000//len(h)))
    results=[]
    for modulation in (4,16):
        bps=int(np.log2(modulation))
        bits=rng.integers(0,2,len(h)*bps,dtype=np.int8)
        x=modulate(bits,modulation)
        for snr_db in (0,5,10):
            nv=10**(-snr_db/10)
            llr,_=demap(h*x+complex_noise(rng,h.shape,nv),h,nv,modulation)
            measured=float(np.mean((llr<0)!=bits))
            expected=float(np.mean(awgn_ber(np.abs(h)**2/nv,modulation)))
            tolerance=6*np.sqrt(max(expected*(1-expected),1/len(bits))/len(bits))+2/len(bits)
            results.append(dict(modulation=modulation,snr_db=snr_db,channel=cfg.backend,
                                ber=measured,theory=expected,tolerance=float(tolerance),
                                passed=bool(abs(measured-expected)<=tolerance)))
    return results
