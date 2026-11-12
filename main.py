#!/usr/bin/env python3
"""photo-geolocator-01 entry: a model-free street-photo geolocator.

Method (stdlib only, deterministic, no model calls, no wall clock, no
unseeded randomness):

* The agent fetches the trial's 25 photos from the env, decodes each JPEG
  with its own baseline decoder, and answers one latitude/longitude per
  photo.
* The default answer is the geometric median of the 75 public practice
  photos of OpenStreetView-5M that the kit ships (56.1409, 32.3977): the
  single point that minimises the total distance to that public pool, and
  the best fixed guess when a photo carries no usable clue.
* One high-precision visual tell moves the answer: a strong yellow road
  marking in the lower-centre of the photo. On the public practice pool
  that tell fires only on photos from the Americas (a yellow centre line
  is the Americas' marking), so those photos are answered with the
  geometric median of the practice pool's Americas photos (23.6255,
  -86.7531). The tell is measured at full pixel resolution (a narrow
  line washes out at DC resolution), so the decoder also runs an IDCT
  over the lower-centre band of each photo.
* Colour, vegetation and brightness cues were tested on the public pool
  and carry no usable signal at this resolution (a ridge fit and several
  hand rules all scored worse than the fixed median out of sample), so
  the agent deliberately does not use them: a wrong region guess costs
  more than the median, and only the yellow-line tell earned its keep.

The two anchor points above are aggregate statistics of the public
practice pool (a geometric median each), not a table of photos and
their places: the repository carries no photo, no place list, no
weights and no embeddings.
"""

import json
import math
import os
import struct
import sys
import time
import urllib.error
import urllib.parse
import urllib.request

# ---------------------------------------------------------------- constants

# Geometric median of the 75 public practice photos (practice.json).
GLOBAL_LAT = 56.1409
GLOBAL_LON = 32.3977
# Geometric median of the practice photos taken in the Americas
# (longitude in [-170, -30)).
AMERICAS_LAT = 23.6255
AMERICAS_LON = -86.7531
# A photo counts as showing a yellow road marking when more than this
# share of the lower-centre band's pixels are strongly yellow.
YELLOW_FRACTION = 0.03

# ---------------------------------------------------------------- JPEG

ZIGZAG = [
    0, 1, 8, 16, 9, 2, 3, 10, 17, 24, 32, 25, 18, 11, 4, 5,
    12, 19, 26, 33, 40, 48, 41, 34, 27, 20, 13, 6, 7, 14, 21,
    28, 35, 42, 49, 56, 57, 50, 43, 36, 29, 22, 15, 23, 30, 37,
    44, 51, 58, 59, 52, 45, 38, 31, 39, 46, 53, 60, 61, 54, 47,
    55, 62, 63,
]


class JpegError(Exception):
    pass


class BitReader:
    """Bits of the entropy stream, honouring 0xFF00 stuffing."""

    __slots__ = ("data", "pos", "acc", "nbits")

    def __init__(self, data, start):
        self.data = data
        self.pos = start
        self.acc = 0
        self.nbits = 0

    def _fill(self):
        d = self.data
        p = self.pos
        acc = self.acc
        n = self.nbits
        while n <= 24 and p < len(d):
            b = d[p]
            p += 1
            if b == 0xFF:
                if p < len(d) and d[p] == 0x00:
                    p += 1  # stuffed 0xFF data byte
                else:
                    break  # a marker (restart, EOI): stop filling here
            acc = (acc << 8) | b
            n += 8
        self.pos = p
        self.acc = acc
        self.nbits = n

    def bits(self, n):
        if n == 0:
            return 0
        if self.nbits < n:
            self._fill()
        if self.nbits < n:  # past the end: pad with 1 bits, JPEG style
            v = (self.acc << (n - self.nbits)) | ((1 << (n - self.nbits)) - 1)
            self.nbits = 0
            self.acc = 0
            return v & ((1 << n) - 1)
        self.nbits -= n
        v = (self.acc >> self.nbits) & ((1 << n) - 1)
        self.acc &= (1 << self.nbits) - 1
        return v

    def bit(self):
        return self.bits(1)


class HuffTable:
    """Canonical JPEG Huffman table, with an 8-bit fast lookup."""

    __slots__ = ("counts", "symbols", "lut", "by_len")

    def __init__(self, counts, symbols):
        self.counts = counts
        self.symbols = symbols
        lut = [None] * 256
        code = 0
        idx = 0
        self.by_len = {}
        for length in range(1, 17):
            c = counts[length - 1]
            if c:
                self.by_len[length] = (code, idx, c)
                if length <= 8:
                    shift = 8 - length
                    for k in range(1 << shift):
                        lut[(code << shift) | k] = (symbols[idx], length)
                code += c
                idx += c
            code <<= 1
        self.lut = lut

    def decode(self, br):
        if br.nbits < 8:
            br._fill()
        if br.nbits >= 8:
            hit = self.lut[(br.acc >> (br.nbits - 8)) & 0xFF]
            if hit is not None:
                sym, ln = hit
                br.nbits -= ln
                br.acc &= (1 << br.nbits) - 1
                return sym
        code = 0
        for length in range(1, 17):
            code = (code << 1) | br.bit()
            e = self.by_len.get(length)
            if e is not None:
                fc, fs, cnt = e
                if fc <= code < fc + cnt:
                    return self.symbols[fs + (code - fc)]
        return 0


def _extend(v, s):
    if s == 0:
        return 0
    if v < (1 << (s - 1)):
        return v - ((1 << s) - 1)
    return v


# Precomputed IDCT cosines: COS[u][x] = C(u) * cos((2x+1)*u*pi/16).
_COS = []
for _u in range(8):
    _cu = 1.0 / math.sqrt(2.0) if _u == 0 else 1.0
    _COS.append([_cu * math.cos((2 * _x + 1) * _u * math.pi / 16.0)
                 for _x in range(8)])


def _idct_block(coeff, qtable):
    """Inverse DCT of one 8x8 block. coeff holds 64 dequantised values in
    raster order (row = vertical frequency, column = horizontal
    frequency). Returns out[x][y]: x horizontal, y vertical, floats with
    the level shift not yet applied."""
    f = [[coeff[r * 8 + c] * qtable[r * 8 + c] for c in range(8)]
         for r in range(8)]
    # Vertical pass: t[y][v] = sum_u COS[u][y] * f[u][v].
    t = [[0.0] * 8 for _ in range(8)]
    for y in range(8):
        cy = _COS[0][y], _COS[1][y], _COS[2][y], _COS[3][y], \
            _COS[4][y], _COS[5][y], _COS[6][y], _COS[7][y]
        for v in range(8):
            t[y][v] = (cy[0] * f[0][v] + cy[1] * f[1][v]
                       + cy[2] * f[2][v] + cy[3] * f[3][v]
                       + cy[4] * f[4][v] + cy[5] * f[5][v]
                       + cy[6] * f[6][v] + cy[7] * f[7][v])
    # Horizontal pass: out[x][y] = 0.25 * sum_v COS[v][x] * t[y][v].
    out = [[0.0] * 8 for _ in range(8)]
    for x in range(8):
        cx = _COS[0][x], _COS[1][x], _COS[2][x], _COS[3][x], \
            _COS[4][x], _COS[5][x], _COS[6][x], _COS[7][x]
        for y in range(8):
            ty = t[y]
            out[x][y] = 0.25 * (cx[0] * ty[0] + cx[1] * ty[1]
                                + cx[2] * ty[2] + cx[3] * ty[3]
                                + cx[4] * ty[4] + cx[5] * ty[5]
                                + cx[6] * ty[6] + cx[7] * ty[7])
    return out


def decode_yellow_fraction(jpeg):
    """Fraction of strongly yellow pixels in the lower-centre band
    (rows 50-100%, columns 30-70%) of a baseline JPEG, at full pixel
    resolution. Raises JpegError on anything this decoder cannot read."""
    if jpeg[:2] != b"\xff\xd8":
        raise JpegError("not a JPEG")
    pos = 2
    qtables = {}
    htables = {}
    width = height = 0
    comps = []  # (id, h, v, quant table id)
    while pos < len(jpeg):
        if jpeg[pos] != 0xFF:
            raise JpegError("bad marker at %d" % pos)
        marker = jpeg[pos + 1]
        pos += 2
        if marker in (0xD8, 0x01) or 0xD0 <= marker <= 0xD7:
            continue
        if marker == 0xD9:
            break
        seglen = struct.unpack(">H", jpeg[pos:pos + 2])[0]
        seg = jpeg[pos + 2:pos + seglen]
        if marker == 0xDB:  # DQT
            p = 0
            while p < len(seg):
                pq = seg[p] >> 4
                tq = seg[p] & 0x0F
                p += 1
                if pq == 0:
                    vals = list(seg[p:p + 64])
                    p += 64
                else:
                    vals = [struct.unpack(">H", seg[p + 2 * i:p + 2 * i + 2])[0]
                            for i in range(64)]
                    p += 128
                qt = [0] * 64
                for i in range(64):
                    qt[ZIGZAG[i]] = vals[i]
                qtables[tq] = qt
        elif marker == 0xC4:  # DHT
            p = 0
            while p < len(seg):
                tc = seg[p] >> 4
                th = seg[p] & 0x0F
                p += 1
                counts = list(seg[p:p + 16])
                p += 16
                total = sum(counts)
                symbols = list(seg[p:p + total])
                p += total
                htables[(tc, th)] = HuffTable(counts, symbols)
        elif marker == 0xC0:  # SOF0 baseline
            height = struct.unpack(">H", seg[1:3])[0]
            width = struct.unpack(">H", seg[3:5])[0]
            ncomp = seg[5]
            comps = []
            for i in range(ncomp):
                comps.append((seg[6 + 3 * i], seg[7 + 3 * i] >> 4,
                              seg[7 + 3 * i] & 0x0F, seg[8 + 3 * i]))
        elif marker == 0xC2:
            raise JpegError("progressive JPEG not supported")
        elif marker == 0xDA:  # SOS
            ncomp = seg[0]
            comp_tables = []
            for i in range(ncomp):
                tt = seg[2 + 2 * i]
                comp_tables.append((seg[1 + 2 * i], tt >> 4, tt & 0x0F))
            return _scan(jpeg, pos + seglen, qtables, htables,
                         width, height, comps, comp_tables)
        pos += seglen
    raise JpegError("no SOS found")


def _scan(jpeg, start, qtables, htables, width, height, comps, comp_tables):
    hmax = max(c[1] for c in comps)
    vmax = max(c[2] for c in comps)
    tmap = {}
    for cid, tdc, tac in comp_tables:
        for (cid2, h, v, tq) in comps:
            if cid2 == cid:
                tmap[cid] = (h, v, tq, htables[(0, tdc)], htables[(1, tac)])
    # Y block grid; the band of interest in Y-block coordinates.
    y_h, y_v = comps[0][1], comps[0][2]
    hb = (width + 8 * hmax - 1) // (8 * hmax) * y_h
    vb = (height + 8 * vmax - 1) // (8 * vmax) * y_v
    x0 = int(hb * 0.30)
    x1 = max(x0 + 1, int(hb * 0.70))
    y0 = vb // 2
    y1 = vb
    # Per component, the blocks overlapping the band, in that component's
    # own block grid (a Y pixel (X, Y) takes the sample
    # (X*h//hmax, Y*v//vmax) of the component).
    want = {}
    for (cid, h, v, tq) in comps:
        sx0 = x0 * 8 * h // hmax
        sx1 = x1 * 8 * h // hmax
        sy0 = y0 * 8 * v // vmax
        sy1 = y1 * 8 * v // vmax
        want[cid] = (sx0 // 8, (sx1 - 1) // 8, sy0 // 8, (sy1 - 1) // 8)
    # Decoded samples of the wanted blocks: cid -> {(bx, by): 8x8 floats}.
    blocks = {cid: {} for cid, _, _, _ in comps}
    br = BitReader(jpeg, start)
    dc_pred = {cid: 0 for cid, _, _, _ in comps}
    mcu_w = (width + 8 * hmax - 1) // (8 * hmax)
    mcu_h = (height + 8 * vmax - 1) // (8 * vmax)
    for my in range(mcu_h):
        for mx in range(mcu_w):
            for (cid, h, v, tq) in comps:
                htab_dc, htab_ac = tmap[cid][3], tmap[cid][4]
                q0 = qtables[tq]
                wx0, wx1, wy0, wy1 = want[cid]
                for by in range(v):
                    for bx in range(h):
                        gx = mx * h + bx
                        gy = my * v + by
                        keep = wx0 <= gx <= wx1 and wy0 <= gy <= wy1
                        coeff = [0] * 64 if keep else None
                        s = htab_dc.decode(br)
                        diff = _extend(br.bits(s), s) if s else 0
                        dc_pred[cid] += diff
                        if keep:
                            coeff[0] = dc_pred[cid]
                        k = 1
                        while k < 64:
                            rs = htab_ac.decode(br)
                            if rs == 0x00:
                                break
                            r = rs >> 4
                            s2 = rs & 0x0F
                            if s2 == 0 and r == 15:
                                k += 16
                                continue
                            val = _extend(br.bits(s2), s2) if s2 else 0
                            if keep:
                                coeff[ZIGZAG[k + r]] = val
                            k += r + 1
                        if keep:
                            blocks[cid][(gx, gy)] = _idct_block(coeff, q0)
    # Assemble the band's RGB pixels and count strongly yellow ones.
    y_id = comps[0][0]
    y_blocks = blocks[y_id]
    cb_blocks = blocks[comps[1][0]] if len(comps) > 1 else None
    cr_blocks = blocks[comps[2][0]] if len(comps) > 2 else None
    ch_h, ch_v = (comps[1][1], comps[1][2]) if len(comps) > 1 else (1, 1)
    cr_h, cr_v = (comps[2][1], comps[2][2]) if len(comps) > 2 else (1, 1)
    total = 0
    yellow = 0
    # Y pixel coordinates of the band.
    for gy in range(y0, y1):
        for gx in range(x0, x1):
            blk = y_blocks.get((gx, gy))
            if blk is None:
                continue
            for iy in range(8):
                py = gy * 8 + iy
                if py >= height:
                    continue
                for ix in range(8):
                    px = gx * 8 + ix
                    if px >= width:
                        continue
                    yv = blk[ix][iy] + 128.0
                    if cb_blocks is not None:
                        cbx = (px * ch_h // hmax) // 8
                        cby = (py * ch_v // vmax) // 8
                        crx = (px * cr_h // hmax) // 8
                        cry = (py * cr_v // vmax) // 8
                        cb_blk = cb_blocks.get((cbx, cby))
                        cr_blk = cr_blocks.get((crx, cry))
                        if cb_blk is None or cr_blk is None:
                            continue
                        cb = cb_blk[px * ch_h // hmax % 8][py * ch_v // vmax % 8] + 128.0
                        cr = cr_blk[px * cr_h // hmax % 8][py * cr_v // vmax % 8] + 128.0
                        cb -= 128.0
                        cr -= 128.0
                    else:
                        cb = cr = 0.0
                    r = yv + 1.402 * cr
                    g = yv - 0.34414 * cb - 0.71414 * cr
                    b = yv + 1.772 * cb
                    total += 1
                    if r > 140 and g > 110 and b < 120 and r - b > 80 and g - b > 40:
                        yellow += 1
    if total == 0:
        raise JpegError("empty band")
    return yellow / total


# ---------------------------------------------------------------- env

_OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}))


def _exchange(method, url, body=None):
    req = urllib.request.Request(url, data=body, method=method,
                                 headers={"Content-Type": "application/json",
                                          "Accept": "application/json"})
    try:
        with _OPENER.open(req, timeout=120) as res:
            return res.status, res.read()
    except urllib.error.HTTPError as err:
        try:
            return err.code, err.read()
        finally:
            err.close()
    except (urllib.error.URLError, OSError):
        return 0, b""


def _get_json(url, retries=3):
    for attempt in range(retries):
        status, raw = _exchange("GET", url)
        if status == 200:
            try:
                return json.loads(raw.decode("utf-8"))
            except ValueError:
                return None
        if attempt + 1 < retries:
            time.sleep(2)
    return None


def _get_bytes(url, retries=2):
    for attempt in range(retries):
        status, raw = _exchange("GET", url)
        if status == 200 and raw:
            return raw
        if attempt + 1 < retries:
            time.sleep(2)
    return None


def _post_answers(url, answers):
    body = json.dumps({"answers": answers}, separators=(",", ":")).encode("utf-8")
    status, raw = _exchange("POST", url, body)
    return status, raw


# ---------------------------------------------------------------- agent

def place(jpeg_bytes):
    """Latitude/longitude for one photo's JPEG bytes."""
    try:
        frac = decode_yellow_fraction(jpeg_bytes)
    except Exception:
        return GLOBAL_LAT, GLOBAL_LON
    if frac > YELLOW_FRACTION:
        return AMERICAS_LAT, AMERICAS_LON
    return GLOBAL_LAT, GLOBAL_LON


def main():
    prompt_file = os.environ.get("MAKEFUN_PROMPT_FILE", "")
    if prompt_file:
        try:
            with open(prompt_file, "rb") as f:
                f.read()  # the brief; read as the trial asks, not parsed
        except OSError as err:
            sys.stderr.write("cannot read prompt file: %s\n" % err)
    env_url = os.environ.get("MAKEFUN_ENV_URL", "").rstrip("/")
    if not env_url:
        sys.stderr.write("MAKEFUN_ENV_URL is not set\n")
        return 1
    listing = _get_json(env_url + "/photos")
    if not listing or not isinstance(listing.get("photos"), list):
        sys.stderr.write("cannot list photos\n")
        return 1
    photos = listing["photos"]
    answers = []
    for photo in photos:
        pid = photo.get("id")
        if pid is None:
            continue  # a photo without an id cannot be answered
        url = photo.get("url", "/photos/%s.jpg" % pid)
        if not url.startswith("http"):
            url = env_url + url
        data = _get_bytes(url)
        if data is None:
            lat, lon = GLOBAL_LAT, GLOBAL_LON
        else:
            lat, lon = place(data)
        answers.append({"id": pid,
                        "lat": round(lat, 6),
                        "lon": round(lon, 6)})
    if not answers:
        return 1
    status, _raw = _post_answers(env_url + "/answers", answers)
    if status != 200:
        # The env takes a POST whole or refuses it whole; retry the
        # photos one at a time, ignoring ones already answered.
        accepted = 0
        for ans in answers:
            st, _r = _post_answers(env_url + "/answers", [ans])
            if st in (200, 409):
                accepted += 1
        sys.stderr.write("batch post got %s; %d/%d accepted singly\n"
                         % (status, accepted, len(answers)))
        return 0 if accepted else 1
    sys.stderr.write("posted %d answers\n" % len(answers))
    return 0


if __name__ == "__main__":
    sys.exit(main())
