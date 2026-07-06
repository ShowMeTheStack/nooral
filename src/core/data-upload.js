// Image folders from a student's own computer: each picture is turned grey, fitted into a small square and kept
// in the browser. The class of a picture is the name of the folder it sits in (or the start of its file name).
(function (S) {
  'use strict';
  const D = (S.Data = S.Data || {});

  /**
   * resizeGrey(rgba, w, h, size, {invert: 'auto' | true | false, fit: 'contain' | 'stretch'}) → Float32Array(size²)
   * Transparent pixels count as white; grey = 0.299 R + 0.587 G + 0.114 B. The picture is shrunk by averaging
   * the pixels each output pixel covers, and 'auto' inverts it when the border is light, so ink becomes 1.
   */
  D.resizeGrey = (rgba, w, h, size, opt = {}) => {
    const g = new Float32Array(w * h);
    for (let i = 0; i < w * h; i++) { const a = rgba[4 * i + 3] / 255, l = (0.299 * rgba[4 * i] + 0.587 * rgba[4 * i + 1] + 0.114 * rgba[4 * i + 2]) / 255; g[i] = a * l + (1 - a); }
    const s = opt.fit === 'stretch' ? null : Math.max(w, h), sx = s ? s / size : w / size, sy = s ? s / size : h / size, ox = s ? (w - s) / 2 : 0, oy = s ? (h - s) / 2 : 0;
    const out = new Float32Array(size * size);
    for (let r = 0; r < size; r++) for (let c = 0; c < size; c++) {
      const x0 = ox + c * sx, x1 = x0 + sx, y0 = oy + r * sy, y1 = y0 + sy; let sum = 0, area = 0;
      for (let y = Math.floor(y0); y < Math.ceil(y1); y++) for (let x = Math.floor(x0); x < Math.ceil(x1); x++) {
        const cover = (Math.min(x1, x + 1) - Math.max(x0, x)) * (Math.min(y1, y + 1) - Math.max(y0, y));
        if (cover <= 0) continue;
        sum += cover * (x >= 0 && y >= 0 && x < w && y < h ? g[y * w + x] : 1); area += cover;
      }
      out[r * size + c] = area ? sum / area : 1;
    }
    let border = 0, nb = 0;
    for (let i = 0; i < size; i++) { border += out[i] + out[(size - 1) * size + i] + out[i * size] + out[i * size + size - 1]; nb += 4; }
    const invert = opt.invert === 'auto' || opt.invert == null ? border / nb > 0.5 : opt.invert;
    if (invert) for (let i = 0; i < out.length; i++) out[i] = 1 - out[i];
    return out;
  };

  /** the class of an uploaded file: its folder name, else the part of its file name before '_', '-' or a digit */
  D.labelOf = (path) => {
    const parts = String(path).split(/[\\/]/).filter(Boolean);
    if (parts.length >= 2) return parts[parts.length - 2];
    const m = parts[0].replace(/\.[^.]*$/, '').match(/^[A-Za-z]+/);
    return m ? m[0].toLowerCase() : 'unlabelled';
  };

  /** build an image set from decoded pictures [{path, rgba, w, h}] (the part that needs no browser) */
  D.imagesFromPixels = (pics, opt = {}) => {
    const size = opt.size || 28, labels = pics.map((p) => D.labelOf(p.path)), classes = [...new Set(labels)].sort();
    const images = new Float32Array(pics.length * size * size);
    pics.forEach((p, i) => images.set(D.resizeGrey(p.rgba, p.w, p.h, size, opt), i * size * size));
    const get = (i) => images.subarray(i * size * size, (i + 1) * size * size);
    return { images, w: size, h: size, n: pics.length, y: labels.map((l) => classes.indexOf(l)), classes, paths: pics.map((p) => p.path), get, X: pics.map((_, i) => get(i)), uploaded: true, target: { name: 'class', type: 'category', categories: classes } };
  };

  /**
   * ingestImages(files, {size: 28, invert: 'auto', maxFiles: 5000, onProgress}) → Promise of an image set.
   * files: a FileList from <input type="file" webkitdirectory> or from a drop. Runs only in the browser.
   */
  D.ingestImages = async (files, opt = {}) => {
    const list = Array.from(files).filter((f) => /^image\//.test(f.type) || /\.(png|jpe?g|gif|bmp|webp)$/i.test(f.name)).slice(0, opt.maxFiles || 5000);
    const pics = [];
    for (let i = 0; i < list.length; i++) {
      const f = list[i], bmp = await createImageBitmap(f), w = bmp.width, h = bmp.height;
      const cv = typeof OffscreenCanvas !== 'undefined' ? new OffscreenCanvas(w, h) : Object.assign(document.createElement('canvas'), { width: w, height: h });
      const cx = cv.getContext('2d'); cx.drawImage(bmp, 0, 0); if (bmp.close) bmp.close();
      pics.push({ path: f.webkitRelativePath || f.name, rgba: cx.getImageData(0, 0, w, h).data, w, h });
      if (opt.onProgress) opt.onProgress(i + 1, list.length);
    }
    return D.imagesFromPixels(pics, opt);
  };

  /** read an uploaded CSV file into a parsed table (browser) */
  D.readCSVFile = async (file, opt) => D.parseCSV(await file.text(), opt);
})(globalThis.Nooral = globalThis.Nooral || {});
