// BANK! network: observation builder + set-transformer forward pass (plain JS, no dependencies).
(function (root) {
  const G = 8192, SUMS = [3, 4, 5, 6, 8, 9, 10, 11], CNTS = [2, 2, 4, 4, 4, 4, 2, 2];
  const cols = new Map();
  function col(g) {            // exact chance the pot reaches g before a 7, for every pot below g
    let h = cols.get(g); if (h) return h;
    h = new Float64Array(2 * g + 12).fill(1);
    for (let p = g - 1; p >= 0; p--) {
      let acc = 0; for (let i = 0; i < 8; i++) acc += CNTS[i] * h[p + SUMS[i]];
      h[p] = p ? (acc + 6 * h[2 * p]) / 36 : acc / 30;
    }
    if (cols.size > 48) cols.clear();
    cols.set(g, h); return h;
  }
  function reach(pot, goal) {
    if (goal <= 0 || pot >= goal) return 1;
    const big = goal > G, gi = big ? G : goal;
    const pi = Math.min(G, Math.max(0, Math.floor(pot * (big ? G / goal : 1))));
    return pi >= gi ? 1 : col(gi)[pi];
  }
  // s: {pot, safe, round, R, rolls, score[], active[]} -> rows of 22 features, one per player
  function makeObs(s, exact = true) {
    const n = s.score.length, pot = s.pot, potc = Math.max(pot, 1), rem = s.R - s.round + 1;
    const mean = s.score.reduce((a, b) => a + b, 0) / n, nAct = s.active.filter(Boolean).length;
    const g = [Math.asinh(pot / 100) / 4, s.R / 20, rem / 20, rem / s.R, n / 8, nAct / n, Math.asinh(s.safe / 100) / 4,
      Math.asinh(s.rolls - 3) / 4, s.round === s.R ? 1 : 0];
    return s.score.map((sc, i) => {
      let om = -1; for (let j = 0; j < n; j++) if (j !== i && s.score[j] > om) om = s.score[j];
      const gap = sc - om, a = s.active[i] ? 1 : 0, goal = Math.round(om - sc + 1);
      let one = 6 * (2 * pot >= goal ? 1 : 0); for (let k = 0; k < 8; k++) one += CNTS[k] * (pot + SUMS[k] >= goal ? 1 : 0);
      return [Math.asinh(sc / 150) / 4, Math.asinh(gap / 150) / 4, Math.asinh(gap / potc) / 3, Math.asinh((gap + pot * a) / 150) / 4,
        Math.asinh((gap + pot * a) / potc) / 3, a, Math.asinh(sc / potc) / 3, Math.asinh((sc - mean) / 150) / 4,
        Math.asinh(gap / (rem * 100)) / 3, gap >= 0 ? 1 : 0, ...(exact ? [reach(pot, goal) * a, one / 36 * a, (goal <= G ? 1 : 0) * a] : []), ...g].map(Math.fround);
    });
  }
  function erf(x) {
    const s = x < 0 ? -1 : 1; x = Math.abs(x);
    const t = 1 / (1 + 0.3275911 * x);
    return s * (1 - (((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t - 0.284496736) * t + 0.254829592) * t * Math.exp(-x * x));
  }
  const gelu = x => 0.5 * x * (1 + erf(x / Math.SQRT2)), silu = x => x / (1 + Math.exp(-x));
  class Net {
    constructor(meta, buf) {
      this.cfg = meta.cfg; this.meta = meta; const all = new Float32Array(buf); this.t = {};
      for (const [k, [off, shape]] of Object.entries(meta.tensors)) this.t[k] = all.subarray(off, off + shape.reduce((a, b) => a * b, 1));
    }
    lin(x, name, fn) {           // x: array of rows
      const W = this.t[name + '.weight'], b = this.t[name + '.bias'], out = b.length, inn = W.length / out;
      return x.map(r => { const y = new Float64Array(out); for (let o = 0; o < out; o++) { let a = b[o]; const k = o * inn; for (let i = 0; i < inn; i++) a += W[k + i] * r[i]; y[o] = fn ? fn(a) : a; } return y; });
    }
    norm(x, name) {
      const w = this.t[name + '.weight'], b = this.t[name + '.bias'], d = w.length;
      return x.map(r => { let m = 0; for (let i = 0; i < d; i++) m += r[i]; m /= d; let v = 0; for (let i = 0; i < d; i++) v += (r[i] - m) ** 2;
        const sd = Math.sqrt(v / d + 1e-5), y = new Float64Array(d); for (let i = 0; i < d; i++) y[i] = (r[i] - m) / sd * w[i] + b[i]; return y; });
    }
    forward(obs) {
      const { d, layers, heads } = this.cfg, n = obs.length, hd = d / heads;
      let x = this.lin(this.lin(obs, 'inp.0', silu), 'inp.2');
      for (let L = 0; L < layers; L++) {
        const p = 'enc.layers.' + L + '.', h = this.norm(x, p + 'norm1');
        const W = this.t[p + 'self_attn.in_proj_weight'], B = this.t[p + 'self_attn.in_proj_bias'];
        const qkv = h.map(r => { const y = new Float64Array(3 * d); for (let o = 0; o < 3 * d; o++) { let a = B[o]; const k = o * d; for (let i = 0; i < d; i++) a += W[k + i] * r[i]; y[o] = a; } return y; });
        const att = Array.from({ length: n }, () => new Float64Array(d));
        for (let hh = 0; hh < heads; hh++) for (let i = 0; i < n; i++) {
          const sc = new Float64Array(n); let mx = -Infinity;
          for (let j = 0; j < n; j++) { let a = 0; for (let k = 0; k < hd; k++) a += qkv[i][hh * hd + k] * qkv[j][d + hh * hd + k]; sc[j] = a / Math.sqrt(hd); if (sc[j] > mx) mx = sc[j]; }
          let z = 0; for (let j = 0; j < n; j++) { sc[j] = Math.exp(sc[j] - mx); z += sc[j]; }
          for (let j = 0; j < n; j++) { const w = sc[j] / z; for (let k = 0; k < hd; k++) att[i][hh * hd + k] += w * qkv[j][2 * d + hh * hd + k]; }
        }
        const o = this.lin(att, p + 'self_attn.out_proj');
        x = x.map((r, i) => r.map((v, k) => v + o[i][k]));
        const f = this.lin(this.lin(this.norm(x, p + 'norm2'), p + 'linear1', gelu), p + 'linear2');
        x = x.map((r, i) => r.map((v, k) => v + f[i][k]));
      }
      x = this.norm(x, 'norm');
      const logit = this.lin(x, 'pi').map(r => r[0]), v = this.lin(x, 'v').map(r => r[0]);
      const mx = Math.max(...v), e = v.map(a => Math.exp(a - mx)), z = e.reduce((a, b) => a + b, 0);
      return { bank: logit.map(l => 1 / (1 + Math.exp(-l))), logit, win: e.map(a => a / z) };
    }
    advise(state) { return this.forward(makeObs(state, !!this.cfg.exact)); }
  }
  root.BankNet = { Net, makeObs, reach };
})(typeof module !== 'undefined' ? module.exports : window);
