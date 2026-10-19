/* Provable: движок аудита любой таблицы в браузере (перенос generic.py и роли столбцов из serve.py).
   Используется, когда локальный сервер не запущен — например, на публичном сайте. Данные не покидают браузер.
   Метод тот же: ранговая корреляция по всем строкам и внутри ячеек (сочетаний категорий), перестановочный тест
   внутри ячеек, бутстрап, связь средних по группе сравнения с повтором по каждому значению, поправка
   Бенджамини — Хохберга. Случайные числа — свой генератор, поэтому p могут чуть отличаться от Python-версии. */
(function (global) {
  const SEED = 20261012, MAX_ROWS = 200000;
  const GROUP_NAME = /(регион|region|^te$|город|urban|^k$|год|year|^god|квартал|quarter|^kv|отрасл|oked|kato|район|district|область|пол$|^pol$)/i;
  const ID_NAME = /(^id$|^id_|_id$|идентификатор|^код_?|номер|^n$|^№)/i;

  function rng(seed) {                      // mulberry32
    let a = seed >>> 0;
    return () => { a = (a + 0x6D2B79F5) >>> 0; let t = a; t = Math.imul(t ^ (t >>> 15), t | 1);
      t ^= t + Math.imul(t ^ (t >>> 7), t | 61); return ((t ^ (t >>> 14)) >>> 0) / 4294967296; };
  }
  function hash(s) { let h = 2166136261; for (const ch of s) { h ^= ch.codePointAt(0); h = Math.imul(h, 16777619); } return h >>> 0; }
  const num = v => { if (v === null || v === undefined) return NaN; const s = String(v).trim(); return s === '' ? NaN : Number(s); };

  function rank(x) {                        // средние ранги при совпадениях, как pandas rank()
    const n = x.length, idx = Array.from({length: n}, (_, i) => i).sort((a, b) => x[a] - x[b]);
    const r = new Float64Array(n);
    for (let i = 0; i < n;) { let j = i; while (j + 1 < n && x[idx[j + 1]] === x[idx[i]]) j++;
      const v = (i + j) / 2 + 1; for (let k = i; k <= j; k++) r[idx[k]] = v; i = j + 1; }
    return r;
  }
  function corr(a, b) {
    const n = a.length; let ma = 0, mb = 0; for (let i = 0; i < n; i++) { ma += a[i]; mb += b[i]; } ma /= n; mb /= n;
    let s = 0, sa = 0, sb = 0; for (let i = 0; i < n; i++) { const x = a[i] - ma, y = b[i] - mb; s += x * y; sa += x * x; sb += y * y; }
    const d = Math.sqrt(sa * sb); return d > 0 ? s / d : 0;
  }
  function demean(x, codes, k) {
    const s = new Float64Array(k), c = new Float64Array(k);
    for (let i = 0; i < x.length; i++) { s[codes[i]] += x[i]; c[codes[i]]++; }
    const o = new Float64Array(x.length); for (let i = 0; i < x.length; i++) o[i] = x[i] - s[codes[i]] / Math.max(c[codes[i]], 1);
    return o;
  }
  const pct = (arr, q) => { const s = Float64Array.from(arr).sort(), p = (s.length - 1) * q, lo = Math.floor(p), hi = Math.ceil(p); return s[lo] + (s[hi] - s[lo]) * (p - lo); };

  /* двусторонний p для t-статистики через регуляризованную неполную бета-функцию */
  function lgamma(x) { const c = [76.18009172947146, -86.50532032941677, 24.01409824083091, -1.231739572450155, 0.1208650973866179e-2, -0.5395239384953e-5];
    let y = x, t = x + 5.5; t -= (x + 0.5) * Math.log(t); let s = 1.000000000190015; for (const v of c) s += v / ++y; return -t + Math.log(2.5066282746310005 * s / x); }
  function betacf(a, b, x) { let qab = a + b, qap = a + 1, qam = a - 1, c = 1, d = 1 - qab * x / qap; if (Math.abs(d) < 1e-300) d = 1e-300; d = 1 / d; let h = d;
    for (let m = 1; m <= 300; m++) { const m2 = 2 * m; let aa = m * (b - m) * x / ((qam + m2) * (a + m2));
      d = 1 + aa * d; if (Math.abs(d) < 1e-300) d = 1e-300; c = 1 + aa / c; if (Math.abs(c) < 1e-300) c = 1e-300; d = 1 / d; h *= d * c;
      aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2)); d = 1 + aa * d; if (Math.abs(d) < 1e-300) d = 1e-300; c = 1 + aa / c; if (Math.abs(c) < 1e-300) c = 1e-300;
      d = 1 / d; const del = d * c; h *= del; if (Math.abs(del - 1) < 3e-14) break; } return h; }
  function ibeta(x, a, b) { if (x <= 0) return 0; if (x >= 1) return 1;
    const bt = Math.exp(lgamma(a + b) - lgamma(a) - lgamma(b) + a * Math.log(x) + b * Math.log(1 - x));
    return x < (a + 1) / (a + b + 2) ? bt * betacf(a, b, x) / a : 1 - bt * betacf(b, a, 1 - x) / b; }
  const tTwoSided = (t, df) => ibeta(df / (df + t * t), df / 2, 0.5);

  function bins(x, y, k = 10) {
    const n = x.length, order = Array.from({length: n}, (_, i) => i).sort((a, b) => x[a] - x[b]), out = [];
    for (let b = 0; b < k; b++) { const lo = Math.floor(b * n / k), hi = Math.floor((b + 1) * n / k); if (hi <= lo) continue;
      let sx = 0, sy = 0; for (let i = lo; i < hi; i++) { sx += x[order[i]]; sy += y[order[i]]; }
      out.push([+(sx / (hi - lo)).toFixed(4), +(sy / (hi - lo)).toFixed(4)]); }
    return out;
  }

  /* роли столбцов: показатель, категория или не использовать */
  function describe(rows, columns) {
    let sampled = false;
    if (rows.length > MAX_ROWS) { const r = rng(1); for (let i = rows.length - 1; i > 0; i--) { const j = Math.floor(r() * (i + 1)); [rows[i], rows[j]] = [rows[j], rows[i]]; }
      rows = rows.slice(0, MAX_ROWS); sampled = true; }
    const n = rows.length, cols = []; let nVar = 0;
    for (const c of columns) {
      const vals = rows.map(r => r[c]), nums = vals.map(num), ok = nums.filter(v => !isNaN(v));
      const isNum = ok.length >= 0.5 * n, uniq = new Set(vals.filter(v => v !== '' && v != null)).size;
      const consecutive = isNum && uniq === n && ok.every(v => v % 1 === 0) && Math.max(...ok) - Math.min(...ok) + 1 === uniq;
      let role;
      if (ID_NAME.test(c) || (!isNum && uniq >= 0.9 * n && uniq > 50) || (consecutive && uniq > 50)) role = 'skip';
      else if (GROUP_NAME.test(c) && uniq <= 100) role = 'group';
      else if (!isNum) role = uniq <= 50 ? 'group' : 'skip';
      else if (uniq >= 3 && nVar < 12) role = 'var';
      else role = 'skip';
      if (role === 'var') nVar++;
      const ex = [...new Set(vals.filter(v => v !== '' && v != null))].slice(0, 4).join(', ');
      cols.push({name: c, role, unique: uniq, example: ex.slice(0, 60),
        suggest_compare: /(регион|region|^te$|kato|область)/i.test(c), suggest_repeat: /(год|year|^god)/i.test(c)});
    }
    return {rows, info: {token: 'browser', rows: n, sampled, columns: cols}};
  }

  /* полный аудит всех пар показателей */
  function auditTable(rows, variables, cellGroups, compareGroup, repeatBy, opts = {}) {
    const nPerm = opts.nPerm || 200, nBoot = opts.nBoot || 200;
    let noGroups = false;
    if (!cellGroups.length) { noGroups = true; cellGroups = ['вся таблица']; compareGroup = 'вся таблица'; rows = rows.map(r => ({...r, 'вся таблица': 'все'})); }
    const need = [...cellGroups, compareGroup, ...(repeatBy ? [repeatBy] : [])];
    const d = rows.filter(r => need.every(g => r[g] !== '' && r[g] != null));
    const N = d.length, cellKey = d.map(r => cellGroups.map(g => r[g]).join('|'));
    const cellIds = new Map(); cellKey.forEach(k => { if (!cellIds.has(k)) cellIds.set(k, cellIds.size); });
    const K = cellIds.size, gv = d.map(r => String(r[compareGroup])), rv = d.map(r => repeatBy ? String(r[repeatBy]) : 'все');
    const reps = [...new Set(rv)].sort();
    const V = {}; variables.forEach(v => { V[v] = d.map(r => num(r[v])); });
    const pairs = [];

    for (let i = 0; i < variables.length; i++) for (let j = i + 1; j < variables.length; j++) {
      const a = variables[i], b = variables[j], key = [a, b].sort(), R = rng(SEED ^ hash(key[0]) ^ Math.imul(hash(key[1]), 31));
      const m = []; for (let t = 0; t < N; t++) if (!isNaN(V[a][t]) && !isNaN(V[b][t])) m.push(t);
      const n = m.length; if (n < 30) continue;
      const loc = new Map(), codes = new Int32Array(n); m.forEach((t, q) => { const c = cellKey[t]; if (!loc.has(c)) loc.set(c, loc.size); codes[q] = loc.get(c); });
      const k = loc.size, xa = m.map(t => V[a][t]), xb = m.map(t => V[b][t]);
      const ra = rank(xa), rb = rank(xb), raw = corr(ra, rb);
      const pa = demean(ra, codes, k), pb = demean(rb, codes, k), part = corr(pa, pb);
      // перестановки внутри ячеек
      const byCell = Array.from({length: k}, () => []); for (let q = 0; q < n; q++) byCell[codes[q]].push(q);
      let ma = 0, mb = 0; for (let q = 0; q < n; q++) { ma += pa[q]; mb += pb[q]; } ma /= n; mb /= n;
      const paz = new Float64Array(n), pbz = new Float64Array(n); let saa = 0, sbb = 0;
      for (let q = 0; q < n; q++) { paz[q] = pa[q] - ma; pbz[q] = pb[q] - mb; saa += paz[q] ** 2; sbb += pbz[q] ** 2; }
      const den = Math.sqrt(saa * sbb) || 1; let hits = 0;
      for (let s = 0; s < nPerm; s++) {
        let acc = 0;
        for (const cell of byCell) { const sh = cell.slice();
          for (let x = sh.length - 1; x > 0; x--) { const y = Math.floor(R() * (x + 1)); const tmp = sh[x]; sh[x] = sh[y]; sh[y] = tmp; }
          for (let x = 0; x < cell.length; x++) acc += paz[cell[x]] * pbz[sh[x]]; }
        if (Math.abs(acc / den) >= Math.abs(part)) hits++;
      }
      const pPerm = (1 + hits) / (nPerm + 1);
      const dfT = Math.max(n - K - 2, 1), tStat = part * Math.sqrt(dfT / Math.max(1e-12, 1 - part * part)), pAn = tTwoSided(Math.abs(tStat), dfT);
      // бутстрап-интервал
      const bs = [], ba = new Float64Array(n), bb = new Float64Array(n);
      for (let s = 0; s < nBoot; s++) { for (let q = 0; q < n; q++) { const z = Math.floor(R() * n); ba[q] = pa[z]; bb[q] = pb[z]; } bs.push(corr(ba, bb)); }
      const ci = [pct(bs, 0.025), pct(bs, 0.975)];
      const retained = Math.abs(raw) > 1e-9 ? Math.abs(part) / Math.abs(raw) : 0;
      const unitOk = pPerm < 0.01 && Math.abs(part) >= 0.03 && (retained >= 0.5 || Math.abs(part) >= 0.1);
      // уровень групп
      const grp = {};
      for (const r of reps) {
        const agg = new Map(); m.forEach((t, q) => { if (rv[t] !== r) return; const g = gv[t]; if (!agg.has(g)) agg.set(g, [0, 0, 0]); const e = agg.get(g); e[0] += xa[q]; e[1] += xb[q]; e[2]++; });
        if (agg.size < 5) continue;
        const names = [...agg.keys()].sort(), mA = names.map(g => agg.get(g)[0] / agg.get(g)[2]), mB = names.map(g => agg.get(g)[1] / agg.get(g)[2]);
        const rho = corr(rank(mA), rank(mB)), G = names.length, boots = [];
        for (let s = 0; s < nBoot; s++) { const x = [], y = []; for (let q = 0; q < G; q++) { const z = Math.floor(R() * G); x.push(mA[z] + (R() - 0.5) * 1e-9); y.push(mB[z] + (R() - 0.5) * 1e-9); } boots.push(corr(rank(x), rank(y))); }
        grp[r] = {rho, ci: [pct(boots, 0.025), pct(boots, 0.975)], groups: G, means: names.map((g, q) => [g, +mA[q].toFixed(4), +mB[q].toFixed(4)])};
      }
      const gs = Object.values(grp), signs = new Set(gs.map(v => Math.sign(v.rho)));
      const groupOk = gs.length > 0 && signs.size === 1 && gs.every(v => v.ci[0] > 0 || v.ci[1] < 0);
      pairs.push({a, b, n, raw, partial: part, ci, p_perm: pPerm, p_analytic: pAn, group: grp,
        curve: {all: bins(Array.from(ra, v => v / n), Array.from(rb, v => v / n)), within: bins(Array.from(pa, v => v / n), Array.from(pb, v => v / n))},
        level: unitOk ? 'UNIT' : (groupOk ? 'GROUP' : 'NONE')});
    }
    // Бенджамини — Хохберг, q = 0,01
    if (pairs.length) {
      const order = pairs.map((p, q) => q).sort((x, y) => pairs[x].p_analytic - pairs[y].p_analytic), M = pairs.length;
      let last = -1; order.forEach((q, r) => { if (pairs[q].p_analytic <= 0.01 * (r + 1) / M) last = r; });
      order.forEach((q, r) => { const p = pairs[q]; p.bh_ok = r <= last;
        if (p.level === 'UNIT' && !p.bh_ok) { const g = Object.values(p.group);
          p.level = g.length && g.every(v => v.ci[0] > 0 || v.ci[1] < 0) ? 'GROUP' : 'NONE'; p.note = 'не прошла поправку на множественные сравнения'; } });
    }
    const counts = {UNIT: 0, GROUP: 0, NONE: 0}; pairs.forEach(p => counts[p.level]++);
    return {rows: N, variables, labels: Object.fromEntries(variables.map(v => [v, v])), cell_groups: cellGroups, compare_group: compareGroup,
      repeat_by: repeatBy || null, repeats: reps, cells: K, n_perm: nPerm, n_boot: nBoot, seed: SEED, pairs, counts, no_groups: noGroups, engine: 'browser'};
  }

  global.ProvableEngine = {describe, auditTable};
})(typeof window !== 'undefined' ? window : globalThis);
