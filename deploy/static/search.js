// Hybrid search in the browser, mirroring the Python retriever:
// BM25Okapi (rank_bm25 defaults) + dense inner product on L2-normalised vectors, fused with RRF.

export function tokenize(text) {
  // Python: text.lower().split()
  return text.toLowerCase().split(/\s+/).filter(Boolean);
}

export class BM25Okapi {
  constructor(docs, { k1 = 1.5, b = 0.75, epsilon = 0.25 } = {}) {
    this.k1 = k1;
    this.b = b;
    this.docFreqs = [];
    this.docLen = new Float64Array(docs.length);
    const nd = new Map();
    let total = 0;
    docs.forEach((tokens, i) => {
      this.docLen[i] = tokens.length;
      total += tokens.length;
      const freqs = new Map();
      for (const t of tokens) freqs.set(t, (freqs.get(t) || 0) + 1);
      this.docFreqs.push(freqs);
      for (const t of freqs.keys()) nd.set(t, (nd.get(t) || 0) + 1);
    });
    this.n = docs.length;
    this.avgdl = total / this.n;
    // rank_bm25: negative idfs are replaced by epsilon * average idf.
    this.idf = new Map();
    let idfSum = 0;
    const negative = [];
    for (const [word, freq] of nd) {
      const idf = Math.log(this.n - freq + 0.5) - Math.log(freq + 0.5);
      this.idf.set(word, idf);
      idfSum += idf;
      if (idf < 0) negative.push(word);
    }
    const eps = epsilon * (idfSum / this.idf.size);
    for (const word of negative) this.idf.set(word, eps);
  }

  scores(queryTokens) {
    const scores = new Float64Array(this.n);
    for (const q of queryTokens) {
      const idf = this.idf.get(q) || 0;
      for (let i = 0; i < this.n; i++) {
        const tf = this.docFreqs[i].get(q) || 0;
        scores[i] += idf * ((tf * (this.k1 + 1)) / (tf + this.k1 * (1 - this.b + (this.b * this.docLen[i]) / this.avgdl)));
      }
    }
    return scores;
  }
}

// Indices of the top-k scores, descending, ties keep index order (numpy argsort kind="stable").
function topK(scores, k, allowed) {
  const idx = [];
  for (let i = 0; i < scores.length; i++) if (!allowed || allowed[i]) idx.push(i);
  idx.sort((a, b) => scores[b] - scores[a] || a - b);
  return idx.slice(0, k);
}

export function rrf(lists, k = 60) {
  const scores = new Map();
  for (const list of lists) list.forEach((id, rank) => scores.set(id, (scores.get(id) || 0) + 1 / (k + rank + 1)));
  // Array.prototype.sort is stable, so ties keep first-seen order like Python's sorted().
  return [...scores.entries()].sort((a, b) => b[1] - a[1]);
}

export function matchesFilters(chunk, filters) {
  if (filters.doc_type && chunk.doc_type !== filters.doc_type) return false;
  if (filters.newest_only && chunk.doc_type === "quy_che" && !chunk.newest_in_group) return false;
  return true;
}

export class HybridIndex {
  constructor(meta, chunks, vectors) {
    this.meta = meta;
    this.chunks = chunks;
    this.vectors = vectors; // Float32Array, row-major, num_chunks x dim
    this.bm25 = new BM25Okapi(chunks.map((c) => tokenize(c.text)), meta.bm25);
  }

  allowedMask(filters) {
    if (!filters || (!filters.doc_type && !filters.newest_only)) return null;
    return this.chunks.map((c) => matchesFilters(c, filters));
  }

  searchBM25(query, k, filters) {
    const allowed = this.allowedMask(filters);
    return topK(this.bm25.scores(tokenize(query.trim())), k, allowed);
  }

  searchDense(queryVector, k, filters) {
    const { dim } = this.meta;
    const scores = new Float64Array(this.chunks.length);
    for (let i = 0; i < this.chunks.length; i++) {
      let s = 0;
      const off = i * dim;
      for (let j = 0; j < dim; j++) s += this.vectors[off + j] * queryVector[j];
      scores[i] = s;
    }
    return topK(scores, k, this.allowedMask(filters));
  }

  search(query, queryVector, k, filters) {
    const r = this.meta.retrieval;
    const lists = [this.searchBM25(query, r.top_k_bm25, filters)];
    if (queryVector) lists.push(this.searchDense(queryVector, r.top_k_dense, filters));
    return rrf(lists, r.rrf_k).slice(0, k).map(([i, score]) => ({ ...this.chunks[i], score }));
  }
}
