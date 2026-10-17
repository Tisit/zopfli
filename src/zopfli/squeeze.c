/*
Copyright 2011 Google Inc. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

Author: lode.vandevenne@gmail.com (Lode Vandevenne)
Author: jyrki.alakuijala@gmail.com (Jyrki Alakuijala)
*/

#include "squeeze.h"

#include <assert.h>
#include <math.h>
#include <stdio.h>

#include "blocksplitter.h"
#include "deflate.h"
#include "symbols.h"
#include "tree.h"
#include "util.h"

#if defined(__x86_64__) || (defined(_M_X64) && !defined(_M_ARM64EC))
/* SSE2 is part of x86-64, and there the scalar double math is SSE2 as well. */
#define ZOPFLI_SSE2
#include <emmintrin.h>
#endif

typedef struct SymbolStats {
  /* The literal and length symbols. */
  size_t litlens[ZOPFLI_NUM_LL];
  /* The 32 unique dist symbols, not the 32768 possible dists. */
  size_t dists[ZOPFLI_NUM_D];

  /* Length of each lit/len symbol in bits. */
  double ll_symbols[ZOPFLI_NUM_LL];
  /* Length of each dist symbol in bits. */
  double d_symbols[ZOPFLI_NUM_D];
} SymbolStats;

/* Sets everything to 0. */
static void InitStats(SymbolStats* stats) {
  memset(stats->litlens, 0, ZOPFLI_NUM_LL * sizeof(stats->litlens[0]));
  memset(stats->dists, 0, ZOPFLI_NUM_D * sizeof(stats->dists[0]));

  memset(stats->ll_symbols, 0, ZOPFLI_NUM_LL * sizeof(stats->ll_symbols[0]));
  memset(stats->d_symbols, 0, ZOPFLI_NUM_D * sizeof(stats->d_symbols[0]));
}

static void CopyStats(SymbolStats* source, SymbolStats* dest) {
  memcpy(dest->litlens, source->litlens,
         ZOPFLI_NUM_LL * sizeof(dest->litlens[0]));
  memcpy(dest->dists, source->dists, ZOPFLI_NUM_D * sizeof(dest->dists[0]));

  memcpy(dest->ll_symbols, source->ll_symbols,
         ZOPFLI_NUM_LL * sizeof(dest->ll_symbols[0]));
  memcpy(dest->d_symbols, source->d_symbols,
         ZOPFLI_NUM_D * sizeof(dest->d_symbols[0]));
}

/* Adds the bit lengths. */
static void AddWeighedStatFreqs(const SymbolStats* stats1, double w1,
                                const SymbolStats* stats2, double w2,
                                SymbolStats* result) {
  size_t i;
  for (i = 0; i < ZOPFLI_NUM_LL; i++) {
    result->litlens[i] =
        (size_t) (stats1->litlens[i] * w1 + stats2->litlens[i] * w2);
  }
  for (i = 0; i < ZOPFLI_NUM_D; i++) {
    result->dists[i] =
        (size_t) (stats1->dists[i] * w1 + stats2->dists[i] * w2);
  }
  result->litlens[256] = 1;  /* End symbol. */
}

typedef struct RanState {
  unsigned int m_w, m_z;
} RanState;

static void InitRanState(RanState* state) {
  state->m_w = 1;
  state->m_z = 2;
}

/* Get random number: "Multiply-With-Carry" generator of G. Marsaglia */
static unsigned int Ran(RanState* state) {
  state->m_z = 36969 * (state->m_z & 65535) + (state->m_z >> 16);
  state->m_w = 18000 * (state->m_w & 65535) + (state->m_w >> 16);
  return (state->m_z << 16) + state->m_w;  /* 32-bit result. */
}

static void RandomizeFreqs(RanState* state, size_t* freqs, int n) {
  int i;
  for (i = 0; i < n; i++) {
    if ((Ran(state) >> 4) % 3 == 0) freqs[i] = freqs[Ran(state) % n];
  }
}

static void RandomizeStatFreqs(RanState* state, SymbolStats* stats) {
  RandomizeFreqs(state, stats->litlens, ZOPFLI_NUM_LL);
  RandomizeFreqs(state, stats->dists, ZOPFLI_NUM_D);
  stats->litlens[256] = 1;  /* End symbol. */
}

static void ClearStatFreqs(SymbolStats* stats) {
  size_t i;
  for (i = 0; i < ZOPFLI_NUM_LL; i++) stats->litlens[i] = 0;
  for (i = 0; i < ZOPFLI_NUM_D; i++) stats->dists[i] = 0;
}

/*
Function that calculates a cost based on a model for the given LZ77 symbol.
litlen: means literal symbol if dist is 0, length otherwise.
*/
typedef double CostModelFun(unsigned litlen, unsigned dist, void* context);

/*
Cost model which should exactly match fixed tree.
type: CostModelFun
*/
static double GetCostFixed(unsigned litlen, unsigned dist, void* unused) {
  (void)unused;
  if (dist == 0) {
    if (litlen <= 143) return 8;
    else return 9;
  } else {
    int dbits = ZopfliGetDistExtraBits(dist);
    int lbits = ZopfliGetLengthExtraBits(litlen);
    int lsym = ZopfliGetLengthSymbol(litlen);
    int cost = 0;
    if (lsym <= 279) cost += 7;
    else cost += 8;
    cost += 5;  /* Every dist symbol has length 5. */
    return cost + dbits + lbits;
  }
}

/*
Cost model based on symbol statistics.
type: CostModelFun
*/
static double GetCostStat(unsigned litlen, unsigned dist, void* context) {
  SymbolStats* stats = (SymbolStats*)context;
  if (dist == 0) {
    return stats->ll_symbols[litlen];
  } else {
    int lsym = ZopfliGetLengthSymbol(litlen);
    int lbits = ZopfliGetLengthExtraBits(litlen);
    int dsym = ZopfliGetDistSymbol(dist);
    int dbits = ZopfliGetDistExtraBits(dist);
    return lbits + dbits + stats->ll_symbols[lsym] + stats->d_symbols[dsym];
  }
}

/*
Finds the minimum possible cost this cost model can return for valid length and
distance symbols.
*/
static double GetCostModelMinCost(CostModelFun* costmodel, void* costcontext) {
  double mincost;
  int bestlength = 0; /* length that has lowest cost in the cost model */
  int bestdist = 0; /* distance that has lowest cost in the cost model */
  int i;
  /*
  Table of distances that have a different distance symbol in the deflate
  specification. Each value is the first distance that has a new symbol. Only
  different symbols affect the cost model so only these need to be checked.
  See RFC 1951 section 3.2.5. Compressed blocks (length and distance codes).
  */
  static const int dsymbols[30] = {
    1, 2, 3, 4, 5, 7, 9, 13, 17, 25, 33, 49, 65, 97, 129, 193, 257, 385, 513,
    769, 1025, 1537, 2049, 3073, 4097, 6145, 8193, 12289, 16385, 24577
  };

  mincost = ZOPFLI_LARGE_FLOAT;
  for (i = 3; i < 259; i++) {
    double c = costmodel(i, 1, costcontext);
    if (c < mincost) {
      bestlength = i;
      mincost = c;
    }
  }

  mincost = ZOPFLI_LARGE_FLOAT;
  for (i = 0; i < 30; i++) {
    double c = costmodel(3, dsymbols[i], costcontext);
    if (c < mincost) {
      bestdist = dsymbols[i];
      mincost = c;
    }
  }

  return costmodel(bestlength, bestdist, costcontext);
}

static size_t zopfli_min(size_t a, size_t b) {
  return a < b ? a : b;
}

/*
Costs of a cost model for every literal and every (dist symbol, length) pair,
so that the inner loop of GetBestLengths is a table lookup rather than a call
of the cost model through a function pointer. The match cost is computed as
((lbits + dbits) + llcost) + dcost, which is the same expression, evaluated in
the same order, as in GetCostStat and GetCostFixed, so the results are
bit-exact.
*/
/*
RelaxLengths processes lengths four at a time, and may read and write back
unchanged up to this many entries past the last length. The cost tables and
the costs, length and dist arrays have this many extra entries.
*/
#define ZOPFLI_RELAX_PAD 3

typedef struct CostTables {
  double literal[256];  /* Cost of each literal byte. */
  /* Cost of each length, per dist symbol. */
  double match[30][259 + ZOPFLI_RELAX_PAD];
} CostTables;

static void InitCostTables(CostModelFun* costmodel, void* costcontext,
                           CostTables* t) {
  double llcost[259];
  double dcost[30];
  int i, k;
  if (costmodel == GetCostFixed) {
    for (i = 0; i < 256; i++) t->literal[i] = i <= 143 ? 8 : 9;
    for (k = 0; k < 259; k++) {
      llcost[k] = ZopfliGetLengthSymbol(k) <= 279 ? 7 : 8;
    }
    for (i = 0; i < 30; i++) dcost[i] = 5;
  } else {
    const SymbolStats* stats = (const SymbolStats*)costcontext;
    assert(costmodel == GetCostStat);
    for (i = 0; i < 256; i++) t->literal[i] = stats->ll_symbols[i];
    for (k = 0; k < 259; k++) {
      llcost[k] = stats->ll_symbols[ZopfliGetLengthSymbol(k)];
    }
    for (i = 0; i < 30; i++) dcost[i] = stats->d_symbols[i];
  }
  for (i = 0; i < 30; i++) {
    int dbits = ZopfliGetDistSymbolExtraBits(i);
    for (k = 0; k < 259; k++) {
      int lbits = ZopfliGetLengthExtraBits(k);
      t->match[i][k] = ((double)(lbits + dbits) + llcost[k]) + dcost[i];
    }
    for (; k < 259 + ZOPFLI_RELAX_PAD; k++) t->match[i][k] = 0;
  }
}

#ifdef ZOPFLI_SSE2
/*
Relaxes the four lengths k to k + 3, of which only those in the valid mask
(one 32-bit lane per length) may be updated, without branches per length:
whether each one improves is unpredictable. A length is updated when its cost
is above vmin and the new cost is lower, which are the tests of the scalar loop
in RelaxLengths, and the vector operations round exactly like the scalar ones,
so the result is the same. The lengths that don't improve are written back
unchanged.
*/
static void RelaxFourLengths(float* c4, unsigned short* l4, unsigned short* d4,
                             const double* match4, unsigned k,
                             __m128d vcostj, __m128d vmin, __m128i valid,
                             __m128i vdist) {
  __m128 c = _mm_loadu_ps(c4);
  __m128d clo = _mm_cvtps_pd(c);
  __m128d chi = _mm_cvtps_pd(_mm_movehl_ps(c, c));
  __m128d nlo = _mm_add_pd(_mm_loadu_pd(match4), vcostj);
  __m128d nhi = _mm_add_pd(_mm_loadu_pd(match4 + 2), vcostj);
  __m128d mlo = _mm_and_pd(_mm_cmpnle_pd(clo, vmin), _mm_cmplt_pd(nlo, clo));
  __m128d mhi = _mm_and_pd(_mm_cmpnle_pd(chi, vmin), _mm_cmplt_pd(nhi, chi));
  __m128 mf = _mm_and_ps(_mm_shuffle_ps(_mm_castpd_ps(mlo), _mm_castpd_ps(mhi),
                                        _MM_SHUFFLE(2, 0, 2, 0)),
                         _mm_castsi128_ps(valid));
  __m128 n;
  __m128i m16, vk, oldlen, olddist;
  /* Often none of the four improves, which is predictable enough to be worth
  skipping the stores for. */
  if (!_mm_movemask_ps(mf)) return;
  n = _mm_movelh_ps(_mm_cvtpd_ps(nlo), _mm_cvtpd_ps(nhi));
  m16 = _mm_packs_epi32(_mm_castps_si128(mf), _mm_castps_si128(mf));
  vk = _mm_add_epi16(_mm_set1_epi16((short)k), _mm_set_epi16(0, 0, 0, 0,
                                                             3, 2, 1, 0));
  oldlen = _mm_loadl_epi64((const __m128i*)l4);
  olddist = _mm_loadl_epi64((const __m128i*)d4);
  _mm_storeu_ps(c4, _mm_or_ps(_mm_and_ps(mf, n), _mm_andnot_ps(mf, c)));
  _mm_storel_epi64((__m128i*)l4, _mm_or_si128(_mm_and_si128(m16, vk),
                                              _mm_andnot_si128(m16, oldlen)));
  _mm_storel_epi64((__m128i*)d4, _mm_or_si128(_mm_and_si128(m16, vdist),
                                              _mm_andnot_si128(m16, olddist)));
}
#endif

/*
Relaxes the costs of reaching j + k from j for all lengths k in [kbegin, kend],
which all use the same distance dist.
*/
static void RelaxLengths(const CostTables* tables, size_t j,
                         unsigned kbegin, unsigned kend, unsigned short dist,
                         double mincostaddcostj, float* costs,
                         unsigned short* length_array,
                         unsigned short* dist_array) {
  const double* match = tables->match[ZopfliGetDistSymbol(dist)];
  double costj = costs[j];
  unsigned k = kbegin;
#ifdef ZOPFLI_SSE2
  {
    __m128d vcostj = _mm_set1_pd(costj);
    __m128d vmin = _mm_set1_pd(mincostaddcostj);
    __m128i vdist = _mm_set1_epi16((short)dist);
    for (; k + 3 <= kend; k += 4) {
      RelaxFourLengths(&costs[j + k], &length_array[j + k], &dist_array[j + k],
                       &match[k], k, vcostj, vmin, _mm_set1_epi32(-1), vdist);
    }
    if (k <= kend) {
      /* The last one to three lengths. The entries past kend exist thanks to
      ZOPFLI_RELAX_PAD, and are not modified. */
      __m128i valid = _mm_cmplt_epi32(_mm_set_epi32(3, 2, 1, 0),
                                      _mm_set1_epi32((int)(kend - k + 1)));
      RelaxFourLengths(&costs[j + k], &length_array[j + k], &dist_array[j + k],
                       &match[k], k, vcostj, vmin, valid, vdist);
    }
  }
#else
  for (; k <= kend; k++) {
    double newCost;
    /* Avoid the update if we are already at the minimum possible cost that
    the cost model can return. */
    if (costs[j + k] <= mincostaddcostj) continue;
    newCost = match[k] + costj;
    assert(newCost >= 0);
    if (newCost < costs[j + k]) {
      assert(k <= ZOPFLI_MAX_MATCH);
      costs[j + k] = (float)newCost;
      length_array[j + k] = (unsigned short)k;
      dist_array[j + k] = dist;
    }
  }
#endif
}

#ifdef ZOPFLI_LONGEST_MATCH_CACHE
/*
Returns whether the longest match cache has the complete result of
ZopfliFindLongestMatch, including all sublen values, for this position.
*/
static int CacheHasSublen(const ZopfliBlockState* s, size_t pos) {
  const ZopfliLongestMatchCache* lmc = s->lmc;
  size_t lmcpos = pos - s->blockstart;
  unsigned length;
  if (!lmc) return 0;
  length = lmc->length[lmcpos];
  /* Length > 0 and dist 0 indicates the cache value is not filled in yet. */
  if (length != 0 && lmc->dist[lmcpos] == 0) return 0;
  return length <= ZopfliMaxCachedSublen(lmc, lmcpos, length);
}

/*
If the longest match cache has the complete sublen information for this
position, relaxes all lengths directly from the cache and returns 1. This gives
the same result as ZopfliFindLongestMatch followed by relaxing each length with
its sublen distance, but without expanding the cache into a sublen array.
*/
static int RelaxLengthsFromCache(const ZopfliBlockState* s,
                                 const CostTables* tables,
                                 size_t pos, size_t j,
                                 double mincostaddcostj, float* costs,
                                 unsigned short* length_array,
                                 unsigned short* dist_array) {
  const ZopfliLongestMatchCache* lmc = s->lmc;
  size_t lmcpos = pos - s->blockstart;
  const unsigned char* cache;
  unsigned length, prevlength, e;
  if (!CacheHasSublen(s, pos)) return 0;
  length = lmc->length[lmcpos];
  if (length < ZOPFLI_MIN_MATCH) return 1;
  cache = &lmc->sublen[ZOPFLI_CACHE_LENGTH * lmcpos * 3];
  prevlength = ZOPFLI_MIN_MATCH;
  for (e = 0; e < ZOPFLI_CACHE_LENGTH; e++) {
    unsigned elength = cache[e * 3] + 3;
    unsigned short edist = cache[e * 3 + 1] + 256 * cache[e * 3 + 2];
    unsigned kend = elength < length ? elength : length;
    if (prevlength <= kend) {
      RelaxLengths(tables, j, prevlength, kend, edist, mincostaddcostj,
                   costs, length_array, dist_array);
    }
    if (elength >= length) break;
    prevlength = elength + 1;
  }
  return 1;
}
#else
static int CacheHasSublen(const ZopfliBlockState* s, size_t pos) {
  (void)s;
  (void)pos;
  return 0;
}
#endif

/*
The matches of the positions for which the longest match cache does not have
the complete result (for example more than ZOPFLI_CACHE_LENGTH different
distances, or a length limited by the end of the block), in the order the
forward pass visits them. Recorded during the first forward pass over a block.
With it, and the cache, later passes over the same block need neither the hash
nor ZopfliFindLongestMatch: they visit the same positions in the same order and
the match results only depend on the position.
*/
typedef struct MatchRecord {
  size_t* pos;  /* Position of each recorded match. */
  size_t numpos;
  size_t* runbegin;  /* Index of the first length run of each position. */
  size_t numrunbegin;
  unsigned short* runend;  /* Last length of each run of equal distance. */
  size_t numrunend;
  unsigned short* rundist;  /* Distance of each run. */
  size_t numrundist;
  size_t cursor;  /* Next position to replay. */
} MatchRecord;

static void InitMatchRecord(MatchRecord* rec) {
  memset(rec, 0, sizeof(*rec));
}

static void CleanMatchRecord(MatchRecord* rec) {
  free(rec->pos);
  free(rec->runbegin);
  free(rec->runend);
  free(rec->rundist);
}

#ifdef ZOPFLI_SHORTCUT_LONG_REPETITIONS
/*
Returns the amount of repetitions of the same byte after pos, the same value as
h->same of the hash has after updating it up to pos, but without needing the
hash. It uses the run of identical bytes [*runbegin, *runend) containing the
most recently queried position, which moves forward as pos advances. The run is
not tracked before instart. For a pos before the current run 0 is returned: the
true value is then smaller than the distance from pos to the current run, which
is all the caller needs to know.
*/
static unsigned SameAt(const unsigned char* in, size_t instart, size_t inend,
                       size_t* runbegin, size_t* runend, size_t pos) {
  size_t amount;
  if (pos >= *runend) {
    /* Find the run containing pos. The bytes scanned here are never scanned
    again, since the runs only move forward. */
    *runbegin = pos;
    while (*runbegin > instart && in[*runbegin - 1] == in[pos]) (*runbegin)--;
    *runend = pos + 1;
    while (*runend < inend && in[*runend] == in[pos]) (*runend)++;
  }
  if (pos < *runbegin) return 0;
  amount = *runend - pos - 1;
  return amount < 65535 ? (unsigned)amount : 65535;
}
#endif

/*
Performs the forward pass for "squeeze". Gets the most optimal length to reach
every byte from a previous byte, using cost calculations.
s: the ZopfliBlockState
in: the input data array
instart: where to start
inend: where to stop (not inclusive)
costmodel: function to calculate the cost of some lit/len/dist pair.
costcontext: abstract context for the costmodel function
length_array: output array of size (inend - instart) which will receive the best
    length to reach this byte from a previous byte.
dist_array: output array of size (inend - instart) which will receive the
    distance belonging to the length in length_array (0 for literals).
rec: if not null, either receives the matches the longest match cache can't
    reproduce (replay 0), or provides them (replay 1), in which case the hash is
    not used at all.
tables: scratch memory for the cost tables of the costmodel.
returns the cost that was, according to the costmodel, needed to get to the end.
*/
static double GetBestLengths(ZopfliBlockState *s,
                             const unsigned char* in,
                             size_t instart, size_t inend,
                             CostModelFun* costmodel, void* costcontext,
                             unsigned short* length_array,
                             unsigned short* dist_array,
                             MatchRecord* rec, int replay,
                             ZopfliHash* h, float* costs,
                             CostTables* tables) {
  /* Best cost to get here so far. */
  size_t blocksize = inend - instart;
  size_t i = 0, k, kend;
  unsigned short leng;
  unsigned short dist;
  unsigned short sublen[259];
  size_t windowstart = instart > ZOPFLI_WINDOW_SIZE
      ? instart - ZOPFLI_WINDOW_SIZE : 0;
  double result;
  double mincost = GetCostModelMinCost(costmodel, costcontext);
  double mincostaddcostj;
  int recording;
#ifdef ZOPFLI_SHORTCUT_LONG_REPETITIONS
  /* The run of identical bytes [runbegin, runend) last seen by SameAt. It is
  used instead of h->same, which is not available when replaying. */
  size_t runbegin = instart, runend = instart;
#endif

  if (instart == inend) return 0;

  InitCostTables(costmodel, costcontext, tables);
  if (rec) rec->cursor = 0;

  if (!replay) {
    ZopfliResetHash(ZOPFLI_WINDOW_SIZE, h);
    ZopfliWarmupHash(in, windowstart, inend, h);
    for (i = windowstart; i < instart; i++) {
      ZopfliUpdateHash(in, i, inend, h);
    }
  }

  for (i = 1; i < blocksize + 1 + ZOPFLI_RELAX_PAD; i++) {
    costs[i] = ZOPFLI_LARGE_FLOAT;
  }
  costs[0] = 0;  /* Because it's the start. */
  length_array[0] = 0;
  dist_array[0] = 0;

  for (i = instart; i < inend; i++) {
    size_t j = i - instart;  /* Index in the costs array and length_array. */
    if (!replay) ZopfliUpdateHash(in, i, inend, h);

#ifdef ZOPFLI_SHORTCUT_LONG_REPETITIONS
    /* If we're in a long repetition of the same character and have more than
    ZOPFLI_MAX_MATCH characters before and after our position. */
    if (i > instart + ZOPFLI_MAX_MATCH + 1
        && i + ZOPFLI_MAX_MATCH * 2 + 1 < inend
        && SameAt(in, instart, inend, &runbegin, &runend, i)
            > ZOPFLI_MAX_MATCH * 2
        && SameAt(in, instart, inend, &runbegin, &runend,
                  i - ZOPFLI_MAX_MATCH) > ZOPFLI_MAX_MATCH) {
      double symbolcost = costmodel(ZOPFLI_MAX_MATCH, 1, costcontext);
      /* Set the length to reach each one to ZOPFLI_MAX_MATCH, and the cost to
      the cost corresponding to that length. Doing this, we skip
      ZOPFLI_MAX_MATCH values to avoid calling ZopfliFindLongestMatch. */
      for (k = 0; k < ZOPFLI_MAX_MATCH; k++) {
        costs[j + ZOPFLI_MAX_MATCH] = costs[j] + symbolcost;
        length_array[j + ZOPFLI_MAX_MATCH] = ZOPFLI_MAX_MATCH;
        /* Inside such a repetition the match at distance 1 is the first one
        with the maximum length. */
        dist_array[j + ZOPFLI_MAX_MATCH] = 1;
        i++;
        j++;
        if (!replay) ZopfliUpdateHash(in, i, inend, h);
      }
    }
#endif

    /* Literal. */
    if (i + 1 <= inend) {
      double newCost = tables->literal[in[i]] + costs[j];
      assert(newCost >= 0);
      if (newCost < costs[j + 1]) {
        costs[j + 1] = (float)newCost;
        length_array[j + 1] = 1;
        dist_array[j + 1] = 0;
      }
    }

    /* Lengths. */
    mincostaddcostj = mincost + costs[j];
#ifdef ZOPFLI_LONGEST_MATCH_CACHE
    if (RelaxLengthsFromCache(s, tables, i, j, mincostaddcostj,
                              costs, length_array, dist_array)) {
      continue;
    }
#endif
    if (replay) {
      size_t r, rend;
      /* The record must belong to this block. */
      assert(rec->cursor < rec->numpos && rec->pos[rec->cursor] == i);
      r = rec->runbegin[rec->cursor];
      rend = rec->runbegin[rec->cursor + 1];
      k = 3;
      for (; r < rend; r++) {
        RelaxLengths(tables, j, (unsigned)k, rec->runend[r], rec->rundist[r],
                     mincostaddcostj, costs, length_array, dist_array);
        k = rec->runend[r] + 1;
      }
      rec->cursor++;
      continue;
    }
    ZopfliFindLongestMatch(s, h, in, i, inend, ZOPFLI_MAX_MATCH, sublen,
                           &dist, &leng);
    kend = zopfli_min(leng, inend-i);
    recording = rec && !CacheHasSublen(s, i);
    if (recording) {
      ZOPFLI_APPEND_DATA(i, &rec->pos, &rec->numpos);
      ZOPFLI_APPEND_DATA(rec->numrunend, &rec->runbegin, &rec->numrunbegin);
    }
    /* Relax runs of lengths that share the same distance together. */
    for (k = 3; k <= kend;) {
      size_t kgroupend = k;
      while (kgroupend < kend && sublen[kgroupend + 1] == sublen[k]) {
        kgroupend++;
      }
      RelaxLengths(tables, j, (unsigned)k, (unsigned)kgroupend, sublen[k],
                   mincostaddcostj, costs, length_array, dist_array);
      if (recording) {
        ZOPFLI_APPEND_DATA((unsigned short)kgroupend, &rec->runend,
                           &rec->numrunend);
        ZOPFLI_APPEND_DATA(sublen[k], &rec->rundist, &rec->numrundist);
      }
      k = kgroupend + 1;
    }
  }
  if (rec && !replay) {
    /* Terminate the run index, so that runbegin[n + 1] is always valid. */
    ZOPFLI_APPEND_DATA(rec->numrunend, &rec->runbegin, &rec->numrunbegin);
  }

  assert(costs[blocksize] >= 0);
  result = costs[blocksize];

  return result;
}

/*
Calculates the optimal path of lz77 lengths to use, from the calculated
length_array. The length_array must contain the optimal length to reach that
byte. The path will be filled with the lengths to use, so its data size will be
the amount of lz77 symbols.
*/
static void TraceBackwards(size_t size, const unsigned short* length_array,
                           unsigned short** path, size_t* pathsize) {
  size_t index = size;
  if (size == 0) return;
  for (;;) {
    ZOPFLI_APPEND_DATA(length_array[index], path, pathsize);
    assert(length_array[index] <= index);
    assert(length_array[index] <= ZOPFLI_MAX_MATCH);
    assert(length_array[index] != 0);
    index -= length_array[index];
    if (index == 0) break;
  }

  /* Mirror result. */
  for (index = 0; index < *pathsize / 2; index++) {
    unsigned short temp = (*path)[index];
    (*path)[index] = (*path)[*pathsize - index - 1];
    (*path)[*pathsize - index - 1] = temp;
  }
}

/*
Adds the LZ77 symbols of the path to the store. The distance of each length in
the path is the one GetBestLengths used for it, which is the same distance
ZopfliFindLongestMatch would find for that length limit, so no hash or match
finding is needed here.
*/
static void StorePath(const unsigned char* in, size_t instart,
                      const unsigned short* path, size_t pathsize,
                      const unsigned short* dist_array,
                      ZopfliLZ77Store* store) {
  size_t i;
  size_t j = 0;  /* Index in dist_array, relative to instart. */
  for (i = 0; i < pathsize; i++) {
    unsigned short length = path[i];
    size_t pos = instart + j;
    if (length >= ZOPFLI_MIN_MATCH) {
      unsigned short dist = dist_array[j + length];
      ZopfliVerifyLenDist(in, pos + length, pos, dist, length);
      ZopfliStoreLitLenDist(length, dist, pos, store);
    } else {
      length = 1;
      ZopfliStoreLitLenDist(in[pos], 0, pos, store);
    }
    j += length;
  }
}

/* Calculates the entropy of the statistics */
static void CalculateStatistics(SymbolStats* stats) {
  ZopfliCalculateEntropy(stats->litlens, ZOPFLI_NUM_LL, stats->ll_symbols);
  ZopfliCalculateEntropy(stats->dists, ZOPFLI_NUM_D, stats->d_symbols);
}

/* Appends the symbol statistics from the store. */
static void GetStatistics(const ZopfliLZ77Store* store, SymbolStats* stats) {
  size_t i;
  for (i = 0; i < store->size; i++) {
    if (store->dists[i] == 0) {
      stats->litlens[store->litlens[i]]++;
    } else {
      stats->litlens[ZopfliGetLengthSymbol(store->litlens[i])]++;
      stats->dists[ZopfliGetDistSymbol(store->dists[i])]++;
    }
  }
  stats->litlens[256] = 1;  /* End symbol. */

  CalculateStatistics(stats);
}

/*
Does a single run for ZopfliLZ77Optimal. For good compression, repeated runs
with updated statistics should be performed.
s: the block state
in: the input data array
instart: where to start
inend: where to stop (not inclusive)
path: pointer to dynamically allocated memory to store the path
pathsize: pointer to the size of the dynamic path array
length_array: array of size (inend - instart) used to store lengths
costmodel: function to use as the cost model for this squeeze run
costcontext: abstract context for the costmodel function
store: place to output the LZ77 data
returns the cost that was, according to the costmodel, needed to get to the end.
    This is not the actual cost.
*/
static double LZ77OptimalRun(ZopfliBlockState* s,
    const unsigned char* in, size_t instart, size_t inend,
    unsigned short** path, size_t* pathsize,
    unsigned short* length_array, unsigned short* dist_array,
    CostModelFun* costmodel, void* costcontext, ZopfliLZ77Store* store,
    MatchRecord* rec, int replay, ZopfliHash* h, float* costs,
    CostTables* tables) {
  double cost = GetBestLengths(s, in, instart, inend, costmodel,
                costcontext, length_array, dist_array, rec, replay, h, costs,
                tables);
  free(*path);
  *path = 0;
  *pathsize = 0;
  TraceBackwards(inend - instart, length_array, path, pathsize);
  StorePath(in, instart, *path, *pathsize, dist_array, store);
  assert(cost < ZOPFLI_LARGE_FLOAT);
  return cost;
}

void ZopfliLZ77Optimal(ZopfliBlockState *s,
                       const unsigned char* in, size_t instart, size_t inend,
                       int numiterations,
                       ZopfliLZ77Store* store) {
  /* Dist to get to here with smallest cost. */
  size_t blocksize = inend - instart;
  unsigned short* length_array =
      (unsigned short*)malloc(sizeof(unsigned short) *
                              (blocksize + 1 + ZOPFLI_RELAX_PAD));
  unsigned short* dist_array =
      (unsigned short*)malloc(sizeof(unsigned short) *
                              (blocksize + 1 + ZOPFLI_RELAX_PAD));
  unsigned short* path = 0;
  size_t pathsize = 0;
  ZopfliLZ77Store currentstore;
  ZopfliHash hash;
  ZopfliHash* h = &hash;
  /* Record the matches of the first run, so that the later runs can replay
  them without the hash. This needs the longest match cache. */
  MatchRecord rec;
  int userec = 0;
  SymbolStats stats, beststats, laststats;
  int i;
  float* costs =
      (float*)malloc(sizeof(float) * (blocksize + 1 + ZOPFLI_RELAX_PAD));
  CostTables* tables = (CostTables*)malloc(sizeof(CostTables));
  double cost;
  double bestcost = ZOPFLI_LARGE_FLOAT;
  double lastcost = 0;
  /* Try randomizing the costs a bit once the size stabilizes. */
  RanState ran_state;
  int lastrandomstep = -1;

  if (!costs || !tables) exit(-1); /* Allocation failed. */
  if (!length_array || !dist_array) exit(-1); /* Allocation failed. */

  InitRanState(&ran_state);
  InitStats(&stats);
  InitMatchRecord(&rec);
#ifdef ZOPFLI_LONGEST_MATCH_CACHE
  userec = s->lmc != 0;
#endif
  ZopfliInitLZ77Store(in, &currentstore);
  ZopfliAllocHash(ZOPFLI_WINDOW_SIZE, h);

  /* Do regular deflate, then loop multiple shortest path runs, each time using
  the statistics of the previous run. */

  /* Initial run. */
  ZopfliLZ77Greedy(s, in, instart, inend, &currentstore, h);
  GetStatistics(&currentstore, &stats);

  /* Repeat statistics with each time the cost model from the previous stat
  run. */
  for (i = 0; i < numiterations; i++) {
    ZopfliClearLZ77Store(&currentstore);
    LZ77OptimalRun(s, in, instart, inend, &path, &pathsize,
                   length_array, dist_array, GetCostStat, (void*)&stats,
                   &currentstore, userec ? &rec : 0, userec && i > 0, h, costs,
                   tables);
    cost = ZopfliCalculateBlockSize(&currentstore, 0, currentstore.size, 2);
    if (s->options->verbose_more || (s->options->verbose && cost < bestcost)) {
      fprintf(stderr, "Iteration %d: %d bit\n", i, (int) cost);
    }
    if (cost < bestcost) {
      /* Copy to the output store. */
      ZopfliCopyLZ77Store(&currentstore, store);
      CopyStats(&stats, &beststats);
      bestcost = cost;
    }
    CopyStats(&stats, &laststats);
    ClearStatFreqs(&stats);
    GetStatistics(&currentstore, &stats);
    if (lastrandomstep != -1) {
      /* This makes it converge slower but better. Do it only once the
      randomness kicks in so that if the user does few iterations, it gives a
      better result sooner. */
      AddWeighedStatFreqs(&stats, 1.0, &laststats, 0.5, &stats);
      CalculateStatistics(&stats);
    }
    if (i > 5 && cost == lastcost) {
      CopyStats(&beststats, &stats);
      RandomizeStatFreqs(&ran_state, &stats);
      CalculateStatistics(&stats);
      lastrandomstep = i;
    }
    lastcost = cost;
  }

  free(length_array);
  free(dist_array);
  free(path);
  free(costs);
  free(tables);
  CleanMatchRecord(&rec);
  ZopfliCleanLZ77Store(&currentstore);
  ZopfliCleanHash(h);
}

void ZopfliLZ77OptimalFixed(ZopfliBlockState *s,
                            const unsigned char* in,
                            size_t instart, size_t inend,
                            ZopfliLZ77Store* store)
{
  /* Dist to get to here with smallest cost. */
  size_t blocksize = inend - instart;
  unsigned short* length_array =
      (unsigned short*)malloc(sizeof(unsigned short) *
                              (blocksize + 1 + ZOPFLI_RELAX_PAD));
  unsigned short* dist_array =
      (unsigned short*)malloc(sizeof(unsigned short) *
                              (blocksize + 1 + ZOPFLI_RELAX_PAD));
  unsigned short* path = 0;
  size_t pathsize = 0;
  ZopfliHash hash;
  ZopfliHash* h = &hash;
  float* costs =
      (float*)malloc(sizeof(float) * (blocksize + 1 + ZOPFLI_RELAX_PAD));
  CostTables* tables = (CostTables*)malloc(sizeof(CostTables));

  if (!costs || !tables) exit(-1); /* Allocation failed. */
  if (!length_array || !dist_array) exit(-1); /* Allocation failed. */

  ZopfliAllocHash(ZOPFLI_WINDOW_SIZE, h);

  s->blockstart = instart;
  s->blockend = inend;

  /* Shortest path for fixed tree This one should give the shortest possible
  result for fixed tree, no repeated runs are needed since the tree is known. */
  LZ77OptimalRun(s, in, instart, inend, &path, &pathsize,
                 length_array, dist_array, GetCostFixed, 0, store, 0, 0, h,
                 costs, tables);

  free(length_array);
  free(dist_array);
  free(path);
  free(costs);
  free(tables);
  ZopfliCleanHash(h);
}
