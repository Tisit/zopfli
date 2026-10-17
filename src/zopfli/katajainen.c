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

/*
Bounded package merge algorithm, based on the paper
"A Fast and Space-Economical Algorithm for Length-Limited Coding
Jyrki Katajainen, Alistair Moffat, Andrew Turpin".
*/

#include "katajainen.h"
#include <assert.h>
#include <stdlib.h>
#include <limits.h>

/* Alphabets up to this size and code length use memory on the stack. */
#define SMALL_ALPHABET 32
#define SMALL_MAXBITS 15

/*
A leaf: one symbol with its weight (frequency).
*/
typedef struct Node {
  size_t weight;  /* Symbol count. */
  int count;  /* Index of the symbol this leaf represents. */
} Node;

/*
Computes the length limited code lengths with the package-merge algorithm.
leaves: the used symbols, sorted from lightest to heaviest.
numsymbols: amount of leaves, at least 3.
maxbits: the maximum code length, at most numsymbols - 1, and with
    (1 << maxbits) >= numsymbols.
bitlengths: receives the code lengths, must be initialized to 0.

List 0 holds the leaves. List i merges the leaves with the packages of list
i - 1, the sums of its consecutive pairs of items, a package going before a leaf
of equal weight. The first 2 * numsymbols - 2 items of the last list are
selected, and the selected packages of each list select twice as many items of
the list before. The code length of a leaf is the number of lists in which it
is selected. This builds the lists completely with plain arrays, giving the
same result as the boundary package-merge of the paper, which creates only the
needed parts of the lists lazily with a recursion over linked chains, but
mispredicts many more branches doing so.
*/
static void PackageMerge(const Node* leaves, int numsymbols, int maxbits,
                         unsigned* bitlengths) {
  /* Every list has fewer items than this. */
  int maxlen = 2 * numsymbols;
  /* Size of the weight arrays below, and of numleaves. */
  size_t bufsize = 4 * maxlen + 4;
  size_t nlsize = maxbits * (maxlen + 1);
  /* For small alphabets, such as the 19 code length codes, the memory is on
  the stack, since this is called very often. */
  size_t localbuffer[4 * 2 * SMALL_ALPHABET + 4];
  int localnumleaves[SMALL_MAXBITS * (2 * SMALL_ALPHABET + 1)];
  int small = numsymbols <= SMALL_ALPHABET && maxbits <= SMALL_MAXBITS;
  /* The leaf weights and the package weights of the previous list, each with a
  sentinel before and after, and the previous and current list. */
  size_t* buffer = small ? localbuffer
      : (size_t*)malloc(bufsize * sizeof(size_t) + nlsize * sizeof(int));
  size_t* weights = buffer + 1;
  size_t* packages = weights + maxlen + 2;
  size_t* prev = packages + maxlen + 1;
  size_t* cur = prev + maxlen;
  /* numleaves[i * (maxlen + 1) + t]: amount of leaves among the first t items
  of list i. */
  int* numleaves = small ? localnumleaves : (int*)(buffer + bufsize);
  int prevlen, i, t;
  int selected;

  if (!buffer) exit(-1);  /* Allocation failed. */

  for (t = 0; t < numsymbols; t++) weights[t] = leaves[t].weight;
  /* Sentinels, lighter and heavier than any real weight (which are nonzero
  and far below the maximum), so that the merges below never take a leaf or
  package past either end. */
  weights[-1] = 0;
  weights[numsymbols] = (size_t)(-1);
  packages[-1] = 0;
  for (t = 0; t < numsymbols; t++) prev[t] = weights[t];
  for (t = 0; t <= numsymbols; t++) numleaves[t] = t;
  prevlen = numsymbols;

  for (i = 1; i < maxbits; i++) {
    int* nl = numleaves + i * (maxlen + 1);
    int numpackages = prevlen / 2;
    int total = numsymbols + numpackages;
    int half = (total + 1) / 2;
    int leaf = 0, package = 0;  /* Next ones from the front. */
    int leafback = numsymbols - 1, packageback = numpackages - 1;
    size_t* swap;
    for (t = 0; t < numpackages; t++) {
      packages[t] = prev[2 * t] + prev[2 * t + 1];
    }
    packages[numpackages] = (size_t)(-1);
    nl[0] = 0;
    nl[total] = numsymbols;
    /* Merge without branches, since which one comes next is unpredictable,
    from both ends at once, since each step depends on the previous one. From
    the front, a package goes before a leaf of equal weight, so from the back
    the leaf goes first. Both halves meet exactly in the middle. */
    for (t = 0; t < total - half; t++) {
      size_t p = packages[package];
      size_t l = weights[leaf];
      int takepackage = p <= l;
      size_t pb = packages[packageback];
      size_t lb = weights[leafback];
      int takepackageback = pb > lb;
      int tb = total - 1 - t;
      cur[t] = takepackage ? p : l;
      package += takepackage;
      leaf += 1 - takepackage;
      nl[t + 1] = leaf;
      cur[tb] = takepackageback ? pb : lb;
      packageback -= takepackageback;
      leafback -= 1 - takepackageback;
      nl[tb] = leafback + 1;
    }
    if (half > total - half) {
      /* The middle item, when the total is odd. */
      size_t p = packages[package];
      size_t l = weights[leaf];
      int takepackage = p <= l;
      cur[t] = takepackage ? p : l;
      leaf += 1 - takepackage;
      nl[t + 1] = leaf;
    }
    swap = prev;
    prev = cur;
    cur = swap;
    prevlen = total;
  }

  selected = 2 * numsymbols - 2;
  for (i = maxbits - 1; i >= 0; i--) {
    int c = numleaves[i * (maxlen + 1) + selected];
    for (t = 0; t < c; t++) bitlengths[leaves[t].count]++;
    selected = 2 * (selected - c);
  }

  if (!small) free(buffer);
}

/*
Sorts the leaves by weight, from lightest to heaviest. The weights must be
unique, so that the result does not depend on the sorting algorithm. This is a
bottom-up merge sort, which is faster than qsort with a comparator callback for
these small arrays. tmp must have room for n leaves.
*/
static void SortLeaves(Node* leaves, Node* tmp, int n) {
  Node* src = leaves;
  Node* dst = tmp;
  int width, i;
  /* Insertion sort runs of 8 first. */
  for (i = 0; i < n; i += 8) {
    int end = i + 8 < n ? i + 8 : n;
    int a;
    for (a = i + 1; a < end; a++) {
      Node v = leaves[a];
      int b = a;
      while (b > i && leaves[b - 1].weight > v.weight) {
        leaves[b] = leaves[b - 1];
        b--;
      }
      leaves[b] = v;
    }
  }
  for (width = 8; width < n; width *= 2) {
    Node* swap;
    for (i = 0; i < n; i += 2 * width) {
      int mid = i + width < n ? i + width : n;
      int end = i + 2 * width < n ? i + 2 * width : n;
      int a = i, b = mid, o = i;
      while (a < mid && b < end) {
        dst[o++] = src[b].weight < src[a].weight ? src[b++] : src[a++];
      }
      while (a < mid) dst[o++] = src[a++];
      while (b < end) dst[o++] = src[b++];
    }
    swap = src;
    src = dst;
    dst = swap;
  }
  if (src != leaves) {
    for (i = 0; i < n; i++) leaves[i] = src[i];
  }
}

int ZopfliLengthLimitedCodeLengths(
    const size_t* frequencies, int n, int maxbits, unsigned* bitlengths) {
  int i;
  int numsymbols = 0;  /* Amount of symbols with frequency > 0. */

  /* One leaf per symbol. Only numsymbols leaves will be used. The second half
  is scratch space for sorting. */
  Node localleaves[2 * SMALL_ALPHABET];
  Node* leaves = n <= SMALL_ALPHABET ? localleaves
      : (Node*)malloc(2 * n * sizeof(*leaves));

  /* Initialize all bitlengths at 0. */
  for (i = 0; i < n; i++) {
    bitlengths[i] = 0;
  }

  /* Count used symbols and place them in the leaves. */
  for (i = 0; i < n; i++) {
    if (frequencies[i]) {
      leaves[numsymbols].weight = frequencies[i];
      leaves[numsymbols].count = i;  /* Index of symbol this leaf represents. */
      numsymbols++;
    }
  }

  /* Check special cases and error conditions. */
  if ((1 << maxbits) < numsymbols) {
    if (leaves != localleaves) free(leaves);
    return 1;  /* Error, too few maxbits to represent symbols. */
  }
  if (numsymbols == 0) {
    if (leaves != localleaves) free(leaves);
    return 0;  /* No symbols at all. OK. */
  }
  if (numsymbols == 1) {
    bitlengths[leaves[0].count] = 1;
    if (leaves != localleaves) free(leaves);
    return 0;  /* Only one symbol, give it bitlength 1, not 0. OK. */
  }
  if (numsymbols == 2) {
    bitlengths[leaves[0].count]++;
    bitlengths[leaves[1].count]++;
    if (leaves != localleaves) free(leaves);
    return 0;
  }

  /* Sort the leaves from lightest to heaviest. Add count into the same
  variable for stable sorting. */
  for (i = 0; i < numsymbols; i++) {
    if (leaves[i].weight >=
        ((size_t)1 << (sizeof(leaves[0].weight) * CHAR_BIT - 9))) {
      if (leaves != localleaves) free(leaves);
      return 1;  /* Error, we need 9 bits for the count. */
    }
    leaves[i].weight = (leaves[i].weight << 9) | leaves[i].count;
  }
  SortLeaves(leaves, leaves + numsymbols, numsymbols);
  for (i = 0; i < numsymbols; i++) {
    leaves[i].weight >>= 9;
  }

  if (numsymbols - 1 < maxbits) {
    maxbits = numsymbols - 1;
  }

  PackageMerge(leaves, numsymbols, maxbits, bitlengths);

  if (leaves != localleaves) free(leaves);
  return 0;  /* OK. */
}
