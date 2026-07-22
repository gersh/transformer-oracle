/* leanrt_nat.h — a freestanding mini Lean runtime for arbitrary-precision Nat,
 * designed to compile through the C→RV32I→NISA pipeline and run Lean's VERBATIM
 * emitted C for Nat-heavy decision procedures on the transformer.
 *
 * Object model matches Lean's ABI as far as the emitted C observes it:
 *   - a `lean_object*` is either a TAGGED SCALAR (low bit 1, value = ptr>>1), used
 *     for Nat < 2^31, or a HEAP pointer (low bit 0) into a bump arena.
 * A heap Nat is [ nlimbs | limb0 | limb1 | ... ] little-endian base 2^32. Only the
 * lean_nat_* functions ever inspect a heap Nat, so this private layout is sufficient.
 */
typedef unsigned int  u32;
typedef unsigned char u8;
typedef unsigned char uint8_t;
typedef unsigned int  uint32_t;
#ifndef LEAN_EXPORT
#define LEAN_EXPORT
#endif
typedef unsigned char lean_object;   /* opaque; we cast to u32* for limb access */

/* ---- bump arena (all objects + scratch; no free, bounded computation) ---- */
#ifndef LEANRT_ARENA_WORDS
#define LEANRT_ARENA_WORDS 6000
#endif
static u32 lrt_arena[LEANRT_ARENA_WORDS];
static u32 lrt_ap = 0;
static u32* lrt_alloc(u32 words){ u32 i = lrt_ap; lrt_ap += words; return &lrt_arena[i]; }

/* ---- scalar tagging (use pointer-width int: u32 on the ilp32 target, 64-bit on host) ---- */
typedef unsigned long lrt_word;
static int      lrt_is_scalar(lean_object* o){ return ((lrt_word)o) & 1u; }
static lean_object* lrt_box(u32 n){ return (lean_object*)(((lrt_word)n << 1) | 1u); }
static u32      lrt_unbox(lean_object* o){ return (u32)(((lrt_word)o) >> 1); }

/* ---- limb view: copy the value of `o` into buf[], return #limbs (>=1) ---- */
static u32 lrt_limbs(lean_object* o, u32* buf){
  if (lrt_is_scalar(o)){ buf[0] = lrt_unbox(o); return 1u; }
  u32* p = (u32*)o; u32 n = p[0];
  for (u32 i = 0; i < n; i++) buf[i] = p[1 + i];
  return n ? n : 1u;
}
/* normalize limb array (strip leading zeros) and build a Nat object (scalar if it fits) */
static lean_object* lrt_mk(u32* limbs, u32 n){
  while (n > 1u && limbs[n-1] == 0u) n--;
  if (n == 1u && limbs[0] < 0x80000000u) return lrt_box(limbs[0]);
  u32* p = lrt_alloc(n + 1u); p[0] = n;
  for (u32 i = 0; i < n; i++) p[1+i] = limbs[i];
  return (lean_object*)p;
}

#define LRT_MAXL 24   /* up to ~768-bit intermediates (keeps NISA stack frames small) */

/* ---- comparison: -1 / 0 / 1 ---- */
static int lrt_cmp(lean_object* a, lean_object* b){
  u32 la[LRT_MAXL], lb[LRT_MAXL];
  u32 na = lrt_limbs(a, la), nb = lrt_limbs(b, lb);
  while (na > 1u && la[na-1]==0u) na--;
  while (nb > 1u && lb[nb-1]==0u) nb--;
  if (na != nb) return na < nb ? -1 : 1;
  for (u32 i = na; i-- > 0; ) if (la[i] != lb[i]) return la[i] < lb[i] ? -1 : 1;
  return 0;
}

/* ---- add ---- */
static lean_object* lean_nat_add(lean_object* a, lean_object* b){
  u32 la[LRT_MAXL], lb[LRT_MAXL], r[LRT_MAXL+1];
  u32 na = lrt_limbs(a, la), nb = lrt_limbs(b, lb);
  u32 n = na > nb ? na : nb; u32 carry = 0;
  for (u32 i = 0; i < n; i++){
    unsigned long long s = (unsigned long long)(i<na?la[i]:0u) + (i<nb?lb[i]:0u) + carry;
    r[i] = (u32)s; carry = (u32)(s >> 32);
  }
  r[n] = carry; return lrt_mk(r, n + 1u);
}
/* ---- truncated subtraction: max(a-b,0) (Nat semantics) ---- */
static lean_object* lean_nat_sub(lean_object* a, lean_object* b){
  if (lrt_cmp(a, b) <= 0) return lrt_box(0);
  u32 la[LRT_MAXL], lb[LRT_MAXL], r[LRT_MAXL];
  u32 na = lrt_limbs(a, la), nb = lrt_limbs(b, lb);
  u32 borrow = 0;
  for (u32 i = 0; i < na; i++){
    unsigned long long bi = (i<nb?lb[i]:0u) + (unsigned long long)borrow;
    unsigned long long ai = la[i];
    if (ai >= bi){ r[i] = (u32)(ai - bi); borrow = 0; }
    else { r[i] = (u32)((ai + 0x100000000ULL) - bi); borrow = 1; }
  }
  return lrt_mk(r, na);
}
/* ---- multiply (schoolbook) ---- */
static lean_object* lean_nat_mul(lean_object* a, lean_object* b){
  u32 la[LRT_MAXL], lb[LRT_MAXL], r[2*LRT_MAXL];
  u32 na = lrt_limbs(a, la), nb = lrt_limbs(b, lb);
  for (u32 i = 0; i < na+nb; i++) r[i] = 0u;
  for (u32 i = 0; i < na; i++){
    u32 carry = 0;
    for (u32 j = 0; j < nb; j++){
      unsigned long long t = (unsigned long long)la[i]*lb[j] + r[i+j] + carry;
      r[i+j] = (u32)t; carry = (u32)(t >> 32);
    }
    r[i+nb] += carry;
  }
  return lrt_mk(r, na + nb);
}
/* ---- divmod via binary long division: sets *qout,*rout = a/b, a%b ---- */
static void lrt_divmod(lean_object* a, lean_object* b, lean_object** qout, lean_object** rout){
  if (lrt_cmp(b, lrt_box(0)) == 0){ *qout = lrt_box(0); *rout = a; return; } /* Lean: /0=0, %0=a */
  if (lrt_cmp(a, b) < 0){ *qout = lrt_box(0); *rout = a; return; }
  u32 la[LRT_MAXL]; u32 na = lrt_limbs(a, la);
  u32 q[LRT_MAXL], rem[LRT_MAXL+1];
  for (u32 i=0;i<na;i++) q[i]=0u;
  u32 rn = 1; rem[0] = 0u;
  u32 lb[LRT_MAXL]; u32 nb = lrt_limbs(b, lb);
  for (u32 bit = na*32u; bit-- > 0; ){
    /* rem = (rem << 1) | a_bit */
    u32 carry = (la[bit>>5] >> (bit & 31)) & 1u;
    for (u32 i = 0; i < rn; i++){ u32 nc = rem[i] >> 31; rem[i] = (rem[i] << 1) | carry; carry = nc; }
    if (carry){ rem[rn++] = carry; }
    /* if rem >= b: rem -= b; set quotient bit */
    /* compare rem (rn limbs) vs b (nb limbs) */
    u32 cmp; { u32 rr=rn; while(rr>1&&rem[rr-1]==0)rr--; u32 bb=nb; while(bb>1&&lb[bb-1]==0)bb--;
      if(rr!=bb) cmp = rr<bb?0:2; else { cmp=1; for(u32 i=rr;i-->0;){ if(rem[i]!=lb[i]){cmp=rem[i]<lb[i]?0:2;break;} } } }
    if (cmp >= 1){ /* rem >= b */
      u32 borrow=0; for(u32 i=0;i<rn;i++){ unsigned long long bi=(i<nb?lb[i]:0u)+(unsigned long long)borrow; unsigned long long ri=rem[i];
        if(ri>=bi){rem[i]=(u32)(ri-bi);borrow=0;} else {rem[i]=(u32)((ri+0x100000000ULL)-bi);borrow=1;} }
      while(rn>1&&rem[rn-1]==0)rn--;
      q[bit>>5] |= (1u << (bit & 31));
    }
  }
  *qout = lrt_mk(q, na); *rout = lrt_mk(rem, rn);
}
static lean_object* lean_nat_div(lean_object* a, lean_object* b){ lean_object *q,*r; lrt_divmod(a,b,&q,&r); return q; }
static lean_object* lean_nat_mod(lean_object* a, lean_object* b){ lean_object *q,*r; lrt_divmod(a,b,&q,&r); return r; }
/* ---- pow: a^b by repeated squaring (b as Nat) ---- */
static lean_object* lean_nat_pow(lean_object* a, lean_object* b){
  u32 lb[LRT_MAXL]; u32 nb = lrt_limbs(b, lb);
  /* exponent bit-length (stop squaring past the top set bit to avoid blow-up) */
  u32 topbit = 0, found = 0;
  for (u32 bit = nb*32u; bit-- > 0; ){ if ((lb[bit>>5] >> (bit&31)) & 1u){ topbit = bit; found = 1; break; } }
  if (!found) return lrt_box(1);   /* b == 0 */
  lean_object* result = lrt_box(1); lean_object* base = a;
  for (u32 bit = 0; bit <= topbit; bit++){
    if ((lb[bit>>5] >> (bit&31)) & 1u) result = lean_nat_mul(result, base);
    if (bit < topbit) base = lean_nat_mul(base, base);
  }
  return result;
}
/* ---- predicates ---- */
static u8 lean_nat_dec_eq(lean_object* a, lean_object* b){ return lrt_cmp(a,b)==0; }
static u8 lean_nat_dec_lt(lean_object* a, lean_object* b){ return lrt_cmp(a,b)<0; }
static u8 lean_nat_dec_le(lean_object* a, lean_object* b){ return lrt_cmp(a,b)<=0; }
/* ---- constructors / misc ---- */
static lean_object* lean_unsigned_to_nat(u32 n){ if(n<0x80000000u) return lrt_box(n); u32 l=n; return lrt_mk(&l,1u); }
static lean_object* lean_box(u32 n){ return lrt_box(n); }
static u32 lean_unbox(lean_object* o){ return lrt_unbox(o); }
static void lean_dec(lean_object* o){ (void)o; }
static void lean_dec_ref(lean_object* o){ (void)o; }
static void lean_inc(lean_object* o){ (void)o; }
static lean_object* lean_nat_land(lean_object* a, lean_object* b){
  u32 la[LRT_MAXL], lb[LRT_MAXL], r[LRT_MAXL]; u32 na=lrt_limbs(a,la),nb=lrt_limbs(b,lb);
  u32 n=na<nb?na:nb; for(u32 i=0;i<n;i++) r[i]=la[i]&lb[i]; return lrt_mk(r,n);
}

/* build a Nat from a decimal string (for constructing test inputs) */
static lean_object* lrt_of_dec(const char* s){
  lean_object* acc = lrt_box(0); lean_object* ten = lrt_box(10);
  for (const char* p = s; *p; p++){ acc = lean_nat_add(lean_nat_mul(acc, ten), lrt_box((u32)(*p - '0'))); }
  return acc;
}
