/* leanrt_int.h — arbitrary-precision Int layer on top of leanrt_nat.h.
 * Int = heap object [ sign | mag ] where mag is a Nat object (sign 0 = nonneg, 1 = neg;
 * magnitude 0 is always stored with sign 0). Only lean_int_* touch these objects. */
#include "leanrt_nat.h"

/* Int object: [ sign (u32) | mag (pointer-width) ]. Store the pointer at full width so
 * this is correct on both the 32-bit target and a 64-bit host (for host validation). */
static lean_object* lrt_mkint(u32 sign, lean_object* mag){
  if (lrt_cmp(mag, lrt_box(0)) == 0) sign = 0;      /* canonical zero */
  u32* p = lrt_alloc(1 + sizeof(lrt_word)/4);
  p[0] = sign; *(lrt_word*)(p + 1) = (lrt_word)mag;
  return (lean_object*)p;
}
static u32          lrt_isign(lean_object* i){ return ((u32*)i)[0]; }
static lean_object* lrt_imag (lean_object* i){ return (lean_object*)(*(lrt_word*)(((u32*)i) + 1)); }

static lean_object* lean_nat_to_int(lean_object* n){ return lrt_mkint(0, n); }
static lean_object* lean_int_neg(lean_object* a){ return lrt_mkint(lrt_isign(a) ^ 1u, lrt_imag(a)); }
static lean_object* lean_nat_abs(lean_object* a){ return lrt_imag(a); }        /* |Int| : Nat */

static lean_object* lean_int_add(lean_object* a, lean_object* b){
  u32 sa = lrt_isign(a), sb = lrt_isign(b);
  lean_object *ma = lrt_imag(a), *mb = lrt_imag(b);
  if (sa == sb) return lrt_mkint(sa, lean_nat_add(ma, mb));
  int c = lrt_cmp(ma, mb);
  if (c == 0) return lrt_mkint(0, lrt_box(0));
  if (c > 0)  return lrt_mkint(sa, lean_nat_sub(ma, mb));   /* |a|>|b|, keep a's sign */
  return lrt_mkint(sb, lean_nat_sub(mb, ma));               /* |b|>|a|, keep b's sign */
}
static lean_object* lean_int_sub(lean_object* a, lean_object* b){ return lean_int_add(a, lean_int_neg(b)); }
static lean_object* lean_int_mul(lean_object* a, lean_object* b){
  return lrt_mkint(lrt_isign(a) ^ lrt_isign(b), lean_nat_mul(lrt_imag(a), lrt_imag(b)));
}
/* Euclidean division: 0 <= emod < |b|, and a = b*ediv + emod. */
static void lrt_ediv_emod(lean_object* a, lean_object* b, lean_object** eq, lean_object** er){
  u32 sa = lrt_isign(a), sb = lrt_isign(b);
  lean_object *nq, *nr; lrt_divmod(lrt_imag(a), lrt_imag(b), &nq, &nr);  /* |a| = |b|*nq + nr */
  int rzero = (lrt_cmp(nr, lrt_box(0)) == 0);
  if (sa == 0){                                  /* a >= 0 */
    *er = nr;
    *eq = lrt_mkint(sb, nq);                      /* sign(b) */
  } else if (rzero){                             /* a < 0, exact */
    *er = lrt_box(0);
    *eq = lrt_mkint(sb ^ 1u, nq);
  } else {                                        /* a < 0, remainder */
    *er = lean_nat_sub(lrt_imag(b), nr);          /* |b| - nr, in (0,|b|) */
    *eq = lrt_mkint(sb ^ 1u, lean_nat_add(nq, lrt_box(1)));
  }
  *eq = lrt_mkint(lrt_isign(*eq), lrt_imag(*eq)); /* canonicalize zero */
}
static lean_object* lean_int_ediv(lean_object* a, lean_object* b){ lean_object *q,*r; lrt_ediv_emod(a,b,&q,&r); return q; }
static lean_object* lean_int_emod(lean_object* a, lean_object* b){ lean_object *q,*r; lrt_ediv_emod(a,b,&q,&r); return lrt_mkint(0, r); }
/* comparisons */
static int lrt_icmp(lean_object* a, lean_object* b){
  u32 sa = lrt_isign(a), sb = lrt_isign(b);
  if (sa != sb) return sa ? -1 : 1;              /* neg < nonneg */
  int c = lrt_cmp(lrt_imag(a), lrt_imag(b));
  return sa ? -c : c;                            /* both neg: larger magnitude is smaller */
}
static u8 lean_int_dec_eq(lean_object* a, lean_object* b){ return lrt_icmp(a,b)==0; }
static u8 lean_int_dec_lt(lean_object* a, lean_object* b){ return lrt_icmp(a,b)<0; }
static u8 lean_int_dec_le(lean_object* a, lean_object* b){ return lrt_icmp(a,b)<=0; }

/* build Int from a signed decimal string (for test drivers) */
static lean_object* lrt_int_of_dec(const char* s){
  u32 sign = 0; if (*s=='-'){ sign=1; s++; }
  return lrt_mkint(sign, lrt_of_dec(s));
}
