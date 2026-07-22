/* leanrt_ctor.h — boxed-structure model on top of leanrt_int.h. Unlocks inductives
 * (List, Option, Prod, records) for structural-recursion decision procedures.
 * ctor object layout: [ tag (u32) | nfields (u32) | field0 | field1 | ... ] with each
 * field stored pointer-width. Only the lean_ctor accessors and lean_obj_tag touch these, so this
 * private layout coexists with the Nat/Int layouts. List: nil = lean_box(0) (scalar tag 0),
 * cons h t = ctor tag 1 {h, t}. */
#include "leanrt_int.h"

#define LRT_W (sizeof(lrt_word)/4)   /* words per pointer field: 1 on target, 2 on host */

static lean_object* lean_alloc_ctor(u32 tag, u32 nfields, u32 scalar_sz){
  u32* p = lrt_alloc(2 + nfields*LRT_W + (scalar_sz + 3)/4);
  p[0] = tag; p[1] = nfields;
  return (lean_object*)p;
}
static lean_object* lean_ctor_get(lean_object* o, u32 i){
  return (lean_object*)(*(lrt_word*)(((u32*)o) + 2 + i*LRT_W));
}
static void lean_ctor_set(lean_object* o, u32 i, lean_object* v){
  *(lrt_word*)(((u32*)o) + 2 + i*LRT_W) = (lrt_word)v;
}
static u32 lean_obj_tag(lean_object* o){
  return lrt_is_scalar(o) ? lrt_unbox(o) : ((u32*)o)[0];
}
/* Lean also emits lean_ctor_set_tag / lean_ctor_release in some paths (no-op-safe here) */
static void lean_ctor_release(lean_object* o, u32 i){ (void)o; (void)i; }

/* build a List Nat from a C array (nil = box 0, cons = ctor tag 1 {head, tail}) */
static lean_object* lrt_list_of(lean_object** items, u32 n){
  lean_object* l = lrt_box(0);            /* [] */
  for (u32 i = n; i-- > 0; ){
    lean_object* c = lean_alloc_ctor(1, 2, 0);
    lean_ctor_set(c, 0, items[i]);        /* head */
    lean_ctor_set(c, 1, l);               /* tail */
    l = c;
  }
  return l;
}
