/*
* jidctint-rvv.c - accurate integer IDCT (RISC-V RVV)
*
* Copyright (c) 2012-2024 Andes Technology Corporation
* All rights reserved.
*/
/*
 * jidctint-neon.c - accurate integer IDCT (Arm Neon)
 *
 * Copyright (C) 2020, Arm Limited.  All Rights Reserved.
 * Copyright (C) 2020, D. R. Commander.  All Rights Reserved.
 *
 * This software is provided 'as-is', without any express or implied
 * warranty.  In no event will the authors be held liable for any damages
 * arising from the use of this software.
 *
 * Permission is granted to anyone to use this software for any purpose,
 * including commercial applications, and to alter it and redistribute it
 * freely, subject to the following restrictions:
 *
 * 1. The origin of this software must not be misrepresented; you must not
 *    claim that you wrote the original software. If you use this software
 *    in a product, an acknowledgment in the product documentation would be
 *    appreciated but is not required.
 * 2. Altered source versions must be plainly marked as such, and must not be
 *    misrepresented as being the original software.
 * 3. This notice may not be removed or altered from any source distribution.
 */

#define JPEG_INTERNALS
#include "../../jinclude.h"
#include "../../jpeglib.h"
#include "../../jsimd.h"
#include "../../jdct.h"
#include "../../jsimddct.h"
#include "../jsimd.h"

#include <riscv_vector.h>


#define CONST_BITS  13
#define PASS1_BITS  2

#define DESCALE_P1  (CONST_BITS - PASS1_BITS)
#define DESCALE_P2  (CONST_BITS + PASS1_BITS + 3)

/* The computation of the inverse DCT requires the use of constants known at
 * compile time.  Scaled integer constants are used to avoid floating-point
 * arithmetic:
 *    0.298631336 =  2446 * 2^-13
 *    0.390180644 =  3196 * 2^-13
 *    0.541196100 =  4433 * 2^-13
 *    0.765366865 =  6270 * 2^-13
 *    0.899976223 =  7373 * 2^-13
 *    1.175875602 =  9633 * 2^-13
 *    1.501321110 = 12299 * 2^-13
 *    1.847759065 = 15137 * 2^-13
 *    1.961570560 = 16069 * 2^-13
 *    2.053119869 = 16819 * 2^-13
 *    2.562915447 = 20995 * 2^-13
 *    3.072711026 = 25172 * 2^-13
 */

#define F_0_298  2446
#define F_0_390  3196
#define F_0_541  4433
#define F_0_765  6270
#define F_0_899  7373
#define F_1_175  9633
#define F_1_501  12299
#define F_1_847  15137
#define F_1_961  16069
#define F_2_053  16819
#define F_2_562  20995
#define F_3_072  25172

#define F_1_175_MINUS_1_961  (F_1_175 - F_1_961)
#define F_1_175_MINUS_0_390  (F_1_175 - F_0_390)
#define F_0_541_MINUS_1_847  (F_0_541 - F_1_847)
#define F_3_072_MINUS_2_562  (F_3_072 - F_2_562)
#define F_0_298_MINUS_0_899  (F_0_298 - F_0_899)
#define F_1_501_MINUS_0_899  (F_1_501 - F_0_899)
#define F_2_053_MINUS_2_562  (F_2_053 - F_2_562)
#define F_0_541_PLUS_0_765   (F_0_541 + F_0_765)


static const int16_t idct_islow_consts[] = {
  F_0_899,             F_0_541,
  F_2_562,             F_0_298_MINUS_0_899,
  F_1_501_MINUS_0_899, F_2_053_MINUS_2_562,
  F_0_541_PLUS_0_765,  F_1_175,
  F_1_175_MINUS_0_390, F_0_541_MINUS_1_847,
  F_3_072_MINUS_2_562, F_1_175_MINUS_1_961,
  0, 0, 0, 0
};

#define TRANS_TABLE_U8_SIZE 64
const uint8_t trans_index8x8_u8[TRANS_TABLE_U8_SIZE] =
{
/* #0  #1  #2  #3  #7  #6  #5  #4  */
	0,  8, 16, 24, 56, 48, 40, 32,
	1,  9, 17, 25, 57, 49, 41, 33,
	2, 10, 18, 26, 58, 50, 42, 34,
	3, 11, 19, 27, 59, 51, 43, 35,
	4, 12, 20, 28, 60, 52, 44, 36,
	5, 13, 21, 29, 61, 53, 45, 37,
	6, 14, 22, 30, 62, 54, 46, 38,
	7, 15, 23, 31, 63, 55, 47, 39,
};


/* Forward declaration of regular and sparse IDCT helper functions */

static INLINE void jsimd_idct_islow_pass1_regular(vint16m1_t row0,
                                                  vint16m1_t row1,
                                                  vint16m1_t row2,
                                                  vint16m1_t row3,
                                                  vint16m1_t row4,
                                                  vint16m1_t row5,
                                                  vint16m1_t row6,
                                                  vint16m1_t row7,
                                                  vint16m1_t quant_row0,
                                                  vint16m1_t quant_row1,
                                                  vint16m1_t quant_row2,
                                                  vint16m1_t quant_row3,
                                                  vint16m1_t quant_row4,
                                                  vint16m1_t quant_row5,
                                                  vint16m1_t quant_row6,
                                                  vint16m1_t quant_row7,
												  vint16m8_t *cols_all_i16m8);

static INLINE void jsimd_idct_islow_pass2_regular(vint16m8_t *cols_all_i16m8,
                                                  JSAMPARRAY output_buf,
                                                  JDIMENSION output_col);


/* Perform dequantization and inverse DCT on one block of coefficients.  For
 * reference, the C implementation (jpeg_idct_slow()) can be found in
 * jidctint.c.
 */

void jsimd_idct_islow_rvv(void *dct_table, JCOEFPTR coef_block,
                           JSAMPARRAY output_buf, JDIMENSION output_col)
{
  ISLOW_MULT_TYPE *quantptr = dct_table;
  vint16m8_t cols_all_i16m8;

  /* Compute IDCT first pass on left 4x8 coefficient block. */

  /* Load DCT coefficients. */
  size_t vl = 8;
  vint16m1_t row0 = __riscv_vle16_v_i16m1(coef_block + 0 * DCTSIZE, vl);
  vint16m1_t row1 = __riscv_vle16_v_i16m1(coef_block + 1 * DCTSIZE, vl);
  vint16m1_t row2 = __riscv_vle16_v_i16m1(coef_block + 2 * DCTSIZE, vl);
  vint16m1_t row3 = __riscv_vle16_v_i16m1(coef_block + 3 * DCTSIZE, vl);
  vint16m1_t row4 = __riscv_vle16_v_i16m1(coef_block + 4 * DCTSIZE, vl);
  vint16m1_t row5 = __riscv_vle16_v_i16m1(coef_block + 5 * DCTSIZE, vl);
  vint16m1_t row6 = __riscv_vle16_v_i16m1(coef_block + 6 * DCTSIZE, vl);
  vint16m1_t row7 = __riscv_vle16_v_i16m1(coef_block + 7 * DCTSIZE, vl);

  /* Load quantization table. */
  vint16m1_t quant_row0 = __riscv_vle16_v_i16m1(quantptr + 0 * DCTSIZE, vl);
  vint16m1_t quant_row1 = __riscv_vle16_v_i16m1(quantptr + 1 * DCTSIZE, vl);
  vint16m1_t quant_row2 = __riscv_vle16_v_i16m1(quantptr + 2 * DCTSIZE, vl);
  vint16m1_t quant_row3 = __riscv_vle16_v_i16m1(quantptr + 3 * DCTSIZE, vl);
  vint16m1_t quant_row4 = __riscv_vle16_v_i16m1(quantptr + 4 * DCTSIZE, vl);
  vint16m1_t quant_row5 = __riscv_vle16_v_i16m1(quantptr + 5 * DCTSIZE, vl);
  vint16m1_t quant_row6 = __riscv_vle16_v_i16m1(quantptr + 6 * DCTSIZE, vl);
  vint16m1_t quant_row7 = __riscv_vle16_v_i16m1(quantptr + 7 * DCTSIZE, vl);

  /* Construct bitmap to test if DCT coefficients are 0. */
  vuint16m1_t vec_zero = __riscv_vmv_s_x_u16m1(0, vl);
  vint16m1_t bitmap = __riscv_vor_vv_i16m1(row7, row6, vl);
  bitmap = __riscv_vor_vv_i16m1(bitmap, row5, vl);
  bitmap = __riscv_vor_vv_i16m1(bitmap, row4, vl);
  bitmap = __riscv_vor_vv_i16m1(bitmap, row3, vl);
  bitmap = __riscv_vor_vv_i16m1(bitmap, row2, vl);
  bitmap = __riscv_vor_vv_i16m1(bitmap, row1, vl);
  vuint16m1_t tmp_u16m1 = __riscv_vredor_vs_u16m1_u16m1(__riscv_vreinterpret_v_i16m1_u16m1(bitmap), vec_zero, vl);
  uint16_t ac_bitmap = __riscv_vmv_x_s_u16m1_u16(tmp_u16m1);
  if (0 == ac_bitmap) {
    vint16m1_t dcval_i16m1 = __riscv_vmul_vv_i16m1(row0, quant_row0, vl);
    dcval_i16m1 = __riscv_vsll_vx_i16m1(dcval_i16m1, PASS1_BITS, vl);

	// combine vectors
	vint16m4_t tmp_i16m4 = __riscv_vlmul_ext_v_i16m1_i16m4(dcval_i16m1);
	tmp_i16m4 = __riscv_vslideup_vx_i16m4(tmp_i16m4, tmp_i16m4, DCTSIZE2/8, DCTSIZE2/4);
	tmp_i16m4 = __riscv_vslideup_vx_i16m4(tmp_i16m4, tmp_i16m4, DCTSIZE2/4, DCTSIZE2/2);
	cols_all_i16m8  = __riscv_vlmul_ext_v_i16m4_i16m8(tmp_i16m4);
	cols_all_i16m8 = __riscv_vslideup_vx_i16m8(cols_all_i16m8, cols_all_i16m8, DCTSIZE2/2, DCTSIZE2);
  } else {
    jsimd_idct_islow_pass1_regular(row0, row1, row2, row3, row4, row5,
                                   row6, row7, quant_row0, quant_row1,
                                   quant_row2, quant_row3, quant_row4,
                                   quant_row5, quant_row6, quant_row7,
								   &cols_all_i16m8);
  }

  /* Second pass: compute IDCT on rows in workspace. */
  jsimd_idct_islow_pass2_regular(&cols_all_i16m8, output_buf, output_col);
}


/*
* This "regular" version assumes that no optimization can be made to the IDCT
* calculation, since no useful set of AC coefficients is all 0.
*
* The original C implementation of the accurate IDCT (jpeg_idct_slow()) can be
* found in jidctint.c.  Algorithmic changes made here are documented inline.
*/

static INLINE void jsimd_idct_islow_pass1_regular(vint16m1_t row0,
                                                  vint16m1_t row1,
                                                  vint16m1_t row2,
                                                  vint16m1_t row3,
                                                  vint16m1_t row4,
                                                  vint16m1_t row5,
                                                  vint16m1_t row6,
                                                  vint16m1_t row7,
                                                  vint16m1_t quant_row0,
                                                  vint16m1_t quant_row1,
                                                  vint16m1_t quant_row2,
                                                  vint16m1_t quant_row3,
                                                  vint16m1_t quant_row4,
                                                  vint16m1_t quant_row5,
                                                  vint16m1_t quant_row6,
                                                  vint16m1_t quant_row7,
												  vint16m8_t *cols_all_i16m8)
{
  /* Even part */
  size_t vl = 8;
  vint16m1_t z2_s16 = __riscv_vmul_vv_i16m1(row2, quant_row2, vl);
  vint16m1_t z3_s16 = __riscv_vmul_vv_i16m1(row6, quant_row6, vl);
  vint32m2_t tmp2 = __riscv_vwmul_vx_i32m2(z2_s16, idct_islow_consts[1], vl);
  vint32m2_t tmp3 = __riscv_vwmul_vx_i32m2(z2_s16, idct_islow_consts[6], vl);
  tmp2 = __riscv_vwmacc_vx_i32m2(tmp2, idct_islow_consts[9], z3_s16, vl);
  tmp3 = __riscv_vwmacc_vx_i32m2(tmp3, idct_islow_consts[1], z3_s16, vl);
 
  z2_s16 = __riscv_vmul_vv_i16m1(row0, quant_row0, vl);
  z3_s16 = __riscv_vmul_vv_i16m1(row4, quant_row4, vl);
  vint32m2_t tmp0 = __riscv_vwmul_vx_i32m2(__riscv_vadd_vv_i16m1(z2_s16, z3_s16, vl), 8192, vl);	// 8192 = 2^(CONST_BITS)
  vint32m2_t tmp1 = __riscv_vwmul_vx_i32m2(__riscv_vsub_vv_i16m1(z2_s16, z3_s16, vl), 8192, vl);
 
  vint32m2_t tmp10 = __riscv_vadd_vv_i32m2(tmp0, tmp3, vl);
  vint32m2_t tmp13 = __riscv_vsub_vv_i32m2(tmp0, tmp3, vl);
  vint32m2_t tmp11 = __riscv_vadd_vv_i32m2(tmp1, tmp2, vl);
  vint32m2_t tmp12 = __riscv_vsub_vv_i32m2(tmp1, tmp2, vl);
 
  /* Odd part */
  vint16m1_t tmp0_s16 = __riscv_vmul_vv_i16m1(row7, quant_row7, vl);
  vint16m1_t tmp1_s16 = __riscv_vmul_vv_i16m1(row5, quant_row5, vl);
  vint16m1_t tmp2_s16 = __riscv_vmul_vv_i16m1(row3, quant_row3, vl);
  vint16m1_t tmp3_s16 = __riscv_vmul_vv_i16m1(row1, quant_row1, vl);
  z3_s16 = __riscv_vadd_vv_i16m1(tmp0_s16, tmp2_s16, vl);
  vint16m1_t z4_s16 = __riscv_vadd_vv_i16m1(tmp1_s16, tmp3_s16, vl);
 
  vint32m2_t z3 = __riscv_vwmul_vx_i32m2(z3_s16, idct_islow_consts[11], vl);
  vint32m2_t z4 = __riscv_vwmul_vx_i32m2(z3_s16, idct_islow_consts[ 7], vl);
  z3 = __riscv_vwmacc_vx_i32m2(z3, idct_islow_consts[7], z4_s16, vl);
  z4 = __riscv_vwmacc_vx_i32m2(z4, idct_islow_consts[8], z4_s16, vl);
 
  tmp0 = __riscv_vwmul_vx_i32m2(tmp0_s16, idct_islow_consts[ 3], vl);
  tmp1 = __riscv_vwmul_vx_i32m2(tmp1_s16, idct_islow_consts[ 5], vl);
  tmp2 = __riscv_vwmul_vx_i32m2(tmp2_s16, idct_islow_consts[10], vl);
  tmp3 = __riscv_vwmul_vx_i32m2(tmp3_s16, idct_islow_consts[ 4], vl);
 
  tmp0 = __riscv_vwmacc_vx_i32m2(tmp0, -idct_islow_consts[0], tmp3_s16, vl);
  tmp1 = __riscv_vwmacc_vx_i32m2(tmp1, -idct_islow_consts[2], tmp2_s16, vl);
  tmp2 = __riscv_vwmacc_vx_i32m2(tmp2, -idct_islow_consts[2], tmp1_s16, vl);
  tmp3 = __riscv_vwmacc_vx_i32m2(tmp3, -idct_islow_consts[0], tmp0_s16, vl);
 
  tmp0 = __riscv_vadd_vv_i32m2(tmp0, z3, vl);
  tmp1 = __riscv_vadd_vv_i32m2(tmp1, z4, vl);
  tmp2 = __riscv_vadd_vv_i32m2(tmp2, z3, vl);
  tmp3 = __riscv_vadd_vv_i32m2(tmp3, z4, vl);

  /* Final output stage: descale and narrow to 16-bit. */
  vint16m4_t cols_0123_i16m4, cols_7654_i16m4;
  vint8m2_t tmp_i8m2;
  vint16m4_t tmp_i16m4;
  vint32m8_t tmp_i32m8;

  // combine (tmp10, tmp11, tmp12, tmp13)
  vint32m4_t tmps_10_11 = __riscv_vlmul_ext_v_i32m2_i32m4(tmp10);
  tmps_10_11 = __riscv_vslideup_vx_i32m4(tmps_10_11, __riscv_vlmul_ext_v_i32m2_i32m4(tmp11), 8, 16);
  vint32m4_t tmps_12_13 = __riscv_vlmul_ext_v_i32m2_i32m4(tmp12);
  tmps_12_13 = __riscv_vslideup_vx_i32m4(tmps_12_13, __riscv_vlmul_ext_v_i32m2_i32m4(tmp13), 8, 16);

  vint32m8_t tmps_10_11_12_13 = __riscv_vlmul_ext_v_i32m4_i32m8(tmps_10_11);
  tmps_10_11_12_13 = __riscv_vslideup_vx_i32m8(tmps_10_11_12_13, __riscv_vlmul_ext_v_i32m4_i32m8(tmps_12_13),  16, 32);

  // combine (tmp3, tmp2, tmp1, tmp0)
  vint32m4_t tmps_3_2 = __riscv_vlmul_ext_v_i32m2_i32m4(tmp3);
  tmps_3_2 = __riscv_vslideup_vx_i32m4(tmps_3_2, __riscv_vlmul_ext_v_i32m2_i32m4(tmp2), 8, 16);
  vint32m4_t tmps_1_0 = __riscv_vlmul_ext_v_i32m2_i32m4(tmp1);
  tmps_1_0 = __riscv_vslideup_vx_i32m4(tmps_1_0, __riscv_vlmul_ext_v_i32m2_i32m4(tmp0), 8, 16);

  vint32m8_t tmps_3_2_1_0 = __riscv_vlmul_ext_v_i32m4_i32m8(tmps_3_2);
  tmps_3_2_1_0 = __riscv_vslideup_vx_i32m8(tmps_3_2_1_0, __riscv_vlmul_ext_v_i32m4_i32m8(tmps_1_0),  16, 32);

  // col 0, 1, 2, 3
  tmp_i32m8 = __riscv_vadd_vv_i32m8(tmps_10_11_12_13, tmps_3_2_1_0, 32);
  tmp_i32m8 = __riscv_vadd_vx_i32m8(tmp_i32m8, 1<<(DESCALE_P1-1), 32);	// for Rounding
  cols_0123_i16m4 = __riscv_vnsra_wx_i16m4(tmp_i32m8, DESCALE_P1, 32);

  // col 7, 6, 5, 4
  tmp_i32m8 = __riscv_vsub_vv_i32m8(tmps_10_11_12_13, tmps_3_2_1_0, 32);
  tmp_i32m8 = __riscv_vadd_vx_i32m8(tmp_i32m8, 1<<(DESCALE_P1-1), 32);	// for Rounding
  cols_7654_i16m4 = __riscv_vnsra_wx_i16m4(tmp_i32m8, DESCALE_P1, 32);

  // combine vectors
  *cols_all_i16m8 = __riscv_vlmul_ext_v_i16m4_i16m8(cols_0123_i16m4);
  *cols_all_i16m8 = __riscv_vslideup_vx_i16m8(*cols_all_i16m8, __riscv_vlmul_ext_v_i16m4_i16m8(cols_7654_i16m4), 32, 64);
}


/* Perform the second pass of the accurate inverse DCT on a 4x8 block of
 * coefficients.  (To process the full 8x8 DCT block, this function-- or some
 * other optimized variant-- needs to be called for both the right and left 4x8
 * blocks.)
 *
 * This "regular" version assumes that no optimization can be made to the IDCT
 * calculation, since no useful set of coefficient values are all 0 after the
 * first pass.
 *
 * Again, the original C implementation of the accurate IDCT (jpeg_idct_slow())
 * can be found in jidctint.c.  Algorithmic changes made here are documented
 * inline.
 */

static INLINE void jsimd_idct_islow_pass2_regular(vint16m8_t *cols_all_i16m8,
                                                  JSAMPARRAY output_buf,
                                                  JDIMENSION output_col)
{
  int16_t workspace[DCTSIZE2];      /* buffers data */

  // load transpose look-up table
  vuint8m4_t vg_reg8 = __riscv_vle8_v_u8m4(trans_index8x8_u8, TRANS_TABLE_U8_SIZE);
 
  // transpose
  vint16m8_t trans_all_i16m8 = __riscv_vrgather_vv_i16m8(*cols_all_i16m8, \
	__riscv_vzext_vf2_u16m8(vg_reg8, TRANS_TABLE_U8_SIZE), TRANS_TABLE_U8_SIZE);

  __riscv_vse16_v_i16m8(&workspace[0], trans_all_i16m8, DCTSIZE2);

  /* Even part */
  size_t vl = 8;
  vint16m1_t z2_s16 = __riscv_vle16_v_i16m1(workspace + 2 * DCTSIZE, vl);
  vint16m1_t z3_s16 = __riscv_vle16_v_i16m1(workspace + 6 * DCTSIZE, vl);

  vint32m2_t tmp2 = __riscv_vwmul_vx_i32m2(z2_s16, idct_islow_consts[1], vl);
  vint32m2_t tmp3 = __riscv_vwmul_vx_i32m2(z2_s16, idct_islow_consts[6], vl);
  tmp2 = __riscv_vwmacc_vx_i32m2(tmp2, idct_islow_consts[9], z3_s16, vl);
  tmp3 = __riscv_vwmacc_vx_i32m2(tmp3, idct_islow_consts[1], z3_s16, vl);

  z2_s16 = __riscv_vle16_v_i16m1(workspace + 0 * DCTSIZE, vl);
  z3_s16 = __riscv_vle16_v_i16m1(workspace + 4 * DCTSIZE, vl);

  vint32m2_t tmp0 = __riscv_vwmul_vx_i32m2(__riscv_vadd_vv_i16m1(z2_s16, z3_s16, vl), 8192, vl);
  vint32m2_t tmp1 = __riscv_vwmul_vx_i32m2(__riscv_vsub_vv_i16m1(z2_s16, z3_s16, vl), 8192, vl);

  vint32m2_t tmp10 = __riscv_vadd_vv_i32m2(tmp0, tmp3, vl);
  vint32m2_t tmp13 = __riscv_vsub_vv_i32m2(tmp0, tmp3, vl);
  vint32m2_t tmp11 = __riscv_vadd_vv_i32m2(tmp1, tmp2, vl);
  vint32m2_t tmp12 = __riscv_vsub_vv_i32m2(tmp1, tmp2, vl);

  /* Odd part */
  vint16m1_t tmp0_s16 = __riscv_vle16_v_i16m1(workspace + 7 * DCTSIZE, vl);
  vint16m1_t tmp1_s16 = __riscv_vle16_v_i16m1(workspace + 5 * DCTSIZE, vl);
  vint16m1_t tmp2_s16 = __riscv_vle16_v_i16m1(workspace + 3 * DCTSIZE, vl);
  vint16m1_t tmp3_s16 = __riscv_vle16_v_i16m1(workspace + 1 * DCTSIZE, vl);

  z3_s16 = __riscv_vadd_vv_i16m1(tmp0_s16, tmp2_s16, vl);
  vint16m1_t z4_s16 = __riscv_vadd_vv_i16m1(tmp1_s16, tmp3_s16, vl);

  vint32m2_t z3 = __riscv_vwmul_vx_i32m2(z3_s16, idct_islow_consts[11], vl);
  vint32m2_t z4 = __riscv_vwmul_vx_i32m2(z3_s16, idct_islow_consts[ 7], vl);
  z3 = __riscv_vwmacc_vx_i32m2(z3, idct_islow_consts[7], z4_s16, vl);
  z4 = __riscv_vwmacc_vx_i32m2(z4, idct_islow_consts[8], z4_s16, vl);

  tmp0 = __riscv_vwmul_vx_i32m2(tmp0_s16, idct_islow_consts[ 3], vl);
  tmp1 = __riscv_vwmul_vx_i32m2(tmp1_s16, idct_islow_consts[ 5], vl);
  tmp2 = __riscv_vwmul_vx_i32m2(tmp2_s16, idct_islow_consts[10], vl);
  tmp3 = __riscv_vwmul_vx_i32m2(tmp3_s16, idct_islow_consts[ 4], vl);

  tmp0 = __riscv_vwmacc_vx_i32m2(tmp0, -idct_islow_consts[0], tmp3_s16, vl);
  tmp1 = __riscv_vwmacc_vx_i32m2(tmp1, -idct_islow_consts[2], tmp2_s16, vl);
  tmp2 = __riscv_vwmacc_vx_i32m2(tmp2, -idct_islow_consts[2], tmp1_s16, vl);
  tmp3 = __riscv_vwmacc_vx_i32m2(tmp3, -idct_islow_consts[0], tmp0_s16, vl);

  tmp0 = __riscv_vadd_vv_i32m2(tmp0, z3, vl);
  tmp1 = __riscv_vadd_vv_i32m2(tmp1, z4, vl);
  tmp2 = __riscv_vadd_vv_i32m2(tmp2, z3, vl);
  tmp3 = __riscv_vadd_vv_i32m2(tmp3, z4, vl);

/* Final output stage: descale and narrow to 8-bit. */
/* Clamp to range [0-255]. */
  vuint8m2_t cols_0123_u8m2, cols_7654_u8m2;
  vint8m2_t tmp_i8m2;
  vint16m4_t tmp_i16m4;
  vint32m8_t tmp_i32m8;

  // combine (tmp10, tmp11, tmp12, tmp13)
  vint32m4_t tmps_10_11 = __riscv_vlmul_ext_v_i32m2_i32m4(tmp10);
  tmps_10_11 = __riscv_vslideup_vx_i32m4(tmps_10_11, __riscv_vlmul_ext_v_i32m2_i32m4(tmp11), 8, 16);
  vint32m4_t tmps_12_13 = __riscv_vlmul_ext_v_i32m2_i32m4(tmp12);
  tmps_12_13 = __riscv_vslideup_vx_i32m4(tmps_12_13, __riscv_vlmul_ext_v_i32m2_i32m4(tmp13), 8, 16);

  vint32m8_t tmps_10_11_12_13 = __riscv_vlmul_ext_v_i32m4_i32m8(tmps_10_11);
  tmps_10_11_12_13 = __riscv_vslideup_vx_i32m8(tmps_10_11_12_13, __riscv_vlmul_ext_v_i32m4_i32m8(tmps_12_13),  16, 32);

  // combine (tmp3, tmp2, tmp1, tmp0)
  vint32m4_t tmps_3_2 = __riscv_vlmul_ext_v_i32m2_i32m4(tmp3);
  tmps_3_2 = __riscv_vslideup_vx_i32m4(tmps_3_2, __riscv_vlmul_ext_v_i32m2_i32m4(tmp2), 8, 16);
  vint32m4_t tmps_1_0 = __riscv_vlmul_ext_v_i32m2_i32m4(tmp1);
  tmps_1_0 = __riscv_vslideup_vx_i32m4(tmps_1_0, __riscv_vlmul_ext_v_i32m2_i32m4(tmp0), 8, 16);

  vint32m8_t tmps_3_2_1_0 = __riscv_vlmul_ext_v_i32m4_i32m8(tmps_3_2);
  tmps_3_2_1_0 = __riscv_vslideup_vx_i32m8(tmps_3_2_1_0, __riscv_vlmul_ext_v_i32m4_i32m8(tmps_1_0),  16, 32);

  // col 0, 1, 2, 3
  tmp_i32m8 = __riscv_vadd_vv_i32m8(tmps_10_11_12_13, tmps_3_2_1_0, 32);
  tmp_i16m4 = __riscv_vnsra_wx_i16m4(tmp_i32m8, 16, 32);
  tmp_i8m2 = __riscv_vnclip_wx_i8m2(tmp_i16m4, DESCALE_P2 - 16, 32);
  cols_0123_u8m2 = __riscv_vreinterpret_v_i8m2_u8m2(tmp_i8m2);
  cols_0123_u8m2 = __riscv_vadd_vx_u8m2(cols_0123_u8m2, CENTERJSAMPLE, 32);

  // col 7, 6, 5, 4
  tmp_i32m8 = __riscv_vsub_vv_i32m8(tmps_10_11_12_13, tmps_3_2_1_0, 32);
  tmp_i16m4 = __riscv_vnsra_wx_i16m4(tmp_i32m8, 16, 32);
  tmp_i8m2 = __riscv_vnclip_wx_i8m2(tmp_i16m4, DESCALE_P2 - 16, 32);
  cols_7654_u8m2 = __riscv_vreinterpret_v_i8m2_u8m2(tmp_i8m2);
  cols_7654_u8m2 = __riscv_vadd_vx_u8m2(cols_7654_u8m2, CENTERJSAMPLE, 32);

  // combine vectors
  vuint8m4_t cols_all_u8m4 = __riscv_vlmul_ext_v_u8m2_u8m4(cols_0123_u8m2);
  cols_all_u8m4 = __riscv_vslideup_vx_u8m4(cols_all_u8m4, __riscv_vlmul_ext_v_u8m2_u8m4(cols_7654_u8m2), 32, 64);

  // transpose
  vuint8m4_t trans_all_u8m4 = __riscv_vrgather(cols_all_u8m4, vg_reg8, TRANS_TABLE_U8_SIZE);

  // extract columns
  vuint8mf2_t col_0_u8mf2, col_1_u8mf2, col_2_u8mf2, col_3_u8mf2;
  vuint8mf2_t col_4_u8mf2, col_5_u8mf2, col_6_u8mf2, col_7_u8mf2;

  vuint8m4_t slidedown_m4 = __riscv_vslidedown_vx_u8m4(trans_all_u8m4, 0, vl);
  col_0_u8mf2 = __riscv_vlmul_trunc_v_u8m4_u8mf2(slidedown_m4);

  slidedown_m4 = __riscv_vslidedown_vx_u8m4(trans_all_u8m4, 8, vl);
  col_1_u8mf2 = __riscv_vlmul_trunc_v_u8m4_u8mf2(slidedown_m4);

  slidedown_m4 = __riscv_vslidedown_vx_u8m4(trans_all_u8m4, 16, vl);
  col_2_u8mf2 = __riscv_vlmul_trunc_v_u8m4_u8mf2(slidedown_m4);

  slidedown_m4 = __riscv_vslidedown_vx_u8m4(trans_all_u8m4, 24, vl);
  col_3_u8mf2 = __riscv_vlmul_trunc_v_u8m4_u8mf2(slidedown_m4);

  slidedown_m4 = __riscv_vslidedown_vx_u8m4(trans_all_u8m4, 32, vl);
  col_4_u8mf2 = __riscv_vlmul_trunc_v_u8m4_u8mf2(slidedown_m4);

  slidedown_m4 = __riscv_vslidedown_vx_u8m4(trans_all_u8m4, 40, vl);
  col_5_u8mf2 = __riscv_vlmul_trunc_v_u8m4_u8mf2(slidedown_m4);

  slidedown_m4 = __riscv_vslidedown_vx_u8m4(trans_all_u8m4, 48, vl);
  col_6_u8mf2 = __riscv_vlmul_trunc_v_u8m4_u8mf2(slidedown_m4);

  slidedown_m4 = __riscv_vslidedown_vx_u8m4(trans_all_u8m4, 56, vl);
  col_7_u8mf2 = __riscv_vlmul_trunc_v_u8m4_u8mf2(slidedown_m4);

  // store to memory
  JSAMPROW outptr0 = output_buf[0] + output_col;
  JSAMPROW outptr1 = output_buf[1] + output_col;
  JSAMPROW outptr2 = output_buf[2] + output_col;
  JSAMPROW outptr3 = output_buf[3] + output_col;
  JSAMPROW outptr4 = output_buf[4] + output_col;
  JSAMPROW outptr5 = output_buf[5] + output_col;
  JSAMPROW outptr6 = output_buf[6] + output_col;
  JSAMPROW outptr7 = output_buf[7] + output_col;

   __riscv_vse8_v_u8mf2(outptr0, col_0_u8mf2, vl);
   __riscv_vse8_v_u8mf2(outptr1, col_1_u8mf2, vl);
   __riscv_vse8_v_u8mf2(outptr2, col_2_u8mf2, vl);
   __riscv_vse8_v_u8mf2(outptr3, col_3_u8mf2, vl);
   __riscv_vse8_v_u8mf2(outptr4, col_4_u8mf2, vl);
   __riscv_vse8_v_u8mf2(outptr5, col_5_u8mf2, vl);
   __riscv_vse8_v_u8mf2(outptr6, col_6_u8mf2, vl);
   __riscv_vse8_v_u8mf2(outptr7, col_7_u8mf2, vl);
}

