/*
* jidctfst-rvv.c - fast integer IDCT (RISC-V RVV)
*
* Copyright (c) 2012-2024 Andes Technology Corporation
* All rights reserved.
*/
/*
 * jidctfst-neon.c - fast integer IDCT (Arm Neon)
 *
 * Copyright (C) 2020, Arm Limited.  All Rights Reserved.
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


/* jsimd_idct_ifast_rvv() performs dequantization and a fast, not so accurate
 * inverse DCT (Discrete Cosine Transform) on one block of coefficients.  It
 * uses the same calculations and produces exactly the same output as IJG's
 * original jpeg_idct_ifast() function, which can be found in jidctfst.c.
 *
 * Scaled integer constants are used to avoid floating-point arithmetic:
 *    0.082392200 =  2688 * 2^-15
 *    0.414213562 = 13568 * 2^-15
 *    0.847759065 = 27776 * 2^-15
 *    0.613125930 = 20096 * 2^-15
 *
 * See jidctfst.c for further details of the IDCT algorithm.  Where possible,
 * the variable names and comments here in jsimd_idct_ifast_rvv() match up
 * with those in jpeg_idct_ifast().
 */

#define PASS1_BITS  2

#define F_0_082  2688
#define F_0_414  13568
#define F_0_847  27776
#define F_0_613  20096


static const int16_t idct_ifast_consts[] = {
  F_0_082, F_0_414, F_0_847, F_0_613
};

void jsimd_idct_ifast_rvv(void *dct_table, JCOEFPTR coef_block,
                           JSAMPARRAY output_buf, JDIMENSION output_col)
{
  IFAST_MULT_TYPE *quantptr = dct_table;
  vint16m8_t rows_all_i16m8;

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

  /* Load quantization table values for DC coefficients. */
  vint16m1_t quant_row0 = __riscv_vle16_v_i16m1(quantptr + 0 * DCTSIZE, vl);

  /* Dequantize DC coefficients. */
  row0 = __riscv_vmul_vv_i16m1(row0, quant_row0, vl);

  /* Construct bitmap to test if all AC coefficients are 0. */
  vint16m1_t bitmap = __riscv_vor_vv_i16m1(row1, row2, vl);
  bitmap = __riscv_vor_vv_i16m1(bitmap, row3, vl);
  bitmap = __riscv_vor_vv_i16m1(bitmap, row4, vl);
  bitmap = __riscv_vor_vv_i16m1(bitmap, row5, vl);
  bitmap = __riscv_vor_vv_i16m1(bitmap, row6, vl);
  bitmap = __riscv_vor_vv_i16m1(bitmap, row7, vl);

  uint16_t ac_bitmap;
  {
  vuint16m1_t vec_zero = __riscv_vmv_s_x_u16m1(0, vl);
  vuint16m1_t tmp_u16m1 = __riscv_vreinterpret_v_i16m1_u16m1(bitmap);
  ac_bitmap = __riscv_vmv_x_s_u16m1_u16(__riscv_vredor_vs_u16m1_u16m1(tmp_u16m1, vec_zero, vl));
  }

  /* Load IDCT conversion constants. */
  if (0 == ac_bitmap)
  {
    /* All AC coefficients are zero.
     * Compute DC values and duplicate into vectors.
     */
    // combine vectors
    vint16m4_t tmp_i16m4 = __riscv_vlmul_ext_v_i16m1_i16m4(row0);
    tmp_i16m4 = __riscv_vslideup_vx_i16m4(tmp_i16m4, tmp_i16m4, DCTSIZE2/8, DCTSIZE2/4);
    tmp_i16m4 = __riscv_vslideup_vx_i16m4(tmp_i16m4, tmp_i16m4, DCTSIZE2/4, DCTSIZE2/2);
    rows_all_i16m8 = __riscv_vlmul_ext_v_i16m4_i16m8(tmp_i16m4);
    rows_all_i16m8 = __riscv_vslideup_vx_i16m8(rows_all_i16m8, rows_all_i16m8, DCTSIZE2/2, DCTSIZE2);
  } 
  else {
    /* full IDCT calculation. */

    /* Load quantization table. */
    vint16m1_t quant_row1 = __riscv_vle16_v_i16m1(quantptr + 1 * DCTSIZE, vl);
    vint16m1_t quant_row2 = __riscv_vle16_v_i16m1(quantptr + 2 * DCTSIZE, vl);
    vint16m1_t quant_row3 = __riscv_vle16_v_i16m1(quantptr + 3 * DCTSIZE, vl);
    vint16m1_t quant_row4 = __riscv_vle16_v_i16m1(quantptr + 4 * DCTSIZE, vl);
    vint16m1_t quant_row5 = __riscv_vle16_v_i16m1(quantptr + 5 * DCTSIZE, vl);
    vint16m1_t quant_row6 = __riscv_vle16_v_i16m1(quantptr + 6 * DCTSIZE, vl);
    vint16m1_t quant_row7 = __riscv_vle16_v_i16m1(quantptr + 7 * DCTSIZE, vl);

    /* Even part: dequantize DCT coefficients. */
    vint16m1_t tmp0 = __riscv_vmv_v_v_i16m1(row0, vl);
    vint16m1_t tmp1 = __riscv_vmul_vv_i16m1(row2, quant_row2, vl);
    vint16m1_t tmp2 = __riscv_vmul_vv_i16m1(row4, quant_row4, vl);
    vint16m1_t tmp3 = __riscv_vmul_vv_i16m1(row6, quant_row6, vl);

    vint16m1_t tmp10 = __riscv_vadd_vv_i16m1(tmp0, tmp2, vl);	/* phase 3 */
    vint16m1_t tmp11 = __riscv_vsub_vv_i16m1(tmp0, tmp2, vl);
    vint16m1_t tmp13 = __riscv_vadd_vv_i16m1(tmp1, tmp3, vl);	/* phases 5-3 */
    vint16m1_t tmp12;
    {
    vint16m1_t tmp1_sub_tmp3 = __riscv_vsub_vv_i16m1(tmp1, tmp3, vl);
    tmp12 = __riscv_vsmul_vx_i16m1(tmp1_sub_tmp3, idct_ifast_consts[1], vl);
    tmp12 = __riscv_vadd_vv_i16m1(tmp12, tmp1_sub_tmp3, vl);
    tmp12 = __riscv_vsub_vv_i16m1(tmp12, tmp13, vl);
    }

    tmp0 = __riscv_vadd_vv_i16m1(tmp10, tmp13, vl);			/* phase 2 */
    tmp3 = __riscv_vsub_vv_i16m1(tmp10, tmp13, vl);
    tmp1 = __riscv_vadd_vv_i16m1(tmp11, tmp12, vl);
    tmp2 = __riscv_vsub_vv_i16m1(tmp11, tmp12, vl);

    /* Odd part: dequantize DCT coefficients. */
    vint16m1_t tmp4 = __riscv_vmul_vv_i16m1(__riscv_vmv_v_v_i16m1(row1, vl), quant_row1, vl);
    vint16m1_t tmp5 = __riscv_vmul_vv_i16m1(__riscv_vmv_v_v_i16m1(row3, vl), quant_row3, vl);
    vint16m1_t tmp6 = __riscv_vmul_vv_i16m1(__riscv_vmv_v_v_i16m1(row5, vl), quant_row5, vl);
    vint16m1_t tmp7 = __riscv_vmul_vv_i16m1(__riscv_vmv_v_v_i16m1(row7, vl), quant_row7, vl);

    vint16m1_t z13 = __riscv_vadd_vv_i16m1(tmp5, tmp6, vl);	/* phase 6 */
    vint16m1_t neg_z10 = __riscv_vsub_vv_i16m1(tmp5, tmp6, vl);
    vint16m1_t z11 = __riscv_vadd_vv_i16m1(tmp4, tmp7, vl);
    vint16m1_t z12 = __riscv_vsub_vv_i16m1(tmp4, tmp7, vl);

    tmp7 = __riscv_vadd_vv_i16m1(z11, z13, vl);				/* phase 5 */
    {
    vint16m1_t z11_sub_z13 = __riscv_vsub_vv_i16m1(z11, z13, vl);
    tmp11 = __riscv_vsmul_vx_i16m1(z11_sub_z13, idct_ifast_consts[1], vl);
    tmp11 = __riscv_vadd_vv_i16m1(tmp11, z11_sub_z13, vl);
    }

    {
    vint16m1_t z10_add_z12 = __riscv_vsub_vv_i16m1(z12, neg_z10, vl);
    vint16m1_t z5 = __riscv_vsmul_vx_i16m1(z10_add_z12, idct_ifast_consts[2], vl);
    z5 = __riscv_vadd_vv_i16m1(z5, z10_add_z12, vl);

    tmp10 = __riscv_vsmul_vx_i16m1(z12, idct_ifast_consts[0], vl);
    tmp10 = __riscv_vadd_vv_i16m1(tmp10, z12, vl);
    tmp10 = __riscv_vsub_vv_i16m1(tmp10, z5, vl);

    tmp12 = __riscv_vsmul_vx_i16m1(neg_z10, idct_ifast_consts[3], vl);
    vint16m1_t tmp_neg_z10_d = __riscv_vadd_vv_i16m1(neg_z10, neg_z10, vl);
    tmp12 = __riscv_vadd_vv_i16m1(tmp12, tmp_neg_z10_d, vl);
    tmp12 = __riscv_vadd_vv_i16m1(tmp12, z5, vl);
    }

    tmp6 = __riscv_vsub_vv_i16m1(tmp12, tmp7, vl);			/* phase 2 */
    tmp5 = __riscv_vsub_vv_i16m1(tmp11, tmp6, vl);
    tmp4 = __riscv_vadd_vv_i16m1(tmp10, tmp5, vl);

    row0 = __riscv_vadd_vv_i16m1(tmp0, tmp7, vl);
    row7 = __riscv_vsub_vv_i16m1(tmp0, tmp7, vl);
    row1 = __riscv_vadd_vv_i16m1(tmp1, tmp6, vl);
    row6 = __riscv_vsub_vv_i16m1(tmp1, tmp6, vl);
    row2 = __riscv_vadd_vv_i16m1(tmp2, tmp5, vl);
    row5 = __riscv_vsub_vv_i16m1(tmp2, tmp5, vl);
    row4 = __riscv_vadd_vv_i16m1(tmp3, tmp4, vl);
    row3 = __riscv_vsub_vv_i16m1(tmp3, tmp4, vl);

    // combine vectors
    vint16m4_t rows_0123_i16m4 = __riscv_vlmul_ext_v_i16m1_i16m4(row0);
    rows_0123_i16m4 = __riscv_vslideup_vx_i16m4(rows_0123_i16m4, __riscv_vlmul_ext_v_i16m1_i16m4(row1),  8, 32);
    rows_0123_i16m4 = __riscv_vslideup_vx_i16m4(rows_0123_i16m4, __riscv_vlmul_ext_v_i16m1_i16m4(row2), 16, 32);
    rows_0123_i16m4 = __riscv_vslideup_vx_i16m4(rows_0123_i16m4, __riscv_vlmul_ext_v_i16m1_i16m4(row2), 24, 32);

    vint16m4_t rows_4567_i16m4 = __riscv_vlmul_ext_v_i16m1_i16m4(row4);
    rows_4567_i16m4 = __riscv_vslideup_vx_i16m4(rows_4567_i16m4, __riscv_vlmul_ext_v_i16m1_i16m4(row5),  8, 32);
    rows_4567_i16m4 = __riscv_vslideup_vx_i16m4(rows_4567_i16m4, __riscv_vlmul_ext_v_i16m1_i16m4(row6), 16, 32);
    rows_4567_i16m4 = __riscv_vslideup_vx_i16m4(rows_4567_i16m4, __riscv_vlmul_ext_v_i16m1_i16m4(row7), 24, 32);

    rows_all_i16m8 = __riscv_vlmul_ext_v_i16m4_i16m8(rows_0123_i16m4);
    rows_all_i16m8 = __riscv_vslideup_vx_i16m8(rows_all_i16m8, __riscv_vlmul_ext_v_i16m4_i16m8(rows_4567_i16m4), 32, 64);
  }

  /* Transpose rows to work on columns in pass 2. */
  const uint8_t trans_index8x8_u8[DCTSIZE2] =
  {
    0,  8, 16, 24, 32, 40, 48, 56,
    1,  9, 17, 25, 33, 41, 49, 57,
    2, 10, 18, 26, 34, 42, 50, 58,
    3, 11, 19, 27, 35, 43, 51, 59,
    4, 12, 20, 28, 36, 44, 52, 60,
    5, 13, 21, 29, 37, 45, 53, 61,
    6, 14, 22, 30, 38, 46, 54, 62,
    7, 15, 23, 31, 39, 47, 55, 63,
  };

  // load transpose look-up table
  vuint8m4_t vg_reg8 = __riscv_vle8_v_u8m4(trans_index8x8_u8, DCTSIZE2);

  // interpret to u16 & transpose
  vint16m8_t vg_reg16 = __riscv_vrgather(rows_all_i16m8, __riscv_vzext_vf2_u16m8(vg_reg8, DCTSIZE2), \
	DCTSIZE2);
  int16_t workspace[DCTSIZE2];      /* buffers data between passes */
  __riscv_vse16_v_i16m8(&workspace[0], vg_reg16, DCTSIZE2);
  vint16m1_t col0 = __riscv_vle16_v_i16m1(&workspace[8*0], vl);
  vint16m1_t col1 = __riscv_vle16_v_i16m1(&workspace[8*1], vl);
  vint16m1_t col2 = __riscv_vle16_v_i16m1(&workspace[8*2], vl);
  vint16m1_t col3 = __riscv_vle16_v_i16m1(&workspace[8*3], vl);
  vint16m1_t col4 = __riscv_vle16_v_i16m1(&workspace[8*4], vl);
  vint16m1_t col5 = __riscv_vle16_v_i16m1(&workspace[8*5], vl);
  vint16m1_t col6 = __riscv_vle16_v_i16m1(&workspace[8*6], vl);
  vint16m1_t col7 = __riscv_vle16_v_i16m1(&workspace[8*7], vl);

  /* 1-D IDCT, pass 2 */

  /* Even part */
  vint16m1_t tmp10 = __riscv_vadd_vv_i16m1(col0, col4, vl);
  vint16m1_t tmp11 = __riscv_vsub_vv_i16m1(col0, col4, vl);
  vint16m1_t tmp13 = __riscv_vadd_vv_i16m1(col2, col6, vl);
  vint16m1_t tmp12;
  {
  vint16m1_t col2_sub_col6 = __riscv_vsub_vv_i16m1(col2, col6, vl);
  tmp12 = __riscv_vsmul_vx_i16m1(col2_sub_col6, idct_ifast_consts[1], vl);
  tmp12 = __riscv_vadd_vv_i16m1(tmp12, col2_sub_col6, vl);
  tmp12 = __riscv_vsub_vv_i16m1(tmp12, tmp13, vl);
  }

  vint16m1_t tmp0 = __riscv_vadd_vv_i16m1(tmp10, tmp13, vl);
  vint16m1_t tmp3 = __riscv_vsub_vv_i16m1(tmp10, tmp13, vl);
  vint16m1_t tmp1 = __riscv_vadd_vv_i16m1(tmp11, tmp12, vl);
  vint16m1_t tmp2 = __riscv_vsub_vv_i16m1(tmp11, tmp12, vl);

  /* Odd part */
  vint16m1_t z13 = __riscv_vadd_vv_i16m1(col5, col3, vl);
  vint16m1_t neg_z10 = __riscv_vsub_vv_i16m1(col3, col5, vl);
  vint16m1_t z11 = __riscv_vadd_vv_i16m1(col1, col7, vl);
  vint16m1_t z12 = __riscv_vsub_vv_i16m1(col1, col7, vl);
 
  vint16m1_t tmp7 = __riscv_vadd_vv_i16m1(z11, z13, vl);	/* phase 5 */
  {
  vint16m1_t z11_sub_z13 = __riscv_vsub_vv_i16m1(z11, z13, vl);
  tmp11 = __riscv_vsmul_vx_i16m1(z11_sub_z13, idct_ifast_consts[1], vl);
  tmp11 = __riscv_vadd_vv_i16m1(tmp11, z11_sub_z13, vl);
  }
 
  {
  vint16m1_t z10_add_z12 = __riscv_vsub_vv_i16m1(z12, neg_z10, vl);
  vint16m1_t z5 = __riscv_vsmul_vx_i16m1(z10_add_z12, idct_ifast_consts[2], vl);
  z5 = __riscv_vadd_vv_i16m1(z5, z10_add_z12, vl);
 
  tmp10 = __riscv_vsmul_vx_i16m1(z12, idct_ifast_consts[0], vl);
  tmp10 = __riscv_vadd_vv_i16m1(tmp10, z12, vl);
  tmp10 = __riscv_vsub_vv_i16m1(tmp10, z5, vl);
 
  tmp12 = __riscv_vsmul_vx_i16m1(neg_z10, idct_ifast_consts[3], vl);
  vint16m1_t tmp_neg_z10_d = __riscv_vadd_vv_i16m1(neg_z10, neg_z10, vl);
  tmp12 = __riscv_vadd_vv_i16m1(tmp12, tmp_neg_z10_d, vl);
  tmp12 = __riscv_vadd_vv_i16m1(tmp12, z5, vl);
  }
 
  vint16m1_t tmp6 = __riscv_vsub_vv_i16m1(tmp12, tmp7, vl);	/* phase 2 */
  vint16m1_t tmp5 = __riscv_vsub_vv_i16m1(tmp11, tmp6, vl);
  vint16m1_t tmp4 = __riscv_vadd_vv_i16m1(tmp10, tmp5, vl);
 
  col0 = __riscv_vadd_vv_i16m1(tmp0, tmp7, vl);
  col7 = __riscv_vsub_vv_i16m1(tmp0, tmp7, vl);
  col1 = __riscv_vadd_vv_i16m1(tmp1, tmp6, vl);
  col6 = __riscv_vsub_vv_i16m1(tmp1, tmp6, vl);
  col2 = __riscv_vadd_vv_i16m1(tmp2, tmp5, vl);
  col5 = __riscv_vsub_vv_i16m1(tmp2, tmp5, vl);
  col4 = __riscv_vadd_vv_i16m1(tmp3, tmp4, vl);
  col3 = __riscv_vsub_vv_i16m1(tmp3, tmp4, vl);

  /* Scale down by a factor of 8, narrowing to 8-bit. */
  /* Clamp to range [0-255]. */
  vint8m2_t tmp_i8m2;
  vuint8m2_t u8m2_col0123, u8m2_col4567;
  vint16m4_t i16m4_col0123 = __riscv_vlmul_ext_v_i16m1_i16m4(col0);
  i16m4_col0123 = __riscv_vslideup_vx_i16m4(i16m4_col0123, __riscv_vlmul_ext_v_i16m1_i16m4(col1),  8, 16);
  i16m4_col0123 = __riscv_vslideup_vx_i16m4(i16m4_col0123, __riscv_vlmul_ext_v_i16m1_i16m4(col2), 16, 24);
  i16m4_col0123 = __riscv_vslideup_vx_i16m4(i16m4_col0123, __riscv_vlmul_ext_v_i16m1_i16m4(col3), 24, 32);
  tmp_i8m2 = __riscv_vnclip_wx_i8m2(i16m4_col0123, PASS1_BITS + 3, 32);	// clamp -128 ~ 127
  u8m2_col0123 = __riscv_vreinterpret_v_i8m2_u8m2(tmp_i8m2);
  u8m2_col0123 = __riscv_vadd_vx_u8m2(u8m2_col0123, CENTERJSAMPLE, 32);	// clamp 0 ~ 255

  vint16m4_t i16m4_col4567 = __riscv_vlmul_ext_v_i16m1_i16m4(col4);
  i16m4_col4567 = __riscv_vslideup_vx_i16m4(i16m4_col4567, __riscv_vlmul_ext_v_i16m1_i16m4(col5),  8, 16);
  i16m4_col4567 = __riscv_vslideup_vx_i16m4(i16m4_col4567, __riscv_vlmul_ext_v_i16m1_i16m4(col6), 16, 24);
  i16m4_col4567 = __riscv_vslideup_vx_i16m4(i16m4_col4567, __riscv_vlmul_ext_v_i16m1_i16m4(col7), 24, 32);
  tmp_i8m2 = __riscv_vnclip_wx_i8m2(i16m4_col4567, PASS1_BITS + 3, 32);	// clamp -128 ~ 127
  u8m2_col4567 = __riscv_vreinterpret_v_i8m2_u8m2(tmp_i8m2);
  u8m2_col4567 = __riscv_vadd_vx_u8m2(u8m2_col4567, CENTERJSAMPLE, 32);	// clamp 0 ~ 255

  vuint8m4_t u8m4_col_all = __riscv_vlmul_ext_v_u8m2_u8m4(u8m2_col0123);
  u8m4_col_all = __riscv_vslideup_vx_u8m4(u8m4_col_all, __riscv_vlmul_ext_v_u8m2_u8m4(u8m2_col4567), 32, 64);

  /* Transpose block to prepare for store. */
  vuint8m4_t u8m4_trans_all = __riscv_vrgather_vv_u8m4(u8m4_col_all, vg_reg8, DCTSIZE2);

  // extract columns
  vuint8mf2_t u8mf2_col_0, u8mf2_col_1, u8mf2_col_2, u8mf2_col_3;
  vuint8mf2_t u8mf2_col_4, u8mf2_col_5, u8mf2_col_6, u8mf2_col_7;

  vuint8m4_t slidedown_m4 = __riscv_vslidedown_vx_u8m4(u8m4_trans_all, 0, vl);
  u8mf2_col_0 = __riscv_vlmul_trunc_v_u8m4_u8mf2(slidedown_m4);

  slidedown_m4 = __riscv_vslidedown_vx_u8m4(u8m4_trans_all, 8, vl);
  u8mf2_col_1 = __riscv_vlmul_trunc_v_u8m4_u8mf2(slidedown_m4);

  slidedown_m4 = __riscv_vslidedown_vx_u8m4(u8m4_trans_all, 16, vl);
  u8mf2_col_2 = __riscv_vlmul_trunc_v_u8m4_u8mf2(slidedown_m4);

  slidedown_m4 = __riscv_vslidedown_vx_u8m4(u8m4_trans_all, 24, vl);
  u8mf2_col_3 = __riscv_vlmul_trunc_v_u8m4_u8mf2(slidedown_m4);

  slidedown_m4 = __riscv_vslidedown_vx_u8m4(u8m4_trans_all, 32, vl);
  u8mf2_col_4 = __riscv_vlmul_trunc_v_u8m4_u8mf2(slidedown_m4);

  slidedown_m4 = __riscv_vslidedown_vx_u8m4(u8m4_trans_all, 40, vl);
  u8mf2_col_5 = __riscv_vlmul_trunc_v_u8m4_u8mf2(slidedown_m4);

  slidedown_m4 = __riscv_vslidedown_vx_u8m4(u8m4_trans_all, 48, vl);
  u8mf2_col_6 = __riscv_vlmul_trunc_v_u8m4_u8mf2(slidedown_m4);

  slidedown_m4 = __riscv_vslidedown_vx_u8m4(u8m4_trans_all, 56, vl);
  u8mf2_col_7 = __riscv_vlmul_trunc_v_u8m4_u8mf2(slidedown_m4);

  JSAMPROW outptr0 = output_buf[0] + output_col;
  JSAMPROW outptr1 = output_buf[1] + output_col;
  JSAMPROW outptr2 = output_buf[2] + output_col;
  JSAMPROW outptr3 = output_buf[3] + output_col;
  JSAMPROW outptr4 = output_buf[4] + output_col;
  JSAMPROW outptr5 = output_buf[5] + output_col;
  JSAMPROW outptr6 = output_buf[6] + output_col;
  JSAMPROW outptr7 = output_buf[7] + output_col;

  /* Store DCT block to memory. */
  __riscv_vse8_v_u8mf2(outptr0, u8mf2_col_0, vl);
  __riscv_vse8_v_u8mf2(outptr1, u8mf2_col_1, vl);
  __riscv_vse8_v_u8mf2(outptr2, u8mf2_col_2, vl);
  __riscv_vse8_v_u8mf2(outptr3, u8mf2_col_3, vl);
  __riscv_vse8_v_u8mf2(outptr4, u8mf2_col_4, vl);
  __riscv_vse8_v_u8mf2(outptr5, u8mf2_col_5, vl);
  __riscv_vse8_v_u8mf2(outptr6, u8mf2_col_6, vl);
  __riscv_vse8_v_u8mf2(outptr7, u8mf2_col_7, vl);
}

