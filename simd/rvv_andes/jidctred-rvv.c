/*
* jidctred-rvv.c - reduced-size IDCT (RISC-V RVV)
*
* Copyright (c) 2012-2024 Andes Technology Corporation
* All rights reserved.
*/
/*
 * jidctred-neon.c - reduced-size IDCT (Arm Neon)
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

#define F_0_211  1730
#define F_0_509  4176
#define F_0_601  4926
#define F_0_720  5906
#define F_0_765  6270
#define F_0_850  6967
#define F_0_899  7373
#define F_1_061  8697
#define F_1_272  10426
#define F_1_451  11893
#define F_1_847  15137
#define F_2_172  17799
#define F_2_562  20995
#define F_3_624  29692


/* jsimd_idct_2x2_rvv() is an inverse DCT function that produces reduced-size
 * 2x2 output from an 8x8 DCT block.  It uses the same calculations and
 * produces exactly the same output as IJG's original jpeg_idct_2x2() function
 * from jpeg-6b, which can be found in jidctred.c.
 *
 * Scaled integer constants are used to avoid floating-point arithmetic:
 *    0.720959822 =  5906 * 2^-13
 *    0.850430095 =  6967 * 2^-13
 *    1.272758580 = 10426 * 2^-13
 *    3.624509785 = 29692 * 2^-13
 *
 * See jidctred.c for further details of the 2x2 IDCT algorithm.  Where
 * possible, the variable names and comments here in jsimd_idct_2x2_rvv()
 * match up with those in jpeg_idct_2x2().
 */

static const int16_t jsimd_idct_2x2_consts[] = {
  -F_0_720, F_0_850, -F_1_272, F_3_624
};

void jsimd_idct_2x2_rvv(void *dct_table, JCOEFPTR coef_block,
                           JSAMPARRAY output_buf, JDIMENSION output_col)
{
  ISLOW_MULT_TYPE *quantptr = dct_table;

  /* Load DCT coefficients. */
  size_t vl = 8;
  vint16m1_t row0 = __riscv_vle16_v_i16m1(coef_block + 0 * DCTSIZE, vl);
  vint16m1_t row1 = __riscv_vle16_v_i16m1(coef_block + 1 * DCTSIZE, vl);
  vint16m1_t row3 = __riscv_vle16_v_i16m1(coef_block + 3 * DCTSIZE, vl);
  vint16m1_t row5 = __riscv_vle16_v_i16m1(coef_block + 5 * DCTSIZE, vl);
  vint16m1_t row7 = __riscv_vle16_v_i16m1(coef_block + 7 * DCTSIZE, vl);

  /* Load quantization table values. */
  vint16m1_t quant_row0 = __riscv_vle16_v_i16m1(quantptr + 0 * DCTSIZE, vl);
  vint16m1_t quant_row1 = __riscv_vle16_v_i16m1(quantptr + 1 * DCTSIZE, vl);
  vint16m1_t quant_row3 = __riscv_vle16_v_i16m1(quantptr + 3 * DCTSIZE, vl);
  vint16m1_t quant_row5 = __riscv_vle16_v_i16m1(quantptr + 5 * DCTSIZE, vl);
  vint16m1_t quant_row7 = __riscv_vle16_v_i16m1(quantptr + 7 * DCTSIZE, vl);

  /* Dequantize DCT coefficients. */
  row0 = __riscv_vmul_vv_i16m1(row0, quant_row0, vl);
  row1 = __riscv_vmul_vv_i16m1(row1, quant_row1, vl);
  row3 = __riscv_vmul_vv_i16m1(row3, quant_row3, vl);
  row5 = __riscv_vmul_vv_i16m1(row5, quant_row5, vl);
  row7 = __riscv_vmul_vv_i16m1(row7, quant_row7, vl);

  /* Pass 1: process columns from input, put results in vectors row0 and
   * row1.
   */

  /* Even part */
  vint32m2_t tmp10 = __riscv_vwmul_vx_i32m2(row0, 8192, vl);	// 2 ^ (CONST_BITS)
  tmp10 = __riscv_vsll_vx_i32m2 (tmp10, 2, vl);

  /* Odd part */
  vint32m2_t tmp0 = __riscv_vwmul_vx_i32m2(row1, jsimd_idct_2x2_consts[3], vl);
  tmp0 = __riscv_vwmacc_vx_i32m2(tmp0, jsimd_idct_2x2_consts[2], row3, vl);
  tmp0 = __riscv_vwmacc_vx_i32m2(tmp0, jsimd_idct_2x2_consts[1], row5, vl);
  tmp0 = __riscv_vwmacc_vx_i32m2(tmp0, jsimd_idct_2x2_consts[0], row7, vl);

  /* Final output stage: descale and narrow to 16-bit. */
  vint32m2_t tmp_tt;
  tmp_tt = __riscv_vadd_vv_i32m2(tmp10, tmp0, vl);
  tmp_tt = __riscv_vadd_vx_i32m2(tmp_tt, 1<<(CONST_BITS-1), vl);
  row0 = __riscv_vnsra_wx_i16m1(tmp_tt, CONST_BITS, vl);

  tmp_tt = __riscv_vsub_vv_i32m2(tmp10, tmp0, vl);
  tmp_tt = __riscv_vadd_vx_i32m2(tmp_tt, 1<<(CONST_BITS-1), vl);
  row1 = __riscv_vnsra_wx_i16m1(tmp_tt, CONST_BITS, vl);

  /* Transpose two rows, ready for second pass. */
  const uint8_t trans_index_tab_u8[16] =
  {
	0,  8,
	1,  9,
	3, 11,
	5, 13,
	7, 15,
	0,  2, /* [10~13] output transpose index order */
	1,  3,
	0,  0,
  };

  // extend m1 to m2
  vint16m2_t m2_row0 = __riscv_vlmul_ext_v_i16m1_i16m2(row0);
  
  // combine
  m2_row0 = __riscv_vslideup_vx_i16m2(m2_row0, __riscv_vlmul_ext_v_i16m1_i16m2(row1),  8, 16);

  // load transpose look-up table
  vuint8m1_t vg_reg8 = __riscv_vle8_v_u8m1(trans_index_tab_u8, 16);

  // saved for output index
  vuint8m1_t index_order_u8m1 = __riscv_vslidedown_vx_u8m1(vg_reg8, 10, 4);

  // interpret to u16 & transpose
  vint16m2_t vg_reg16 = __riscv_vrgather(m2_row0, __riscv_vzext_vf2_u16m2(vg_reg8, 10), 10);

  // extract columns
  vl = 2;
  vint16m1_t col0 = __riscv_vget_v_i16m2_i16m1(vg_reg16, 0);
  vint16m1_t col1 = __riscv_vslidedown_vx_i16m1(col0, 2, vl);
  vint16m1_t col3 = __riscv_vslidedown_vx_i16m1(col0, 4, vl);
  vint16m1_t col5 = __riscv_vslidedown_vx_i16m1(col0, 6, vl);

  vint16m2_t slidedown_m2 = __riscv_vslidedown_vx_i16m2(vg_reg16, 8, vl);
  vint16m1_t col7 = __riscv_vlmul_trunc_v_i16m2_i16m1(slidedown_m2);

  /* Pass 2: process two rows, store to output array. */

  /* Even part: we're only interested in col0; the top half of tmp10 is "don't
   * care."
   */
  tmp10 = __riscv_vwmul_vx_i32m2(col0, 8192, vl);
  tmp10 = __riscv_vsll_vx_i32m2 (tmp10, 2, vl);

  /* Odd part: we're only interested in the bottom half of tmp0. */
  tmp0 = __riscv_vwmul_vx_i32m2(col1, jsimd_idct_2x2_consts[3], vl);
  tmp0 = __riscv_vwmacc_vx_i32m2(tmp0, jsimd_idct_2x2_consts[2], col3, vl);
  tmp0 = __riscv_vwmacc_vx_i32m2(tmp0, jsimd_idct_2x2_consts[1], col5, vl);
  tmp0 = __riscv_vwmacc_vx_i32m2(tmp0, jsimd_idct_2x2_consts[0], col7, vl);

  /* Final output stage: descale and clamp to range [0-255]. */
  /* Narrow to 8-bit and convert to unsigned. */
  vint32m2_t tmp_i32m2_0 = __riscv_vadd_vv_i32m2(tmp10, tmp0, 2);
  vint32m2_t tmp_i32m2_1 = __riscv_vsub_vv_i32m2(tmp10, tmp0, 2);
  tmp_i32m2_0 = __riscv_vslideup_vx_i32m2 (tmp_i32m2_0, tmp_i32m2_1, 2, 4);
  vint16m1_t tmp_i16m1 = __riscv_vnsra_wx_i16m1(tmp_i32m2_0, 16, 4);
  vint8mf2_t tmp_i8mf2 = __riscv_vnclip_wx_i8mf2(tmp_i16m1, CONST_BITS + PASS1_BITS + 3 + 2 - 16, 4);
  vuint8mf2_t u8mf2_col0 = __riscv_vreinterpret_v_i8mf2_u8mf2(tmp_i8mf2);
  u8mf2_col0 = __riscv_vadd_vx_u8mf2(u8mf2_col0, CENTERJSAMPLE, 4);

  /* Transpose */
  vuint8mf2_t v0_out = __riscv_vrgather_vv_u8mf2(u8mf2_col0, \
		  __riscv_vlmul_trunc_v_u8m1_u8mf2(index_order_u8m1), 4);
  vuint8mf2_t v1_out = __riscv_vslidedown_vx_u8mf2(v0_out, 2, 2);

  /* Store 2x2 block to memory. */
  JSAMPROW outptr0 = output_buf[0] + output_col;
  JSAMPROW outptr1 = output_buf[1] + output_col;
  __riscv_vse8_v_u8mf2(outptr0, v0_out, 2);
  __riscv_vse8_v_u8mf2(outptr1, v1_out, 2);
}


/* jsimd_idct_4x4_rvv() is an inverse DCT function that produces reduced-size
 * 4x4 output from an 8x8 DCT block.  It uses the same calculations and
 * produces exactly the same output as IJG's original jpeg_idct_4x4() function
 * from jpeg-6b, which can be found in jidctred.c.
 *
 * Scaled integer constants are used to avoid floating-point arithmetic:
 *    0.211164243 =  1730 * 2^-13
 *    0.509795579 =  4176 * 2^-13
 *    0.601344887 =  4926 * 2^-13
 *    0.765366865 =  6270 * 2^-13
 *    0.899976223 =  7373 * 2^-13
 *    1.061594337 =  8697 * 2^-13
 *    1.451774981 = 11893 * 2^-13
 *    1.847759065 = 15137 * 2^-13
 *    2.172734803 = 17799 * 2^-13
 *    2.562915447 = 20995 * 2^-13
 *
 * See jidctred.c for further details of the 4x4 IDCT algorithm.  Where
 * possible, the variable names and comments here in jsimd_idct_4x4_rvv()
 * match up with those in jpeg_idct_4x4().
 */

static const int16_t jsimd_idct_4x4_consts[] = {
  F_1_847, -F_0_765, -F_0_211,  F_1_451,
 -F_2_172,  F_1_061, -F_0_509, -F_0_601,
  F_0_899,  F_2_562,        0,        0
};

void jsimd_idct_4x4_rvv(void *dct_table, JCOEFPTR coef_block,
                           JSAMPARRAY output_buf, JDIMENSION output_col)
{
  ptrdiff_t bstride;
  ISLOW_MULT_TYPE *quantptr = dct_table;
  vint16m4_t rows_0132_i16m4;

  /* Load DCT coefficients. */
  size_t vl = 8;
  vint16m1_t row0 = __riscv_vle16_v_i16m1(coef_block + 0 * DCTSIZE, vl);
  vint16m1_t row1 = __riscv_vle16_v_i16m1(coef_block + 1 * DCTSIZE, vl);
  vint16m1_t row2 = __riscv_vle16_v_i16m1(coef_block + 2 * DCTSIZE, vl);
  vint16m1_t row3 = __riscv_vle16_v_i16m1(coef_block + 3 * DCTSIZE, vl);
  vint16m1_t row5 = __riscv_vle16_v_i16m1(coef_block + 5 * DCTSIZE, vl);
  vint16m1_t row6 = __riscv_vle16_v_i16m1(coef_block + 6 * DCTSIZE, vl);
  vint16m1_t row7 = __riscv_vle16_v_i16m1(coef_block + 7 * DCTSIZE, vl);

  /* Load quantization table values for DC coefficients. */
  /* Dequantize DC coefficients. */
  vint16m1_t quant_row0 = __riscv_vle16_v_i16m1(quantptr + 0 * DCTSIZE, vl);
  row0 = __riscv_vmul_vv_i16m1(row0, quant_row0, vl);

  #define TRANS_TABLE_U8_SIZE 32
  const uint8_t trans_index_4x4_u8[TRANS_TABLE_U8_SIZE] =
  {
	0,  8, 24, 16,
	1,  9, 25, 17,
	2, 10, 26, 18,
	3, 11, 27, 19,
	5, 13, 29, 21,
	6, 14, 30, 22,
	7, 15, 31, 23,
	0,  0,  0,  0,
  };

  /* load transpose look-up table */
  vuint8m2_t vg_reg8 = __riscv_vle8_v_u8m2(trans_index_4x4_u8, TRANS_TABLE_U8_SIZE);

  /* Construct bitmap to test if all AC coefficients are 0. */
  vuint16m1_t vec_zero = __riscv_vmv_s_x_u16m1(0, vl);
  vint16m1_t bitmap = __riscv_vor_vv_i16m1(row7, row6, vl);
  bitmap = __riscv_vor_vv_i16m1(bitmap, row5, vl);
  bitmap = __riscv_vor_vv_i16m1(bitmap, row3, vl);
  bitmap = __riscv_vor_vv_i16m1(bitmap, row2, vl);
  bitmap = __riscv_vor_vv_i16m1(bitmap, row1, vl);
  vuint16m1_t tmp_u16m1 = __riscv_vredor_vs_u16m1_u16m1(__riscv_vreinterpret_v_i16m1_u16m1(bitmap), vec_zero, vl);
  uint16_t ac_bitmap = __riscv_vmv_x_s_u16m1_u16(tmp_u16m1);
  if (0 == ac_bitmap) {
    /* All AC coefficients are zero. */
    vint16m1_t dcval_i16m1 = __riscv_vsll_vx_i16m1(row0, PASS1_BITS, vl);

	/* combine vectors */
	rows_0132_i16m4 = __riscv_vlmul_ext_v_i16m1_i16m4(dcval_i16m1);
	rows_0132_i16m4 = __riscv_vslideup_vx_i16m4(rows_0132_i16m4, rows_0132_i16m4, DCTSIZE2/8, DCTSIZE2/4);
	rows_0132_i16m4 = __riscv_vslideup_vx_i16m4(rows_0132_i16m4, rows_0132_i16m4, DCTSIZE2/4, DCTSIZE2/2);
  } else {
    /* Load quantization table. */
    vint16m1_t quant_row1 = __riscv_vle16_v_i16m1(quantptr + 1 * DCTSIZE, vl);
    vint16m1_t quant_row2 = __riscv_vle16_v_i16m1(quantptr + 2 * DCTSIZE, vl);
    vint16m1_t quant_row3 = __riscv_vle16_v_i16m1(quantptr + 3 * DCTSIZE, vl);
    vint16m1_t quant_row5 = __riscv_vle16_v_i16m1(quantptr + 5 * DCTSIZE, vl);
    vint16m1_t quant_row6 = __riscv_vle16_v_i16m1(quantptr + 6 * DCTSIZE, vl);
    vint16m1_t quant_row7 = __riscv_vle16_v_i16m1(quantptr + 7 * DCTSIZE, vl);

    /* Even part */
	vint32m2_t tmp0, tmp2;
    tmp0 = __riscv_vwmul_vx_i32m2(row0, 16384, vl);	// 16384 = 2^(CONST_BITS + 1)
	row2 = __riscv_vmul_vv_i16m1(row2, quant_row2, vl);
	row6 = __riscv_vmul_vv_i16m1(row6, quant_row6, vl);
	tmp2 = __riscv_vwmul_vx_i32m2(row2, jsimd_idct_4x4_consts[0], vl);
	tmp2 = __riscv_vwmacc_vx_i32m2(tmp2, jsimd_idct_4x4_consts[1], row6, vl);
	vint32m2_t tmp10 = __riscv_vadd_vv_i32m2(tmp0, tmp2, vl);
	vint32m2_t tmp12 = __riscv_vsub_vv_i32m2(tmp0, tmp2, vl);

    /* Odd part */
	row7 = __riscv_vmul_vv_i16m1(row7, quant_row7, vl);
	row5 = __riscv_vmul_vv_i16m1(row5, quant_row5, vl);
	row3 = __riscv_vmul_vv_i16m1(row3, quant_row3, vl);
	row1 = __riscv_vmul_vv_i16m1(row1, quant_row1, vl);

	tmp0 = __riscv_vwmul_vx_i32m2 (row7, jsimd_idct_4x4_consts[2], vl);
	tmp0 = __riscv_vwmacc_vx_i32m2(tmp0, jsimd_idct_4x4_consts[3], row5, vl);
	tmp0 = __riscv_vwmacc_vx_i32m2(tmp0, jsimd_idct_4x4_consts[4], row3, vl);
	tmp0 = __riscv_vwmacc_vx_i32m2(tmp0, jsimd_idct_4x4_consts[5], row1, vl);

	tmp2 = __riscv_vwmul_vx_i32m2 (row7, jsimd_idct_4x4_consts[6], vl);
	tmp2 = __riscv_vwmacc_vx_i32m2(tmp2, jsimd_idct_4x4_consts[7], row5, vl);
	tmp2 = __riscv_vwmacc_vx_i32m2(tmp2, jsimd_idct_4x4_consts[8], row3, vl);
	tmp2 = __riscv_vwmacc_vx_i32m2(tmp2, jsimd_idct_4x4_consts[9], row1, vl);

    /* Final output stage */
    vint32m4_t tmp10_tmp12_i32m4 = __riscv_vlmul_ext_v_i32m2_i32m4(tmp10);
	tmp10_tmp12_i32m4 = __riscv_vslideup_vx_i32m4(tmp10_tmp12_i32m4, __riscv_vlmul_ext_v_i32m2_i32m4(tmp12), 8, 16);
    vint32m4_t tmp2_tmp0_i32m4 = __riscv_vlmul_ext_v_i32m2_i32m4(tmp2);
	tmp2_tmp0_i32m4 = __riscv_vslideup_vx_i32m4(tmp2_tmp0_i32m4, __riscv_vlmul_ext_v_i32m2_i32m4(tmp0), 8, 16);

	vint32m4_t tmp_add_i32m4 = __riscv_vadd_vv_i32m4(tmp10_tmp12_i32m4, tmp2_tmp0_i32m4, 16);
	vint32m4_t tmp_sub_i32m4 = __riscv_vsub_vv_i32m4(tmp10_tmp12_i32m4, tmp2_tmp0_i32m4, 16);
	tmp_add_i32m4 = __riscv_vadd_vx_i32m4(tmp_add_i32m4, 1<<(CONST_BITS - PASS1_BITS + 1 - 1), 16);
	tmp_sub_i32m4 = __riscv_vadd_vx_i32m4(tmp_sub_i32m4, 1<<(CONST_BITS - PASS1_BITS + 1 - 1), 16);
	vint16m2_t tmp_rows_01 = __riscv_vnsra_wx_i16m2(tmp_add_i32m4, (CONST_BITS - PASS1_BITS + 1), 16);
	vint16m2_t tmp_rows_32 = __riscv_vnsra_wx_i16m2(tmp_sub_i32m4, (CONST_BITS - PASS1_BITS + 1), 16);

	/* combine vectors */
	rows_0132_i16m4 = __riscv_vlmul_ext_v_i16m2_i16m4(tmp_rows_01);
	rows_0132_i16m4 = __riscv_vslideup_vx_i16m4(rows_0132_i16m4, __riscv_vlmul_ext_v_i16m2_i16m4(tmp_rows_32), 16, 32);
  }

  /* Transpose 8x4 block to perform IDCT on rows in second pass. */

  /* interpret to u16 & transpose */
  vint16m4_t vg_reg16 = __riscv_vrgather(rows_0132_i16m4, __riscv_vzext_vf2_u16m4(vg_reg8, 28), 28);

  /* extract columns */
  vl = 4;
  vint16m1_t col0 = __riscv_vget_v_i16m4_i16m1(vg_reg16, 0);
  vint16m1_t col1 = __riscv_vslidedown_vx_i16m1(col0, 4, vl);

  vint16m4_t slidedown_m4 = __riscv_vslidedown_vx_i16m4(vg_reg16, 8, 8);
  vint16m2_t tmp_i16m2 = __riscv_vlmul_trunc_v_i16m4_i16m2(slidedown_m4);
  vint16m1_t col2 = __riscv_vlmul_trunc_v_i16m2_i16m1(tmp_i16m2);
  tmp_i16m2 = __riscv_vslidedown_vx_i16m2(tmp_i16m2, 4, vl);
  vint16m1_t col3 = __riscv_vlmul_trunc_v_i16m2_i16m1(tmp_i16m2);

  slidedown_m4 = __riscv_vslidedown_vx_i16m4(vg_reg16, 16, 8);
  tmp_i16m2 = __riscv_vlmul_trunc_v_i16m4_i16m2(slidedown_m4);
  vint16m1_t col5 = __riscv_vlmul_trunc_v_i16m2_i16m1(tmp_i16m2);
  tmp_i16m2 = __riscv_vslidedown_vx_i16m2(tmp_i16m2, 4, vl);
  vint16m1_t col6 = __riscv_vlmul_trunc_v_i16m2_i16m1(tmp_i16m2);

  slidedown_m4 = __riscv_vslidedown_vx_i16m4(vg_reg16, 24, vl);
  vint16m1_t col7 = __riscv_vlmul_trunc_v_i16m4_i16m1(slidedown_m4);

  /* Commence second pass of IDCT. */

  /* Even part */
  vint32m2_t tmp0, tmp2, tmp10, tmp12;
  tmp0 = __riscv_vwmul_vx_i32m2(col0, 16384, vl);	// 2^(CONST_BITS + 1)
  tmp2 = __riscv_vwmul_vx_i32m2 (col2, jsimd_idct_4x4_consts[0], vl);
  tmp2 = __riscv_vwmacc_vx_i32m2(tmp2, jsimd_idct_4x4_consts[1], col6, vl);
  tmp10 = __riscv_vadd_vv_i32m2(tmp0, tmp2, vl);
  tmp12 = __riscv_vsub_vv_i32m2(tmp0, tmp2, vl);

  /* Odd part */
  tmp0 = __riscv_vwmul_vx_i32m2 (col7, jsimd_idct_4x4_consts[2], vl);
  tmp0 = __riscv_vwmacc_vx_i32m2(tmp0, jsimd_idct_4x4_consts[3], col5, vl);
  tmp0 = __riscv_vwmacc_vx_i32m2(tmp0, jsimd_idct_4x4_consts[4], col3, vl);
  tmp0 = __riscv_vwmacc_vx_i32m2(tmp0, jsimd_idct_4x4_consts[5], col1, vl);

  tmp2 = __riscv_vwmul_vx_i32m2 (col7, jsimd_idct_4x4_consts[6], vl);
  tmp2 = __riscv_vwmacc_vx_i32m2(tmp2, jsimd_idct_4x4_consts[7], col5, vl);
  tmp2 = __riscv_vwmacc_vx_i32m2(tmp2, jsimd_idct_4x4_consts[8], col3, vl);
  tmp2 = __riscv_vwmacc_vx_i32m2(tmp2, jsimd_idct_4x4_consts[9], col1, vl);

 /* Final output stage: descale and clamp to range [0-255]. */
  vuint8mf2_t u8mf2_col01, u8mf2_col23;
  vint8mf2_t tmp_i8mf2;
  vint16m1_t tmp_i16m1;
  vint32m2_t tmp_i32m2_0, tmp_i32m2_1;
  tmp_i32m2_0 = __riscv_vadd_vv_i32m2(tmp10, tmp2, 4);
  tmp_i32m2_1 = __riscv_vadd_vv_i32m2(tmp12, tmp0, 4);
  tmp_i32m2_0 = __riscv_vslideup_vx_i32m2(tmp_i32m2_0, tmp_i32m2_1, 4, 8);
  tmp_i16m1 = __riscv_vnsra_wx_i16m1(tmp_i32m2_0, 16, 8);
  tmp_i8mf2 = __riscv_vnclip_wx_i8mf2(tmp_i16m1, CONST_BITS + PASS1_BITS + 3 + 1 - 16, 8);
  u8mf2_col01 = __riscv_vreinterpret_v_i8mf2_u8mf2(tmp_i8mf2);
  u8mf2_col01 = __riscv_vadd_vx_u8mf2(u8mf2_col01, CENTERJSAMPLE, 8);

  tmp_i32m2_0 = __riscv_vsub_vv_i32m2(tmp12, tmp0, 4);
  tmp_i32m2_1 = __riscv_vsub_vv_i32m2(tmp10, tmp2, 4);
  tmp_i32m2_0 = __riscv_vslideup_vx_i32m2(tmp_i32m2_0, tmp_i32m2_1, 4, 8);
  tmp_i16m1 = __riscv_vnsra_wx_i16m1(tmp_i32m2_0, 16, 8);
  tmp_i8mf2 = __riscv_vnclip_wx_i8mf2(tmp_i16m1, CONST_BITS + PASS1_BITS + 3 + 1 - 16, 8);
  u8mf2_col23 = __riscv_vreinterpret_v_i8mf2_u8mf2(tmp_i8mf2);
  u8mf2_col23 = __riscv_vadd_vx_u8mf2(u8mf2_col23, CENTERJSAMPLE, 8);

  vuint8m1_t u8m1_col0123 = __riscv_vslideup_vx_u8m1(__riscv_vlmul_ext_v_u8mf2_u8m1(u8mf2_col01),
		  __riscv_vlmul_ext_v_u8mf2_u8m1(u8mf2_col23), 8, 16);

  /* Transpose */
  uint8_t out_index_tab[16] =
  {
	0,  4,  8, 12, // output transpose index order
	1,  5,  9, 13,
	2,  6, 10, 14,
	3,  7, 11, 15,
  };
  vuint8m1_t index_order_u8m1 = __riscv_vle8_v_u8m1(out_index_tab, 16);
  vuint8m1_t v0_out, v1_out, v2_out, v3_out;
  v0_out = __riscv_vrgather_vv_u8m1(u8m1_col0123, index_order_u8m1, 16);
  v1_out = __riscv_vslidedown_vx_u8m1(v0_out,  4, 4);
  v2_out = __riscv_vslidedown_vx_u8m1(v0_out,  8, 4);
  v3_out = __riscv_vslidedown_vx_u8m1(v0_out, 12, 4);

  /* Store 4x4 block to memory. */
  JSAMPROW outptr0 = output_buf[0] + output_col;
  JSAMPROW outptr1 = output_buf[1] + output_col;
  JSAMPROW outptr2 = output_buf[2] + output_col;
  JSAMPROW outptr3 = output_buf[3] + output_col;
  __riscv_vse8_v_u8m1(outptr0, v0_out, 4);
  __riscv_vse8_v_u8m1(outptr1, v1_out, 4);
  __riscv_vse8_v_u8m1(outptr2, v2_out, 4);
  __riscv_vse8_v_u8m1(outptr3, v3_out, 4);
}
