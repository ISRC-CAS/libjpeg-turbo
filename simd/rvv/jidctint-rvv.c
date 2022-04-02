/*
 * jidctint-rvv.c - accurate integer IDCT
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
#include "jsimd_rvv.h"


#define CONST_BITS  13
#define PASS1_BITS  2

#define F_0_298  2446           /* FIX(0.298631336) */
#define F_0_390  3196           /* FIX(0.390180644) */
#define F_0_541  4433           /* FIX(0.541196100) */
#define F_0_765  6270           /* FIX(0.765366865) */
#define F_0_899  7373           /* FIX(0.899976223) */
#define F_1_175  9633           /* FIX(1.175875602) */
#define F_1_501  12299          /* FIX(1.501321110) */
#define F_1_847  15137          /* FIX(1.847759065) */
#define F_1_961  16069          /* FIX(1.961570560) */
#define F_2_053  16819          /* FIX(2.053119869) */
#define F_2_562  20995          /* FIX(2.562915447) */
#define F_3_072  25172          /* FIX(3.072711026) */

#define ROUND_ADD(n)    (int32_t)1 << ((n) - 1)

#define DO_COMMON_IDCT(in) { \
    /* Even part */ \
    z1 = vadd_vv_i16m2(in##2, in##6, vl); \
    p1 = vwmul_vx_i32m4(z1, F_0_541, vl); \
    tmp2 = vwmul_vx_i32m4(in##6, -F_1_847, vl); \
    tmp2 = vadd_vv_i32m4(p1, tmp2, vl); \
    tmp3 = vwmul_vx_i32m4(in##2, F_0_765, vl); \
    tmp3 = vadd_vv_i32m4(p1, tmp3, vl); \
    \
    tmp0 = vwadd_vv_i32m4(in##0, in##4, vl); \
    tmp0 = vsll_vx_i32m4(tmp0, CONST_BITS, vl); \
    tmp1 = vwsub_vv_i32m4(in##0, in##4, vl); \
    tmp1 = vsll_vx_i32m4(tmp1, CONST_BITS, vl); \
    \
    tmp10 = vadd_vv_i32m4(tmp0, tmp3, vl);  \
    tmp13 = vsub_vv_i32m4(tmp0, tmp3, vl);  \
    tmp11 = vadd_vv_i32m4(tmp1, tmp2, vl);  \
    tmp12 = vsub_vv_i32m4(tmp1, tmp2, vl);  \
    \
    /* Odd Part */ \
    z1 = vadd_vv_i16m2(in##7, in##1, vl); \
    z2 = vadd_vv_i16m2(in##5, in##3, vl); \
    z3 = vadd_vv_i16m2(in##7, in##3, vl); \
    z4 = vadd_vv_i16m2(in##5, in##1, vl); \
    z5 = vadd_vv_i16m2(z3, z4, vl); \
    p5 = vwmul_vx_i32m4(z5, F_1_175, vl); \
     \
    tmp0 = vwmul_vx_i32m4(in##7, F_0_298, vl); \
    tmp1 = vwmul_vx_i32m4(in##5, F_2_053, vl); \
    tmp2 = vwmul_vx_i32m4(in##3, F_3_072, vl); \
    tmp3 = vwmul_vx_i32m4(in##1, F_1_501, vl); \
    p1 = vwmul_vx_i32m4(z1, -F_0_899, vl); \
    p2 = vwmul_vx_i32m4(z2, -F_2_562, vl); \
    p3 = vwmul_vx_i32m4(z3, -F_1_961, vl); \
    p4 = vwmul_vx_i32m4(z4, -F_0_390, vl); \
    \
    p3 = vadd_vv_i32m4(p3, p5, vl); \
    p4 = vadd_vv_i32m4(p4, p5, vl); \
    \
    tmp0 = vadd_vv_i32m4(tmp0, p1, vl); \
    tmp0 = vadd_vv_i32m4(tmp0, p3, vl); \
    tmp1 = vadd_vv_i32m4(tmp1, p2, vl); \
    tmp1 = vadd_vv_i32m4(tmp1, p4, vl); \
    tmp2 = vadd_vv_i32m4(tmp2, p2, vl); \
    tmp2 = vadd_vv_i32m4(tmp2, p3, vl); \
    tmp3 = vadd_vv_i32m4(tmp3, p1, vl); \
    tmp3 = vadd_vv_i32m4(tmp3, p4, vl); \
}


/* TODO: how many bits does a vector register has at least? 
         how to process const number(8 here) of 16-bit elements? */
void jsimd_idct_islow_rvv(void *dct_table, JCOEFPTR coef_block,
                          JSAMPARRAY output_buf, JDIMENSION output_col)
{
    ISLOW_MULT_TYPE *quantptr = dct_table;
    DCTELEM workspace[DCTSIZE2];

    vint8m1_t nrr0, nrr1, nrr2, nrr3, nrr4, nrr5, nrr6, nrr7;
    vuint8m1_t dst0, dst1, dst2, dst3, dst4, dst5, dst6, dst7;
    vint16m2_t row0, row1, row2, row3, row4, row5, row6, row7,
               col0, col1, col2, col3, col4, col5, col6, col7,
               z1, z2, z3, z4, z5,
               out0, out1, out2, out3, out4, out5, out6, out7,
               quant0, quant1, quant2, quant3, quant4, quant5, quant6, quant7;
    vint32m4_t out, p1, p2, p3, p4, p5,
               tmp0, tmp1, tmp2, tmp3, tmp4, 
               tmp10, tmp11, tmp12, tmp13;
    size_t vl = vsetvl_e16m2(DCTSIZE), 
           col_stride = DCTSIZE * sizeof(DCTELEM);

    /* Pass 1: process columns from input, store into work array. */
    /* Load DCT coefficients. */
    row0 = vle16_v_i16m2(coef_block + DCTSIZE * 0, vl);
    row1 = vle16_v_i16m2(coef_block + DCTSIZE * 1, vl);
    row2 = vle16_v_i16m2(coef_block + DCTSIZE * 2, vl);
    row3 = vle16_v_i16m2(coef_block + DCTSIZE * 3, vl);
    row4 = vle16_v_i16m2(coef_block + DCTSIZE * 4, vl);
    row5 = vle16_v_i16m2(coef_block + DCTSIZE * 5, vl);
    row6 = vle16_v_i16m2(coef_block + DCTSIZE * 6, vl);
    row7 = vle16_v_i16m2(coef_block + DCTSIZE * 7, vl);

    /* Load quantization table values for DC coefficients. */
    quant0 = vle16_v_i16m2(quantptr + 0 * DCTSIZE, vl);
    quant1 = vle16_v_i16m2(quantptr + 1 * DCTSIZE, vl);
    quant2 = vle16_v_i16m2(quantptr + 2 * DCTSIZE, vl);
    quant3 = vle16_v_i16m2(quantptr + 3 * DCTSIZE, vl);
    quant4 = vle16_v_i16m2(quantptr + 4 * DCTSIZE, vl);
    quant5 = vle16_v_i16m2(quantptr + 5 * DCTSIZE, vl);
    quant6 = vle16_v_i16m2(quantptr + 6 * DCTSIZE, vl);
    quant7 = vle16_v_i16m2(quantptr + 7 * DCTSIZE, vl);

    row0 = vmul_vv_i16m2(row0, quant0, vl);
    row1 = vmul_vv_i16m2(row1, quant1, vl);
    row2 = vmul_vv_i16m2(row2, quant2, vl);
    row3 = vmul_vv_i16m2(row3, quant3, vl);
    row4 = vmul_vv_i16m2(row4, quant4, vl);
    row5 = vmul_vv_i16m2(row5, quant5, vl);
    row6 = vmul_vv_i16m2(row6, quant6, vl);
    row7 = vmul_vv_i16m2(row7, quant7, vl);

    DO_COMMON_IDCT(row);

    out = vadd_vv_i32m4(tmp10, tmp3, vl);
    out = vadd_vx_i32m4(out, ROUND_ADD(CONST_BITS - PASS1_BITS), vl);
    out0 = vnsra_wx_i16m2(out, CONST_BITS - PASS1_BITS, vl);
    out = vsub_vv_i32m4(tmp10, tmp3, vl);
    out = vadd_vx_i32m4(out, ROUND_ADD(CONST_BITS - PASS1_BITS), vl);
    out7 = vnsra_wx_i16m2(out, CONST_BITS - PASS1_BITS, vl);
    out = vadd_vv_i32m4(tmp11, tmp2, vl);
    out = vadd_vx_i32m4(out, ROUND_ADD(CONST_BITS - PASS1_BITS), vl);
    out1 = vnsra_wx_i16m2(out, CONST_BITS - PASS1_BITS, vl);
    out = vsub_vv_i32m4(tmp11, tmp2, vl);
    out = vadd_vx_i32m4(out, ROUND_ADD(CONST_BITS - PASS1_BITS), vl);
    out6 = vnsra_wx_i16m2(out, CONST_BITS - PASS1_BITS, vl);
    out = vadd_vv_i32m4(tmp12, tmp1, vl);
    out = vadd_vx_i32m4(out, ROUND_ADD(CONST_BITS - PASS1_BITS), vl);
    out2 = vnsra_wx_i16m2(out, CONST_BITS - PASS1_BITS, vl);
    out = vsub_vv_i32m4(tmp12, tmp1, vl);
    out = vadd_vx_i32m4(out, ROUND_ADD(CONST_BITS - PASS1_BITS), vl);
    out5 = vnsra_wx_i16m2(out, CONST_BITS - PASS1_BITS, vl);
    out = vadd_vv_i32m4(tmp13, tmp0, vl);
    out = vadd_vx_i32m4(out, ROUND_ADD(CONST_BITS - PASS1_BITS), vl);
    out3 = vnsra_wx_i16m2(out, CONST_BITS - PASS1_BITS, vl);
    out = vadd_vv_i32m4(tmp13, tmp0, vl);
    out = vadd_vx_i32m4(out, ROUND_ADD(CONST_BITS - PASS1_BITS), vl);
    out4 = vnsra_wx_i16m2(out, CONST_BITS - PASS1_BITS, vl);

    /* Store rows */
    vse16_v_i16m2(workspace + DCTSIZE * 0, out0, vl);
    vse16_v_i16m2(workspace + DCTSIZE * 1, out1, vl);
    vse16_v_i16m2(workspace + DCTSIZE * 2, out2, vl);
    vse16_v_i16m2(workspace + DCTSIZE * 3, out3, vl);
    vse16_v_i16m2(workspace + DCTSIZE * 4, out4, vl);
    vse16_v_i16m2(workspace + DCTSIZE * 5, out5, vl);
    vse16_v_i16m2(workspace + DCTSIZE * 6, out6, vl);
    vse16_v_i16m2(workspace + DCTSIZE * 7, out7, vl);


    /* Pass 2: process rows from work array, store into output array. */
    /* Load columns */
    col0 = vlse16_v_i16m2(workspace + 0, col_stride, vl);
    col1 = vlse16_v_i16m2(workspace + 1, col_stride, vl);
    col2 = vlse16_v_i16m2(workspace + 2, col_stride, vl);
    col3 = vlse16_v_i16m2(workspace + 3, col_stride, vl);
    col4 = vlse16_v_i16m2(workspace + 4, col_stride, vl);
    col5 = vlse16_v_i16m2(workspace + 5, col_stride, vl);
    col6 = vlse16_v_i16m2(workspace + 6, col_stride, vl);
    col7 = vlse16_v_i16m2(workspace + 7, col_stride, vl);

    DO_COMMON_IDCT(col);

    out = vadd_vv_i32m4(tmp10, tmp3, vl);
    out = vadd_vx_i32m4(out, ROUND_ADD(CONST_BITS + PASS1_BITS + 3), vl);
    out0 = vnsra_wx_i16m2(out, CONST_BITS + PASS1_BITS + 3, vl);
    out = vsub_vv_i32m4(tmp10, tmp3, vl);
    out = vadd_vx_i32m4(out, ROUND_ADD(CONST_BITS + PASS1_BITS + 3), vl);
    out7 = vnsra_wx_i16m2(out, CONST_BITS + PASS1_BITS + 3, vl);
    out = vadd_vv_i32m4(tmp11, tmp2, vl);
    out = vadd_vx_i32m4(out, ROUND_ADD(CONST_BITS + PASS1_BITS + 3), vl);
    out1 = vnsra_wx_i16m2(out, CONST_BITS + PASS1_BITS + 3, vl);
    out = vsub_vv_i32m4(tmp11, tmp2, vl);
    out = vadd_vx_i32m4(out, ROUND_ADD(CONST_BITS + PASS1_BITS + 3), vl);
    out6 = vnsra_wx_i16m2(out, CONST_BITS + PASS1_BITS + 3, vl);
    out = vadd_vv_i32m4(tmp12, tmp1, vl);
    out = vadd_vx_i32m4(out, ROUND_ADD(CONST_BITS + PASS1_BITS + 3), vl);
    out2 = vnsra_wx_i16m2(out, CONST_BITS + PASS1_BITS + 3, vl);
    out = vsub_vv_i32m4(tmp12, tmp1, vl);
    out = vadd_vx_i32m4(out, ROUND_ADD(CONST_BITS + PASS1_BITS + 3), vl);
    out5 = vnsra_wx_i16m2(out, CONST_BITS + PASS1_BITS + 3, vl);
    out = vadd_vv_i32m4(tmp13, tmp0, vl);
    out = vadd_vx_i32m4(out, ROUND_ADD(CONST_BITS + PASS1_BITS + 3), vl);
    out3 = vnsra_wx_i16m2(out, CONST_BITS + PASS1_BITS + 3, vl);
    out = vadd_vv_i32m4(tmp13, tmp0, vl);
    out = vadd_vx_i32m4(out, ROUND_ADD(CONST_BITS + PASS1_BITS + 3), vl);
    out4 = vnsra_wx_i16m2(out, CONST_BITS + PASS1_BITS + 3, vl);


    /* Transpose matrix */
    /* Store columns */
    vsse16_v_i16m2(workspace + 0, col_stride, out0, vl);
    vsse16_v_i16m2(workspace + 1, col_stride, out1, vl);
    vsse16_v_i16m2(workspace + 2, col_stride, out2, vl);
    vsse16_v_i16m2(workspace + 3, col_stride, out3, vl);
    vsse16_v_i16m2(workspace + 4, col_stride, out4, vl);
    vsse16_v_i16m2(workspace + 5, col_stride, out5, vl);
    vsse16_v_i16m2(workspace + 6, col_stride, out6, vl);
    vsse16_v_i16m2(workspace + 7, col_stride, out7, vl);

    out0 = vle16_v_i16m2(workspace + DCTSIZE * 0, vl);
    out1 = vle16_v_i16m2(workspace + DCTSIZE * 1, vl);
    out2 = vle16_v_i16m2(workspace + DCTSIZE * 2, vl);
    out3 = vle16_v_i16m2(workspace + DCTSIZE * 3, vl);
    out4 = vle16_v_i16m2(workspace + DCTSIZE * 4, vl);
    out5 = vle16_v_i16m2(workspace + DCTSIZE * 5, vl);
    out6 = vle16_v_i16m2(workspace + DCTSIZE * 6, vl);
    out7 = vle16_v_i16m2(workspace + DCTSIZE * 7, vl);

#if BITS_IN_JSAMPLE == 8
    nrr0 = vnsra_wx_i8m1(out0, 0, vl);
    nrr1 = vnsra_wx_i8m1(out1, 0, vl);
    nrr2 = vnsra_wx_i8m1(out2, 0, vl);
    nrr3 = vnsra_wx_i8m1(out3, 0, vl);
    nrr4 = vnsra_wx_i8m1(out4, 0, vl);
    nrr5 = vnsra_wx_i8m1(out5, 0, vl);
    nrr6 = vnsra_wx_i8m1(out6, 0, vl);
    nrr7 = vnsra_wx_i8m1(out7, 0, vl);

    nrr0 = vadd_vx_i8m1(nrr0, CENTERJSAMPLE, vl);
    nrr1 = vadd_vx_i8m1(nrr1, CENTERJSAMPLE, vl);
    nrr2 = vadd_vx_i8m1(nrr2, CENTERJSAMPLE, vl);
    nrr3 = vadd_vx_i8m1(nrr3, CENTERJSAMPLE, vl);
    nrr4 = vadd_vx_i8m1(nrr4, CENTERJSAMPLE, vl);
    nrr5 = vadd_vx_i8m1(nrr5, CENTERJSAMPLE, vl);
    nrr6 = vadd_vx_i8m1(nrr6, CENTERJSAMPLE, vl);
    nrr7 = vadd_vx_i8m1(nrr7, CENTERJSAMPLE, vl);

    dst0 = vreinterpret_v_i8m1_u8m1(nrr0);
    dst1 = vreinterpret_v_i8m1_u8m1(nrr1);
    dst2 = vreinterpret_v_i8m1_u8m1(nrr2);
    dst3 = vreinterpret_v_i8m1_u8m1(nrr3);
    dst4 = vreinterpret_v_i8m1_u8m1(nrr4);
    dst5 = vreinterpret_v_i8m1_u8m1(nrr5);
    dst6 = vreinterpret_v_i8m1_u8m1(nrr6);
    dst7 = vreinterpret_v_i8m1_u8m1(nrr7);

    vse8_v_u8m1(output_buf[0] + output_col, dst0, vl);
    vse8_v_u8m1(output_buf[1] + output_col, dst1, vl);
    vse8_v_u8m1(output_buf[2] + output_col, dst2, vl);
    vse8_v_u8m1(output_buf[3] + output_col, dst3, vl);
    vse8_v_u8m1(output_buf[4] + output_col, dst4, vl);
    vse8_v_u8m1(output_buf[5] + output_col, dst5, vl);
    vse8_v_u8m1(output_buf[6] + output_col, dst6, vl);
    vse8_v_u8m1(output_buf[7] + output_col, dst7, vl);


#else
    out0 = vsra_vx_i16m2(out0, 0, vl);
    out1 = vsra_vx_i16m2(out1, 0, vl);
    out2 = vsra_vx_i16m2(out2, 0, vl);
    out3 = vsra_vx_i16m2(out3, 0, vl);
    out4 = vsra_vx_i16m2(out4, 0, vl);
    out5 = vsra_vx_i16m2(out5, 0, vl);
    out6 = vsra_vx_i16m2(out6, 0, vl);
    out7 = vsra_vx_i16m2(out7, 0, vl);

    out0 = vadd_vx_i16m2(out0, CENTERJSAMPLE, vl);
    out1 = vadd_vx_i16m2(out1, CENTERJSAMPLE, vl);
    out2 = vadd_vx_i16m2(out2, CENTERJSAMPLE, vl);
    out3 = vadd_vx_i16m2(out3, CENTERJSAMPLE, vl);
    out4 = vadd_vx_i16m2(out4, CENTERJSAMPLE, vl);
    out5 = vadd_vx_i16m2(out5, CENTERJSAMPLE, vl);
    out6 = vadd_vx_i16m2(out6, CENTERJSAMPLE, vl);
    out7 = vadd_vx_i16m2(out7, CENTERJSAMPLE, vl);

    vse16_v_i16m2(output_buf[0] + output_col, out0, vl);
    vse16_v_i16m2(output_buf[1] + output_col, out1, vl);
    vse16_v_i16m2(output_buf[2] + output_col, out2, vl);
    vse16_v_i16m2(output_buf[3] + output_col, out3, vl);
    vse16_v_i16m2(output_buf[4] + output_col, out4, vl);
    vse16_v_i16m2(output_buf[5] + output_col, out5, vl);
    vse16_v_i16m2(output_buf[6] + output_col, out6, vl);
    vse16_v_i16m2(output_buf[7] + output_col, out7, vl);
#endif
}
